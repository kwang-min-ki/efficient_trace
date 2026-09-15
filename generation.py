"""공통 HF 모델 로딩·응답 생성·필터링과 추론 접두사 KV 캐시 처리"""

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer

import model_config
import protocol
import reward


def _progress(label, done, total, t0):
    """처리 개수와 경과 시간 출력"""
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(f"[{label}] {done}/{total} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
          file=sys.stderr, flush=True)


def load_model(model, dtype="bfloat16", tokenizer=None, thinking=True):
    """HF 토크나이저·모델·설정 로딩

    체크포인트에 토크나이저가 없으면 tokenizer 인자로 별도 지정
    """
    tok = AutoTokenizer.from_pretrained(tokenizer or model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    profile = model_config.get_model_profile(model, tokenizer=tok, thinking=thinking)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(device)
    net.eval()
    return tok, net, profile


def _cut_at_stop(text, stop):
    """첫 중단 문자열부터 이후 텍스트 제거"""
    if not stop:
        return text
    idx = min((text.find(s) for s in stop if s in text), default=-1)
    return text[:idx] if idx >= 0 else text


class Generator:
    """원본 응답·라벨·모니터에 사용하는 공통 HF 생성기"""

    def __init__(self, model, dtype="bfloat16", batch_size=16, tokenizer=None,
                 thinking=True, max_model_len=None):
        """배치 크기 검증 후 모델·토크나이저와 입력 길이 한도 설정"""
        self.model = model
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_model_len is not None and max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.tok, self.net, self.profile = load_model(
            model, dtype, tokenizer=tokenizer, thinking=thinking)
        self.tok.padding_side = "left"  # 입력별 마지막 토큰 위치를 맞추기 위한 왼쪽 패딩

    def render(self, records):
        """모델 고유 대화 템플릿으로 입력 레코드를 문자열로 변환"""
        return model_config.render_records(self.tok, records, self.profile)

    @torch.inference_mode()
    def generate(self, prompts, n=1, temperature=0.7, max_tokens=1024, stop=None):
        """입력별 n개 응답을 배치 생성해 입력 순서의 중첩 리스트로 반환"""
        if not prompts:
            return []
        out = [None] * len(prompts)
        t0 = time.time()
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start:start + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            if self.max_model_len is not None:
                prompt_lengths = enc["attention_mask"].sum(dim=1)
                if (prompt_lengths + max_tokens > self.max_model_len).any():
                    raise ValueError("prompt plus requested output exceeds --max-model-len; "
                                     "use a shorter prompt/output or a larger supported limit")
            enc = {k: v.to(self.net.device) for k, v in enc.items()}
            kwargs = dict(**enc, max_new_tokens=max_tokens, num_return_sequences=n,
                          use_cache=True)
            # 체크포인트에서 지정한 모든 종료·패딩 ID 유지
            kwargs.update(model_config.sampling_kwargs(
                self.profile, temperature, backend="hf",
                tokenizer=self.tok, config_or_model=self.net))
            if stop:
                kwargs.update(stop_strings=stop, tokenizer=self.tok)
            seq = self.net.generate(**kwargs)
            gen_only = seq[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(gen_only, skip_special_tokens=True)
            for i in range(len(batch)):
                # 입력 i의 응답 n개는 [i*n:(i+1)*n]에 연속 배치
                out[start + i] = [_cut_at_stop(texts[i * n + j], stop) for j in range(n)]
            _progress("rollout", min(start + self.batch_size, len(prompts)), len(prompts), t0)
        return out


def rollout_and_filter(gen, samples, task, targets, temperature=protocol.ROLLOUT_TEMPERATURE,
                       source_records=None, progress_label="rollout"):
    """응답 생성 또는 재사용 후 보상 1·비어 있지 않은 추론만 선택

    source_records는 pid로 연결, 없는 ID는 missing_record로 집계
    반환 레코드에 답과 추론 직전의 정확한 접두사 포함
    동일 응답 비교를 위해 같은 모델·데이터 조건의 records 사용 필요
    """
    max_response = gen.profile.max_response_tokens(task)
    prompts = gen.render(samples)
    if source_records is None:
        rollouts = gen.generate(prompts, n=1,
                                temperature=temperature, max_tokens=max_response)
        responses = [out[0] for out in rollouts]
        source = "generated"
    else:
        by_pid = {}
        for record in source_records:
            pid = record.get("pid")
            if pid in by_pid:
                raise ValueError(f"duplicate pid in --records: {pid}")
            by_pid[pid] = record
        responses = [by_pid.get(s["pid"], {}).get("response") for s in samples]
        source = "records"

    # 추론 분리는 순차 처리, 자식 프로세스를 기다리는 코드 채점은 스레드로 병렬화
    parseable = []
    missing_record = no_reasoning = 0
    for s, prompt, response in zip(samples, prompts, responses):
        if not isinstance(response, str):
            missing_record += 1
            continue
        parts = model_config.split_reasoning_response(response, prompt=prompt)
        if parts is None or not parts.reasoning.strip():
            no_reasoning += 1
            continue
        parseable.append((s, response, parts))

    with ThreadPoolExecutor(max_workers=protocol.REWARD_WORKERS) as pool:
        rewards = pool.map(
            lambda item: reward.proxy(task, item[1], targets[item[0]["pid"]], item[0]["loophole"]),
            parseable,
        )

    kept = []
    incorrect = 0
    for (s, response, parts), r in zip(parseable, rewards):
        if r != 1.0:
            incorrect += 1
            continue
        answer_text = (
            reward.extract_answer(response)
            if task == "math"
            else reward.extract_code(response)
        )
        kept.append({"sample": s, "response": response, "cot": parts.reasoning,
                     "prefix_before_cot": parts.prefix_before_cot,
                     "answer_text": answer_text})
    print(f"[{progress_label}] source={source} inputs={len(samples)} "
          f"kept={len(kept)} missing_record={missing_record} "
          f"no_reasoning={no_reasoning} incorrect={incorrect}", flush=True)
    return kept



def cutoff_tok_len(n_cot_tokens, frac):
    """추론 토큰 수와 비율로 절단 위치 계산, 재토큰화 없이 캐시 위치 유지"""
    return max(1, math.ceil(n_cot_tokens * frac))


def clone_cache(cache):
    """절단점별 변경이 원본 캐시에 영향을 주지 않도록 텐서 복사"""
    new = DynamicCache()
    for layer in cache.layers:
        nl = DynamicLayer()
        nl.keys = layer.keys.clone()
        nl.values = layer.values.clone()
        nl.dtype = layer.dtype
        nl.device = layer.device
        nl.is_initialized = layer.is_initialized
        new.layers.append(nl)
    return new


def _build_prefix_cache(net, full_ids):
    """프롬프트와 전체 추론을 한 번 계산해 위치별 KV 캐시 생성"""
    cache = DynamicCache(config=net.config)
    net(
        input_ids=torch.tensor([full_ids], device=net.device),
        past_key_values=cache,
        use_cache=True,
    )
    return cache


def _longest_common_prefix_len(left, right):
    """두 토큰 ID 목록의 공통 접두사 길이 반환"""
    common = 0
    while common < len(left) and common < len(right) and left[common] == right[common]:
        common += 1
    return common


def _cutoff_attention_mask(target_lens, L, tail_len, device):
    """각 행의 절단점 이후 캐시를 가리고 새 입력은 유지하는 마스크 생성"""
    n = len(target_lens)
    mask = torch.zeros(n, L, dtype=torch.long, device=device)
    for j, tl in enumerate(target_lens):
        mask[j, :tl] = 1
    if tail_len:
        mask = torch.cat([mask, torch.ones(n, tail_len, dtype=torch.long, device=device)], dim=1)
    return mask


def _eos_ids(tok, net):
    """체크포인트 설정의 모든 종료 토큰 ID 반환"""
    return set(model_config.resolve_generation_token_ids(tok, net).eos)


def _filter_logits(logits, sampling):
    """온도로 조정된 logits에 top-k, top-p, min-p 순서로 필터 적용

    직접 구현한 캐시 생성에서도 HF 생성과 같은 필터 설정 사용
    """
    if sampling is None:
        return logits
    if sampling.top_k and sampling.top_k > 0:
        k = min(int(sampling.top_k), logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if sampling.top_p is not None and sampling.top_p < 1.0:
        ordered, order = torch.sort(logits, dim=-1, descending=True)
        probs = ordered.softmax(dim=-1)
        # top-p를 넘는 누적 확률의 꼬리 제거, 최상위 토큰은 항상 유지
        drop = (probs.cumsum(dim=-1) - probs) > sampling.top_p
        drop[:, 0] = False
        logits = logits.masked_fill(drop.scatter(1, order, drop), float("-inf"))
    if sampling.min_p and sampling.min_p > 0:
        probs = logits.softmax(dim=-1)
        floor = sampling.min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < floor, float("-inf"))
    return logits


@torch.inference_mode()
def _decode_from_cache_masked(tok, net, branch, attn_mask_base, start_ids, start_positions,
                              max_new_tokens, temperature, stop, sampling=None):
    """마스크된 접두사 캐시에서 배치별 답 토큰 생성

    캐시는 자르지 않고 attention_mask와 position_ids로 절단 위치 지정
    각 행을 함께 생성하며 중단 문자열을 제외한 문자열 목록 반환
    """
    B = start_ids.shape[0]
    F = start_ids.shape[1]
    device = net.device
    eos = _eos_ids(tok, net)
    generated = [[] for _ in range(B)]
    finished = [False] * B
    do_sample = bool(temperature and temperature > 0)
    # 중단 문자열 검사를 최근 16개 토큰으로 제한해 반복 디코딩 비용 절감
    WINDOW = 16

    attn_mask = attn_mask_base
    cur_input = start_ids
    cur_pos = start_positions.unsqueeze(1) + torch.arange(F, device=device).unsqueeze(0)
    next_pos = start_positions + F  # 각 행에서 다음에 생성할 토큰의 절대 위치

    for _ in range(max_new_tokens):
        attn_mask = torch.cat(
            [
                attn_mask,
                torch.ones(
                    B, cur_input.shape[1], dtype=attn_mask.dtype, device=device
                ),
            ],
            dim=1,
        )
        logits = net(input_ids=cur_input, past_key_values=branch, attention_mask=attn_mask,
                     position_ids=cur_pos, use_cache=True).logits[:, -1, :]
        if do_sample:
            scaled = _filter_logits(logits.float() / temperature, sampling)
            probs = torch.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tok = logits.argmax(dim=-1)
        for b in range(B):
            if not finished[b]:
                generated[b].append(next_tok[b].item())
                if next_tok[b].item() in eos:
                    finished[b] = True
                elif stop and any(
                    s in tok.decode(generated[b][-WINDOW:], skip_special_tokens=True)
                    for s in stop
                ):
                    finished[b] = True
        if all(finished):
            break
        cur_input = next_tok.unsqueeze(-1)
        cur_pos = next_pos.unsqueeze(1)
        next_pos = next_pos + 1
    return [_cut_at_stop(tok.decode(g, skip_special_tokens=True), stop) for g in generated]


def trace_curve_cached(tok, net, prompt_ids, cot_ids, force_ids, fracs, n_samples, temperature,
                       max_new_tokens, stop, sampling=None):
    """전체 추론 캐시를 공유해 모든 절단점의 답을 배치 생성

    배치 크기는 len(fracs) * n_samples, 답 생성은 토큰별 순차 진행
    fracs 순서로 절단점별 n_samples개 답 반환
    """
    full_ids = prompt_ids + cot_ids
    L = len(full_ids)
    n_cut = len(fracs)
    B = n_cut * n_samples
    device = net.device

    cache = _build_prefix_cache(net, full_ids)
    branch = clone_cache(cache)
    branch.batch_repeat_interleave(B)  # 모든 배치 행에 독립적인 원본 캐시 복사본 사용

    target_lens = [len(prompt_ids) + cutoff_tok_len(len(cot_ids), f) for f in fracs]
    row_target_lens = [target_lens[i // n_samples] for i in range(B)]
    F = len(force_ids)
    attn_mask_base = _cutoff_attention_mask(row_target_lens, L, 0, device)
    start_ids = torch.tensor([force_ids] * B, device=device)
    start_positions = torch.tensor(row_target_lens, device=device)

    texts = _decode_from_cache_masked(tok, net, branch, attn_mask_base, start_ids, start_positions,
                                      max_new_tokens, temperature, stop, sampling=sampling)
    return [texts[j * n_samples:(j + 1) * n_samples] for j in range(n_cut)]



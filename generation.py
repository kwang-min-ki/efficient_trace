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
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(f"[{label}] {done}/{total} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
          file=sys.stderr, flush=True)


def load_model(model, dtype="bfloat16", tokenizer=None, thinking=True):
    """tokenizer: override for checkpoints pushed without their own tokenizer files --
    fine-tuning doesn't change the vocab, so pointing this at the base model is safe.

    Also resolves the model profile from the checkpoint's own config (never its name).
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
    """HF's StopStringCriteria stops only once a token completing the stop string has
    been generated, so the decoded text includes it (and can overshoot it, e.g. a token
    like "stopper" fulfilling "stop"). Truncate here so REOPEN[task] + o
    reconstructs the text expected by reward parsing."""
    if not stop:
        return text
    idx = min((text.find(s) for s in stop if s in text), default=-1)
    return text[:idx] if idx >= 0 else text


class Generator:
    """HF generation with `.tok`, `.model` (name string, for record
    metadata), `.generate(prompts, n, temperature, max_tokens, stop) -> list[list[str]]`.
    The actual `PreTrainedModel` lives on `.net` so it doesn't collide with `.model`.
    Used for source rollouts, counterfactual labels, and CoT monitoring."""

    def __init__(self, model, dtype="bfloat16", batch_size=16, tokenizer=None,
                 thinking=True, max_model_len=None):
        self.model = model
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_model_len is not None and max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.tok, self.net, self.profile = load_model(
            model, dtype, tokenizer=tokenizer, thinking=thinking)
        self.tok.padding_side = "left"  # required so every row's next token is in the same column

    def render(self, records):
        """Dataset records -> model input strings, via this checkpoint's template."""
        return model_config.render_records(self.tok, records, self.profile)

    @torch.inference_mode()
    def generate(self, prompts, n=1, temperature=0.7, max_tokens=1024, stop=None):
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
            # Preserve all EOS/pad IDs from the checkpoint's generation_config.
            kwargs.update(model_config.sampling_kwargs(
                self.profile, temperature, backend="hf",
                tokenizer=self.tok, config_or_model=self.net))
            if stop:
                kwargs.update(stop_strings=stop, tokenizer=self.tok)
            seq = self.net.generate(**kwargs)
            gen_only = seq[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(gen_only, skip_special_tokens=True)
            for i in range(len(batch)):
                # repeat_interleave order: row i's n samples are contiguous at [i*n:(i+1)*n]
                out[start + i] = [_cut_at_stop(texts[i * n + j], stop) for j in range(n)]
            _progress("rollout", min(start + self.batch_size, len(prompts)), len(prompts), t0)
        return out


def rollout_and_filter(gen, samples, task, targets, temperature=protocol.ROLLOUT_TEMPERATURE,
                       source_records=None, progress_label="rollout"):
    """Prepare `kept` responses (generate one rollout per sample,
    keep only responses that both have a non-empty CoT and obtain proxy reward 1.0; Sec.
    4.1: "only responses that obtain a reward of 1 are scored"). Also extracts
    `answer_text`, the string likelihood_trace.py teacher-forces at every cutoff.

    `prefix_before_cot` is the exact text preceding the reasoning span: the rendered
    prompt, plus any generated `<think>` marker when there is one. Both HF
    scorers tokenize that -- not the raw dataset prompt -- so the cutoff denominator is
    reasoning tokens only.

    `source_records` reuses previously written responses (matched by pid) instead of
    generating new ones. Two reasons it matters
    for math/code: the rollout is 92-98% of a run's wall clock while scoring is 20-75s,
    and it is *sampled* (ROLLOUT_TEMPERATURE=0.7), so re-generating per --aggregation
    both pays that cost again and lands on a different `kept` population -- comparisons
    across aggregations then mix the aggregation's effect with sampling noise. Reusing
    records keeps the sampled-rollout protocol intact while making those comparisons
    exactly paired. The caller must pass the same `samples` the records came from;
    pids present in `samples` but absent from the records are reported as
    `missing_record` rather than silently dropped.
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

    # Reasoning-span parsing is cheap and sequential; reward.proxy for code shells out
    # to run_tests's subprocess.run, which blocks on the child process and releases the
    # GIL, so a thread pool parallelizes it across samples (see protocol.REWARD_WORKERS).
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


# ---------------------------------------------------------------------------
# Hand-rolled KV-prefix caching shared by trace.py and likelihood_trace.py.
# ---------------------------------------------------------------------------

def cutoff_tok_len(n_cot_tokens, frac):
    """Compute each token cutoff directly
    instead of encode->slice->decode->(re-encode later) -- this guarantees each
    cutoff's tokens are an exact prefix of the next-longer cutoff's tokens, which is
    what makes reusing one KV cache across cutoffs valid at all."""
    return max(1, math.ceil(n_cot_tokens * frac))


def clone_cache(cache):
    """Copy mutable layer tensors so cutoff branches cannot alter the base cache."""
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
    """One forward pass over prompt+full_cot, filling a KV cache for every position."""
    cache = DynamicCache(config=net.config)
    net(
        input_ids=torch.tensor([full_ids], device=net.device),
        past_key_values=cache,
        use_cache=True,
    )
    return cache


def _longest_common_prefix_len(left, right):
    """Number of leading token IDs shared by two independently tokenized texts."""
    common = 0
    while common < len(left) and common < len(right) and left[common] == right[common]:
        common += 1
    return common


def _cutoff_attention_mask(target_lens, L, tail_len, device):
    """(len(target_lens), L + tail_len) mask: row j sees cached positions
    [0, target_lens[j]) and all of the tail_len new positions appended after L.
    Row j's target_lens[j] <= L always (cutoff_tok_len never exceeds the CoT length),
    so this only ever hides a *suffix* of the cached prefix, never the tail."""
    n = len(target_lens)
    mask = torch.zeros(n, L, dtype=torch.long, device=device)
    for j, tl in enumerate(target_lens):
        mask[j, :tl] = 1
    if tail_len:
        mask = torch.cat([mask, torch.ones(n, tail_len, dtype=torch.long, device=device)], dim=1)
    return mask


def _eos_ids(tok, net):
    """Every stop token this checkpoint declares."""
    return set(model_config.resolve_generation_token_ids(tok, net).eos)


def _filter_logits(logits, sampling):
    """Apply top-k / top-p / min-p to already-temperature-scaled logits.

    `model.generate` applies these through LogitsProcessors, but this decoder drives
    `forward()` by hand (see the docstring below), so the same filtering has to be
    applied here or the two HF paths would silently sample from different
    distributions. Order matches transformers' default warper order: top-k, then
    top-p, then min-p. With the experiment protocol's settings (top_k=0, top_p=1.0,
    min_p=0.0) every branch is a no-op, so this is exact for the current protocol and
    correct if the profile ever changes.
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
        # Drop the tail whose cumulative mass *before* this token already covers top_p,
        # always keeping the single most likely token.
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
    """Hand-rolled autoregressive decode against a pre-filled, masked `branch` cache
    (batch B): row b starts at absolute position `start_positions[b]` (its cutoff
    boundary) rather than after the full cache, via explicit `position_ids` -- the
    cache tensor itself is never cropped, `attn_mask_base` (B, L) is what makes each
    row only "see" cached positions [0, start_positions[b]). Every decode step
    advances all B rows together in one forward call (instead of trace_curve_cached
    looping per cutoff), so the B = len(fracs)*n_samples rows here typically cover
    every cutoff of one sample at once. We manage `attention_mask`/`position_ids` by
    hand (not `model.generate(past_key_values=...)`) because that path mishandles a
    pre-filled cache combined with a short new attention_mask (confirmed by a live
    crash: 0-length reshape a few decode steps in) -- forward() is the primitive
    already proven correct for pre-filled caches (likelihood_curve_cached).
    Returns a list of B decoded strings, stop text excluded.
    """
    B = start_ids.shape[0]
    F = start_ids.shape[1]
    device = net.device
    eos = _eos_ids(tok, net)
    generated = [[] for _ in range(B)]
    finished = [False] * B
    do_sample = bool(temperature and temperature > 0)
    # Decoding only the last WINDOW tokens each step
    # (instead of the whole, ever-growing `generated[b]`) keeps the per-step
    # stop-check O(1) instead of O(step).
    WINDOW = 16

    attn_mask = attn_mask_base
    cur_input = start_ids
    cur_pos = start_positions.unsqueeze(1) + torch.arange(F, device=device).unsqueeze(0)
    next_pos = start_positions + F  # absolute position of the next token to generate, per row

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
    """TRACE for one sample: 1 forward pass builds a KV cache over prompt+full_cot,
    then ALL len(fracs) cutoffs are decoded together in ONE batched loop (batch =
    len(fracs) * n_samples; row i belongs to cutoff i // n_samples), using the same
    masking approach as likelihood_curve_cached, instead of looping over cutoffs and
    cropping the cache for each. Autoregressive decoding itself still needs up to
    max_new_tokens sequential forward calls -- that dependency can't be removed, it's
    the actual TRACE-vs-likelihood-TRACE difference this reimplements -- but all
    len(fracs)*n_samples rows now advance together each step instead of len(fracs)
    separate small-batch loops, removing the same per-cutoff call overhead
    likelihood_curve_cached's batching removes.
    Returns a list of per-cutoff list-of-n_samples decoded strings, in `fracs` order.
    """
    full_ids = prompt_ids + cot_ids
    L = len(full_ids)
    n_cut = len(fracs)
    B = n_cut * n_samples
    device = net.device

    cache = _build_prefix_cache(net, full_ids)
    branch = clone_cache(cache)
    branch.batch_repeat_interleave(B)  # all B rows start as identical copies of the base cache

    target_lens = [len(prompt_ids) + cutoff_tok_len(len(cot_ids), f) for f in fracs]
    row_target_lens = [target_lens[i // n_samples] for i in range(B)]
    F = len(force_ids)
    attn_mask_base = _cutoff_attention_mask(row_target_lens, L, 0, device)
    start_ids = torch.tensor([force_ids] * B, device=device)
    start_positions = torch.tensor(row_target_lens, device=device)

    texts = _decode_from_cache_masked(tok, net, branch, attn_mask_base, start_ids, start_positions,
                                      max_new_tokens, temperature, stop, sampling=sampling)
    return [texts[j * n_samples:(j + 1) * n_samples] for j in range(n_cut)]



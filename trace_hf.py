"""Transformers-backed runtime, KV-cache machinery, and TRACE CLI.

Both scripts reimplement trace.py's pipeline on `transformers` instead of vLLM: vLLM's
`prompt_logprobs` invalidates its own prefix cache and ends up slower than a bare HF
forward pass for the likelihood-scoring variant (confirmed empirically). trace_hf.py
keeps trace.py's exact algorithm -- re-decode an answer at every cutoff -- but, like
likelihood_trace_hf.py, reuses the SAME KV cache across all 10 cutoffs of one sample,
since they are nested prefixes of the same CoT: the shared prompt+CoT prefix is
forwarded through the model exactly once, then ALL 10 cutoffs are processed in a
SINGLE additional batched call (batch = 10, or 10*n_samples for TRACE's decoding)
instead of ten small sequential ones. The cache is replicated once per cutoff but
never physically cropped; each row's `attention_mask` instead hides cached positions
past that row's own cutoff, and `position_ids` places that row's FORCE tag [+answer]
right after its own cutoff boundary. This is equivalent to cropping-then-forwarding
per cutoff (HF's causal mask + attention_mask combination doesn't care where in the
sequence the masked-out positions sit) but removes 9 of 10 per-cutoff forward-call
overheads (kernel launch, Python loop) -- both methods get the same prefix-caching
AND the same cutoff-batching, so the only algorithmic difference left is TRACE's
per-cutoff autoregressive decoding vs likelihood-TRACE's per-cutoff teacher-forced
single forward pass, which is the actual comparison this reimplements.

AR-LSAT is the exact-text exception: its cumulative word contexts can retokenize
at every boundary. Both AR-LSAT methods therefore use `_exact_prefix_caches` to
crop a master DynamicCache to the token-ID longest common prefix and forward only
the changed suffix; TRACE then branches that exact cache K ways while
Likelihood-TRACE teacher-forces its answer.

`HFGenerator` duck-types trace.Generator (`.tok`, `.model`, `.generate(prompts, n,
temperature, max_tokens, stop)`) and is used only for the one-shot rollout generation
(Sec 4.1) -- that step has no cutoffs to batch or a cache to reuse, so it's just
ordinary batched `model.generate()`.

API points below were confirmed against transformers docs/source, not guessed:
- `generate(..., stop_strings=[...], tokenizer=tok)` is required (tokenizer is needed to
  turn stop strings into token boundaries; generation/stopping_criteria.py,
  StopStringCriteria).
- `num_return_sequences=n` expands the batch via `repeat_interleave`
  (generation/utils.py, `_expand_inputs_for_generation`), so the n outputs for input row
  i land contiguously at [i*n : (i+1)*n] -- not interleaved across rows.
- `top_k` defaults to 50 in GenerationConfig (unlike vLLM's SamplingParams, which
  defaults top_k to disabled) and `top_k=0` is the documented way to disable it
  (generation/utils.py: `if generation_config.top_k is not None and
  generation_config.top_k != 0`). We set it explicitly so HF sampling matches trace.py's
  original (vLLM) sampling distribution instead of silently top-k-truncating it.
- `DynamicLayer` stores exactly `.keys`, `.values`, `.dtype`, `.device`,
  `.is_initialized` (cache_utils.py, `CacheLayerMixin.__init__` /
  `DynamicLayer.lazy_initialization`) -- `clone_cache` below copies precisely those
  fields rather than relying on `copy.deepcopy` on the whole `Cache` object, which also
  carries a `config` reference not worth copying every cutoff.
- `Cache.batch_repeat_interleave(n)` repeats `.keys`/`.values` along dim 0
  (cache_utils.py, `DynamicLayer.batch_repeat_interleave`) -- used to branch one cached
  prefix into `len(fracs)` (or `len(fracs)*n_samples`) independent masked rows.
- A 2D `attention_mask` combined with `past_key_values` and explicit `position_ids` is
  the same mechanism standard batched generation uses for left-padded, differently-
  started sequences (generation/utils.py, `_prepare_attention_mask_for_generation` /
  `_update_model_kwargs_for_generation`); it does not require the masked-out positions
  to be at the start or end of the sequence, only that attention_mask==1 marks which
  cached positions a given row may attend to and position_ids gives each new token's
  true RoPE position -- which is what lets us hide a per-row *suffix* of the cache
  instead of the more common left-padding prefix.
"""

import argparse
import json
import math
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer

import model_config
import reward
import trace
from data import read_jsonl, targets_for, write_jsonl
from arlsat import rollout_and_filter_arlsat


def _progress(label, done, total, t0):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(f"[{label}] {done}/{total} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
          file=sys.stderr, flush=True)


def load_model(model, dtype="bfloat16", tokenizer=None, thinking=True):
    """tokenizer: override for checkpoints pushed without their own tokenizer files --
    fine-tuning doesn't change the vocab, so pointing this at the base model is safe.

    Also resolves the model profile from the checkpoint's own config (never its name),
    validating that a Qwen3 tokenizer really owns the <think>/</think> markers the
    cutoff protocol depends on.
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
    like "stopper" fulfilling "stop"). vLLM's default (used by trace.py) excludes the
    stop string. Truncate here so REOPEN[task] + o reconstructs the same shape of text
    trace.py's reward parsing expects."""
    if not stop:
        return text
    idx = min((text.find(s) for s in stop if s in text), default=-1)
    return text[:idx] if idx >= 0 else text


class HFGenerator:
    """Same interface as trace.Generator: `.tok`, `.model` (name string, for record
    metadata), `.generate(prompts, n, temperature, max_tokens, stop) -> list[list[str]]`.
    The actual `PreTrainedModel` lives on `.net` so it doesn't collide with `.model`.
    Used only for the uncached, one-shot rollout generation -- see module docstring."""

    def __init__(self, model, dtype="bfloat16", batch_size=16, tokenizer=None,
                 thinking=True):
        self.model = model
        self.batch_size = batch_size
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
            enc = {k: v.to(self.net.device) for k, v in enc.items()}
            kwargs = dict(**enc, max_new_tokens=max_tokens, num_return_sequences=n,
                          use_cache=True)
            # EOS/pad come from the checkpoint's generation_config, not a literal: Qwen3
            # stops on <|im_end|> AND <|endoftext|>, which tok.eos_token_id alone misses.
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


def rollout_and_filter(gen, samples, task, targets, temperature=trace.ROLLOUT_TEMPERATURE,
                       source_records=None, progress_label="rollout"):
    """Mirrors trace.score()'s lines that build `kept` (generate one rollout per sample,
    keep only responses that both have a non-empty CoT and obtain proxy reward 1.0; Sec.
    4.1: "only responses that obtain a reward of 1 are scored"). Also extracts
    `answer_text`, the string likelihood_trace_hf.py teacher-forces at every cutoff.

    `prefix_before_cot` is the exact text preceding the reasoning span: the rendered
    prompt, plus Qwen3's own generated `<think>` marker when there is one. Both HF
    scorers tokenize that -- not the raw dataset prompt -- so the cutoff denominator is
    reasoning tokens only, matching trace.py.

    `source_records` reuses previously written responses (matched by pid) instead of
    generating new ones, mirroring the AR-LSAT path's --records. Two reasons it matters
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

    kept = []
    missing_record = no_reasoning = incorrect = 0
    for s, prompt, response in zip(samples, prompts, responses):
        if not isinstance(response, str):
            missing_record += 1
            continue
        parts = model_config.split_reasoning_response(response, prompt=prompt)
        if parts is None or not parts.reasoning.strip():
            no_reasoning += 1
            continue
        if reward.proxy(task, response, targets[s["pid"]], s["loophole"]) != 1.0:
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
# Hand-rolled KV-prefix caching shared by trace_hf.py and likelihood_trace_hf.py.
# ---------------------------------------------------------------------------

def cutoff_tok_len(n_cot_tokens, frac):
    """Same slice-point math as trace.truncate(), but on a token count directly
    instead of encode->slice->decode->(re-encode later) -- this guarantees each
    cutoff's tokens are an exact prefix of the next-longer cutoff's tokens, which is
    what makes reusing one KV cache across cutoffs valid at all."""
    return max(1, math.ceil(n_cot_tokens * frac))


def clone_cache(cache):
    """Copy a DynamicCache's tensors (see module docstring for why not copy.deepcopy)."""
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


def _exact_prefix_caches(tok, net, prefix_texts):
    """Yield isolated exact-prefix caches while reusing their shared token prefix.

    AR-LSAT cutoffs are exact cumulative word prefixes. Their independently
    tokenized ID sequences are usually almost, but not literally, nested because
    BPE can retokenize the token at the previous word boundary. Keep a master cache
    for the previous exact prefix, roll it back to the token-ID longest common
    prefix, and forward only the new suffix. The yielded clone may then be mutated
    by a likelihood tail or repeated into sampling branches without corrupting the
    master cache used by the next cutoff.

    ``DynamicLayer.crop(0)`` means "remove zero tokens" rather than "crop to zero"
    in transformers, so a zero-length common prefix must rebuild the cache.
    """
    cached_ids = None
    cache = None
    for prefix_text in prefix_texts:
        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        if not prefix_ids:
            raise ValueError("exact prefix must contain at least one token")

        if cache is None:
            cache = _build_prefix_cache(net, prefix_ids)
        else:
            common = _longest_common_prefix_len(cached_ids, prefix_ids)
            if common == 0:
                cache = _build_prefix_cache(net, prefix_ids)
            else:
                cache.crop(common)
                suffix_ids = prefix_ids[common:]
                if suffix_ids:
                    net(input_ids=torch.tensor([suffix_ids], device=net.device),
                        past_key_values=cache, use_cache=True)
        cached_ids = prefix_ids
        yield clone_cache(cache), len(prefix_ids)


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
    """Every stop token this checkpoint declares (Qwen3 declares two)."""
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
    # See trace_hf's earlier fix: decoding only the last WINDOW tokens each step
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

@torch.inference_mode()
def trace_curve_exact_prefixes(tok, net, prefix_texts, force_text, n_samples, temperature,
                               max_new_tokens, stop, sampling=None):
    """Sample TRACE answers from exact text prefixes with incremental KV reuse.

    This is the autoregressive counterpart of
    :func:`likelihood_curve_exact_prefixes` for protocols such as AR-LSAT whose
    word-ratio cutoffs must be tokenized independently. One master cache is updated
    between adjacent exact prefixes by `_exact_prefix_caches`; its isolated clone is
    repeated into ``n_samples`` rows, and all K rows for that cutoff decode together.
    Returns one list of K decoded strings per prefix, in input order.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    force_ids = tok.encode(force_text, add_special_tokens=False)
    if not force_ids:
        raise ValueError("force_text must contain at least one token")

    device = net.device
    result = []
    for branch, prefix_len in _exact_prefix_caches(tok, net, prefix_texts):
        branch.batch_repeat_interleave(n_samples)
        attn_mask_base = torch.ones(
            n_samples, prefix_len, dtype=torch.long, device=device)
        start_ids = torch.tensor([force_ids] * n_samples, device=device)
        start_positions = torch.full(
            (n_samples,), prefix_len, dtype=torch.long, device=device)
        result.append(_decode_from_cache_masked(
            tok, net, branch, attn_mask_base, start_ids, start_positions,
            max_new_tokens, temperature, stop, sampling=sampling))
    return result


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


# ---------------------------------------------------------------------------
# Math/code autoregressive TRACE orchestration and CLI.
# ---------------------------------------------------------------------------

def score(gen, samples, task, targets):
    config = trace.TASK_CFG[task]
    n_samples, temp, ans_tokens = (
        config["n_samples"], config["temp"], config["ans_tokens"]
    )

    t0 = time.time()
    kept = rollout_and_filter(gen, samples, task, targets,
                              temperature=trace.ROLLOUT_TEMPERATURE)
    rollout_time = time.time() - t0

    force_ids = gen.tok.encode(trace.FORCE[task], add_special_tokens=False)

    t1 = time.time()
    records = []
    for n, k in enumerate(kept, 1):
        s = k["sample"]
        # prefix_before_cot, not the dataset prompt: it carries the rendered chat
        # template and Qwen3's generated <think>, so cot_ids is reasoning only.
        prompt_ids = gen.tok.encode(k["prefix_before_cot"], add_special_tokens=False)
        cot_ids = gen.tok.encode(k["cot"], add_special_tokens=False)
        per_cutoff_outs = trace_curve_cached(
            gen.tok,
            gen.net,
            prompt_ids,
            cot_ids,
            force_ids,
            trace.FRACS,
            n_samples,
            temp,
            ans_tokens,
            trace.STOP[task],
            sampling=gen.profile.sampling,
        )
        curve = [0.0] * len(trace.FRACS)
        for j, outs in enumerate(per_cutoff_outs):
            texts = [trace.REOPEN[task] + o for o in outs]
            curve[j] = reward.expected(task, texts, targets[s["pid"]], s["loophole"])
        records.append({
            "pid": s["pid"],
            "task": task,
            "variant": s["variant"],
            "model": gen.model,
            "curve": curve,
            "auc": trace.auc(curve),
            "response": k["response"],
            "impl": "trace_hf",
            "score_window": None,
        })
        if n % 50 == 0 or n == len(kept):
            print(f"[score] {n}/{len(kept)} ({time.time() - t1:.0f}s elapsed)", flush=True)
    scoring_time = time.time() - t1
    return records, rollout_time, scoring_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["math", "code"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="ic_correct")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val",
                    help="comma-separated split names, e.g. train,val,heldout")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16, help="rollout-generation batch size only")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tokenizer",
                    help="override if --model's checkpoint wasn't pushed with its own "
                         "tokenizer files, e.g. --tokenizer /workspace/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch.manual_seed before the rollout generation, so a "
                         "likelihood_trace_hf.py run with the same --seed/--batch-size "
                         "reproduces the same 'kept' population for a fair comparison")
    args = ap.parse_args()

    splits = set(args.split.split(","))
    samples = [r for r in read_jsonl(f"{args.data}/prompts.{args.variant}.jsonl")
               if r["split"] in splits]
    samples.sort(key=lambda r: r["pid"])
    if args.limit:
        samples = samples[:args.limit]

    targets = targets_for(args.task, args.data, samples)

    torch.manual_seed(args.seed)
    gen = HFGenerator(
        args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        tokenizer=args.tokenizer,
    )

    records, rollout_time, scoring_time = score(gen, samples, args.task, targets)

    write_jsonl(args.out, records)
    mean = sum(r["auc"] for r in records) / len(records) if records else 0.0
    print(f"{len(records)} scored, mean TRACE score {mean:.1f}, "
          f"rollout {rollout_time:.1f}s + scoring {scoring_time:.1f}s")
    stats = {
        "n": len(records),
        "mean_auc": mean,
        "wall_clock_s": rollout_time + scoring_time,
        "rollout_time_s": rollout_time,
        "scoring_time_s": scoring_time,
        "impl": "trace_hf",
    }
    with open(args.out + ".stats", "w") as handle:
        json.dump(stats, handle)


if __name__ == "__main__":
    main()

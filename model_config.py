#!/usr/bin/env python3
"""Central model-family behavior for math/code training and evaluation.

Every difference between the supported base models lives here, so the rest of the
pipeline stays model-agnostic instead of branching on a checkpoint name.  A profile
carries exactly four model-dependent things:

  ``thinking``          Qwen3's native chat-template switch, always passed explicitly.
  ``legacy_prefill``    Qwen2.5's historical assistant prefill that opened ``<think>``
                        inside the prompt.  Qwen3 opens it itself, so it must not.
  ``response_budget``   supported tasks and their rollout token caps.
  ``sampling``          shared protocol distribution (see PROTOCOL_SAMPLING).

Everything else -- datasets, rewards, cutoff ratios, AUC, output schema -- is protocol
and is deliberately NOT model-dependent.

This module imports only the standard library at import time; ``transformers`` is
loaded inside the helpers that inspect a checkpoint, keeping dataset and reward-only
processes lightweight.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


THINK_START = "<think>"
THINK_END = "</think>"

# Qwen2.5 has no native thinking mode, so the original experiments opened the
# reasoning block for it by prefilling the assistant turn.  Qwen3 emits <think>
# itself; prefilling it there would produce a doubled marker.
LEGACY_ASSISTANT_PREFILL = "Let me solve this step by step.\n<think>"

TASKS = ("math", "code", "arlsat")


class ModelFamily(str, Enum):
    """Supported text-model families (Qwen2 also covers Qwen2.5)."""

    QWEN2 = "qwen2"
    QWEN3 = "qwen3"  # archived AR-LSAT compatibility only
    LLAMA = "llama"
    PHI3 = "phi3"  # Phi-4-mini-instruct uses Phi3ForCausalLM

    def __str__(self) -> str:
        return self.value


class UnsupportedModelError(ValueError):
    """Raised when a checkpoint is not one of the supported model families."""


class Qwen3TokenizerError(ValueError):
    """Raised when a tokenizer cannot represent Qwen3 thinking markers."""


@dataclass(frozen=True)
class SamplingSettings:
    """Nucleus/top-k settings shared by every backend.

    Temperature is intentionally absent: it is a per-task protocol value owned by
    ``trace.TASK_CFG``, not a model property.  These fields only pin the filtering
    distribution so the vLLM, ``model.generate``, and hand-rolled cache samplers all
    draw from the same one.

    ``top_k=0`` disables top-k filtering and ``min_p=0`` disables min-p filtering in
    both Transformers and vLLM, so one set of values is valid for either backend.
    (vLLM still "quietly accepts -1 as disabled" for historical reasons, but 0 is the
    spelling it documents and defaults to; sampling_params.py: SamplingParams.)
    """

    top_p: float
    top_k: int
    min_p: float = 0.0

    def as_kwargs(self) -> dict[str, Any]:
        return {"top_p": self.top_p, "top_k": self.top_k, "min_p": self.min_p}

    # Backend-named aliases kept so call sites read explicitly at the boundary.
    hf_kwargs = as_kwargs
    vllm_kwargs = as_kwargs


# The math/code experiment protocol samples with plain temperature and no nucleus or
# top-k truncation.  Qwen3's model card recommends 0.6 / 0.95 / 20 for thinking mode,
# but adopting that would change the measured distribution and therefore the reported
# E[R-hat] curves and AUC.  The protocol wins; the recommendation is recorded in
# QWEN3_RECOMMENDED_SAMPLING for reference only and is not used by the pipeline.
PROTOCOL_SAMPLING = SamplingSettings(top_p=1.0, top_k=0, min_p=0.0)

QWEN3_RECOMMENDED_SAMPLING = SamplingSettings(top_p=0.95, top_k=20, min_p=0.0)


# Controlled comparison budgets inherited from the Qwen2.5 TRACE baseline.
# Llama/Phi truncation rates still need measurement. Qwen3 is archived AR-LSAT only;
# its historical budget and measurements are documented in OPEN_ISSUES.md.
RESPONSE_BUDGETS: dict[ModelFamily, dict[str, int]] = {
    ModelFamily.QWEN2: {"math": 1024, "code": 600, "arlsat": 1024},
    ModelFamily.QWEN3: {"arlsat": 4096},
    ModelFamily.LLAMA: {"math": 1024, "code": 600},
    ModelFamily.PHI3: {"math": 1024, "code": 600},
}


@dataclass(frozen=True)
class ModelProfile:
    """All behavior that differs between the supported model families."""

    family: ModelFamily
    thinking: bool
    legacy_prefill: bool
    response_budget: Mapping[str, int]
    sampling: SamplingSettings = PROTOCOL_SAMPLING

    @property
    def chat_template_kwargs(self) -> dict[str, bool]:
        # Qwen2 templates do not define this model-level feature.
        return {"enable_thinking": self.thinking} if self.family is ModelFamily.QWEN3 else {}

    def max_response_tokens(self, task: str) -> int:
        """Rollout budget for ``task`` -- one of ``TASKS``, AR-LSAT included."""
        try:
            return self.response_budget[task]
        except KeyError:
            raise ValueError(
                f"no response budget for task {task!r}; expected one of {sorted(self.response_budget)}"
            ) from None


@dataclass(frozen=True)
class ThinkTokenIds:
    start: int
    end: int


@dataclass(frozen=True)
class GenerationTokenIds:
    """Resolved generation IDs, sourced from the loaded objects rather than literals."""

    eos: tuple[int, ...]
    pad: int | None

    @property
    def eos_for_backend(self) -> int | list[int] | None:
        if not self.eos:
            return None
        return self.eos[0] if len(self.eos) == 1 else list(self.eos)

    def hf_kwargs(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        eos = self.eos_for_backend
        if eos is not None:
            result["eos_token_id"] = eos
        if self.pad is not None:
            result["pad_token_id"] = self.pad
        return result


@dataclass(frozen=True)
class ResponseParts:
    """A generated response split at its closing thinking marker.

    ``prefix_before_cot + reasoning + THINK_END + after_think`` reconstructs the
    original prompt and response.  In particular, Qwen3's generated ``<think>`` marker
    lives in ``prefix_before_cot`` rather than contaminating the token-ratio CoT
    cutoffs used by TRACE, so both families measure cutoffs over reasoning text only.
    """

    prefix_before_cot: str
    reasoning: str
    after_think: str
    opening_marker: str

    @property
    def cot(self) -> str:
        return self.reasoning


_PROFILES = {
    ModelFamily.QWEN2: ModelProfile(
        family=ModelFamily.QWEN2,
        thinking=True,
        legacy_prefill=True,
        response_budget=RESPONSE_BUDGETS[ModelFamily.QWEN2],
    ),
    ModelFamily.LLAMA: ModelProfile(
        family=ModelFamily.LLAMA, thinking=True, legacy_prefill=True,
        response_budget=RESPONSE_BUDGETS[ModelFamily.LLAMA],
    ),
    ModelFamily.PHI3: ModelProfile(
        family=ModelFamily.PHI3, thinking=True, legacy_prefill=True,
        response_budget=RESPONSE_BUDGETS[ModelFamily.PHI3],
    ),
    ModelFamily.QWEN3: ModelProfile(
        family=ModelFamily.QWEN3,
        thinking=True,
        legacy_prefill=False,
        response_budget=RESPONSE_BUDGETS[ModelFamily.QWEN3],
    ),
}


_FAMILY_ALIASES = {
    "qwen2": ModelFamily.QWEN2,
    "qwen2.5": ModelFamily.QWEN2,
    "qwen2_5": ModelFamily.QWEN2,
    "qwen3": ModelFamily.QWEN3,
    "llama": ModelFamily.LLAMA,
    "phi3": ModelFamily.PHI3,
}


def family_from_model_type(model_type: str) -> ModelFamily:
    """Map an AutoConfig ``model_type`` to an experiment model family."""

    if not isinstance(model_type, str) or not model_type.strip():
        raise UnsupportedModelError("config.model_type must be a non-empty string")
    normalized = model_type.lower().replace("-", "_").replace(".", "_")
    if normalized == "llama":
        return ModelFamily.LLAMA
    if normalized == "phi3":
        return ModelFamily.PHI3
    if normalized == "qwen3" or normalized.startswith("qwen3_"):
        return ModelFamily.QWEN3
    if normalized == "qwen2" or normalized.startswith("qwen2_"):
        return ModelFamily.QWEN2
    raise UnsupportedModelError(
        f"unsupported model_type {model_type!r}; expected llama, phi3, qwen2/Qwen2.5, or archived qwen3"
    )


def _config_model_type(config: Any) -> str:
    if isinstance(config, Mapping):
        model_type = config.get("model_type")
    else:
        model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise UnsupportedModelError("AutoConfig object has no model_type")
    return model_type


def detect_model_family(
    model_or_config: str | os.PathLike[str] | Mapping[str, Any] | Any,
    *,
    config_kwargs: Mapping[str, Any] | None = None,
) -> ModelFamily:
    """Detect a family from AutoConfig, loading it lazily for a model path/name.

    A checkpoint name is never parsed heuristically: renamed and merged checkpoints
    stay correct because their ``config.model_type`` is authoritative.
    """

    if isinstance(model_or_config, (str, os.PathLike)):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            os.fspath(model_or_config), **dict(config_kwargs or {})
        )
    else:
        config = model_or_config
    return family_from_model_type(_config_model_type(config))


def profile_for_family(
    family: ModelFamily | str,
    *,
    thinking: bool = True,
) -> ModelProfile:
    """Return an immutable profile; thinking defaults explicitly to enabled."""

    if not isinstance(thinking, bool):
        raise TypeError("thinking must be a bool")
    if not isinstance(family, ModelFamily):
        family = family_from_model_type(family)
    return replace(_PROFILES[family], thinking=thinking)


def get_model_profile(
    model_or_config: str | os.PathLike[str] | Mapping[str, Any] | Any,
    *,
    tokenizer: Any | None = None,
    thinking: bool = True,
    config_kwargs: Mapping[str, Any] | None = None,
) -> ModelProfile:
    """Detect and build a profile, optionally validating a loaded tokenizer."""

    profile = profile_for_family(
        detect_model_family(model_or_config, config_kwargs=config_kwargs),
        thinking=thinking,
    )
    if profile.family is ModelFamily.QWEN3 and tokenizer is not None:
        validate_qwen3_think_tokens(tokenizer)
    return profile


# Descriptive alias for callers that treat profile resolution as model loading setup.
resolve_model_profile = get_model_profile


def validate_qwen3_think_tokens(tokenizer: Any) -> ThinkTokenIds:
    """Require Qwen3's opening and closing markers to each be one vocabulary token.

    Qwen3 owns ``<think>``/``</think>`` as real vocabulary entries (they are added
    tokens but *not* special tokens, so ``skip_special_tokens=True`` preserves them --
    the response parsing below depends on that).  A tokenizer that splits them is a
    mismatched checkpoint, which would silently break every cutoff.
    """

    try:
        vocab = tokenizer.get_vocab()
    except (AttributeError, TypeError) as exc:
        raise Qwen3TokenizerError("tokenizer does not expose get_vocab()") from exc

    missing = [token for token in (THINK_START, THINK_END) if token not in vocab]
    if missing:
        raise Qwen3TokenizerError(
            "Qwen3 tokenizer is missing required thinking token(s): " + ", ".join(missing)
        )

    ids: list[int] = []
    for token in (THINK_START, THINK_END):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != vocab[token]:
            raise Qwen3TokenizerError(
                f"Qwen3 thinking marker {token!r} must encode as its single vocabulary token"
            )
        ids.append(int(encoded[0]))
    if ids[0] == ids[1]:
        raise Qwen3TokenizerError("Qwen3 opening and closing thinking tokens share an ID")
    return ThinkTokenIds(start=ids[0], end=ids[1])


def _coerce_profile(
    profile_or_family: ModelProfile | ModelFamily | str | None,
    tokenizer: Any,
    thinking: bool | None,
) -> ModelProfile:
    if isinstance(profile_or_family, ModelProfile):
        return (
            profile_or_family
            if thinking is None
            else replace(profile_or_family, thinking=thinking)
        )
    if isinstance(profile_or_family, ModelFamily):
        return profile_for_family(
            profile_or_family, thinking=True if thinking is None else thinking
        )
    if isinstance(profile_or_family, str) and profile_or_family.lower() in _FAMILY_ALIASES:
        return profile_for_family(
            _FAMILY_ALIASES[profile_or_family.lower()],
            thinking=True if thinking is None else thinking,
        )

    source = profile_or_family or getattr(tokenizer, "name_or_path", None)
    if not source:
        raise ValueError("provide a model profile/family when tokenizer has no name_or_path")
    return get_model_profile(
        source,
        tokenizer=tokenizer,
        thinking=True if thinking is None else thinking,
    )


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    profile_or_family: ModelProfile | ModelFamily | str | None = None,
    *,
    thinking: bool | None = None,
    legacy_assistant_prefill: bool | None = None,
) -> str:
    """Render the checkpoint's own template plus its assistant generation prompt.

    Qwen3 receives ``enable_thinking`` even when it matches the template default, so
    behavior never depends on a tokenizer-version-dependent implicit value.  Note that
    ``enable_thinking=False`` makes Qwen3's template inject a *closed, empty*
    ``<think></think>`` block into the prompt, which would destroy the CoT-cutoff
    protocol in archived Qwen3 runs.

    ``legacy_assistant_prefill`` defaults to the profile's own setting: on for Qwen2.5, Llama and Phi
    (which have no native thinking mode) and off for Qwen3 (which opens ``<think>``
    itself).
    """

    profile = _coerce_profile(profile_or_family, tokenizer, thinking)
    prefill = profile.legacy_prefill if legacy_assistant_prefill is None else legacy_assistant_prefill

    if profile.family is ModelFamily.QWEN3:
        validate_qwen3_think_tokens(tokenizer)
        if prefill:
            raise ValueError(
                "legacy assistant prefill is only valid for non-native-thinking models; Qwen3 "
                "generates its own <think> marker"
            )

    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        **profile.chat_template_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(tokenize=False) did not return text")
    return rendered + LEGACY_ASSISTANT_PREFILL if prefill else rendered


# Short alias useful at dataset call sites.
render_prompt = render_chat_prompt


def render_records(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    profile: ModelProfile,
) -> list[str]:
    """Render dataset records (which carry ``messages``) into model input strings.

    Dataset files store a model-neutral ``prompt`` for artifacts and clustering; the
    actual model input is always rendered here, with the tokenizer that will run.
    """

    return [render_chat_prompt(tokenizer, record["messages"], profile) for record in records]


# The historical math/code split point: the FIRST closing marker.  Qwen3 emits exactly
# one </think> boundary, so this also matches its native layout while keeping Qwen2.5
# byte-identical to the original runs.
def split_reasoning_response(
    response: str,
    *,
    prompt: str = "",
    family: ModelFamily | str | None = None,
) -> ResponseParts | None:
    """Split at ``</think>`` and move any generated opening marker into the prefix.

    ``None`` denotes the malformed-response case previously represented by a missing
    close marker.  Opening-marker handling is deliberately tolerant so a Qwen2.5
    response (marker prefilled in the prompt) and a Qwen3 response (marker generated)
    both yield ``reasoning`` containing reasoning text only.
    """

    if not isinstance(response, str) or not isinstance(prompt, str):
        raise TypeError("response and prompt must be strings")
    if family is not None and not isinstance(family, ModelFamily):
        family_from_model_type(family)

    close = response.find(THINK_END)
    if close < 0:
        return None

    raw_reasoning = response[:close]
    opening = re.match(r"\s*" + re.escape(THINK_START), raw_reasoning)
    if opening is None:
        prefix_before_cot = prompt
        reasoning = raw_reasoning
        opening_text = ""
    else:
        opening_text = opening.group(0)
        prefix_before_cot = prompt + opening_text
        reasoning = raw_reasoning[opening.end():]
    return ResponseParts(
        prefix_before_cot=prefix_before_cot,
        reasoning=reasoning,
        after_think=response[close + len(THINK_END):],
        opening_marker=opening_text,
    )


split_response = split_reasoning_response


def _normalize_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    result: list[int] = []
    for token_id in values:
        if token_id is None or isinstance(token_id, bool):
            continue
        try:
            token_id = int(token_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid token ID {token_id!r}") from exc
        if token_id >= 0 and token_id not in result:
            result.append(token_id)
    return tuple(result)


def resolve_generation_token_ids(
    tokenizer: Any,
    config_or_model: Any | None = None,
) -> GenerationTokenIds:
    """Resolve EOS/padding IDs from actual generation, model, and tokenizer config.

    Runtime ``generation_config`` has precedence for EOS -- Qwen3 stops on both
    ``<|im_end|>`` and ``<|endoftext|>``, which a bare ``tokenizer.eos_token_id`` would
    miss.  Padding prefers the tokenizer because it owns batch construction, then falls
    back to model config and finally the first resolved EOS token.  No family-specific
    numeric ID is assumed anywhere.
    """

    generation_config = getattr(config_or_model, "generation_config", None)
    model_config = getattr(config_or_model, "config", config_or_model)

    eos = _normalize_ids(getattr(generation_config, "eos_token_id", None))
    if not eos:
        eos = _normalize_ids(getattr(model_config, "eos_token_id", None))
    if not eos:
        eos = _normalize_ids(getattr(tokenizer, "eos_token_id", None))

    pad_candidates = (
        getattr(tokenizer, "pad_token_id", None),
        getattr(generation_config, "pad_token_id", None),
        getattr(model_config, "pad_token_id", None),
    )
    pad = next((ids[0] for value in pad_candidates if (ids := _normalize_ids(value))), None)
    if pad is None and eos:
        pad = eos[0]
    return GenerationTokenIds(eos=eos, pad=pad)


generation_token_ids = resolve_generation_token_ids


def sampling_kwargs(
    profile: ModelProfile,
    temperature: float,
    *,
    backend: str = "hf",
    tokenizer: Any | None = None,
    config_or_model: Any | None = None,
) -> dict[str, Any]:
    """Backend-correct decoding kwargs: protocol temperature + profile filtering.

    ``temperature <= 0`` means greedy; the filtering settings are still returned for
    vLLM (which ignores them at temperature 0) but ``do_sample=False`` is set for HF.
    """

    sampling = profile.sampling
    if backend == "vllm":
        # vLLM forces greedy filtering itself when temperature < eps; passing the
        # values anyway keeps one code path.
        return {"temperature": temperature, **sampling.vllm_kwargs()}
    if backend != "hf":
        raise ValueError("backend must be 'hf' or 'vllm'")

    if temperature and temperature > 0:
        result: dict[str, Any] = {"do_sample": True, "temperature": temperature}
        result.update(sampling.hf_kwargs())
    else:
        result = {"do_sample": False}
    if tokenizer is not None:
        result.update(resolve_generation_token_ids(tokenizer, config_or_model).hf_kwargs())
    return result


def verl_hydra_overrides(
    model_or_profile: ModelProfile | ModelFamily | str | os.PathLike[str] | Mapping[str, Any] | Any,
    *,
    task: str | None = None,
    thinking: bool = True,
) -> list[str]:
    """Return the verl overrides that depend on the model, and nothing else.

    Only two things are model-dependent at training time: Qwen3's explicit thinking
    switch (verl forwards ``data.apply_chat_template_kwargs`` into
    ``apply_chat_template``; see verl/utils/dataset/rl_dataset.py) and the response
    budget.  Rollout sampling is intentionally left at verl's defaults for both
    families so the training distribution stays a protocol constant.
    """

    if isinstance(model_or_profile, ModelProfile):
        profile = replace(model_or_profile, thinking=thinking)
    elif isinstance(model_or_profile, ModelFamily):
        profile = profile_for_family(model_or_profile, thinking=thinking)
    elif isinstance(model_or_profile, str) and model_or_profile.lower() in _FAMILY_ALIASES:
        profile = profile_for_family(_FAMILY_ALIASES[model_or_profile.lower()], thinking=thinking)
    else:
        profile = get_model_profile(model_or_profile, thinking=thinking)

    overrides: list[str] = []
    if profile.family is ModelFamily.QWEN3:
        overrides.append(
            f"+data.apply_chat_template_kwargs.enable_thinking={str(profile.thinking).lower()}"
        )
    if task is not None:
        overrides.append(f"data.max_response_length={profile.max_response_tokens(task)}")
    return overrides


def training_chat_template(tokenizer: Any, profile: ModelProfile) -> str:
    """Append the evaluation prefill without modifying/saving the base tokenizer.

    Passed as apply_chat_template(chat_template=...) by both verl's dataset
    length filter and rollout agent. No assistant end-of-turn token is inserted.
    """
    template = tokenizer.get_chat_template()
    if not profile.legacy_prefill:
        return template
    return template + "{% if add_generation_prompt %}{{ " + json.dumps(
        LEGACY_ASSISTANT_PREFILL
    ) + " }}{% endif %}"


def hydra_string(value: str) -> str:
    """Quote for Hydra: escape quotes and adjacent slashes, keep Jinja's \n."""
    escaped = re.sub(r'(\\*)"', lambda m: "\\" * (2 * len(m[1]) + 1) + '"', value)
    trailing = len(escaped) - len(escaped.rstrip("\\"))
    return '"' + escaped + "\\" * trailing + '"'


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit model-family-specific settings for the training driver"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="verl-overrides",
        choices=["verl-overrides", "response-budget", "family"],
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="checkpoint path/name; family comes from AutoConfig")
    source.add_argument("--model-type", help="offline/debug config.model_type value")
    parser.add_argument("--tokenizer", help="optional tokenizer path when different from --model")
    parser.add_argument("--task", choices=TASKS, help="task whose response budget to emit")
    parser.add_argument("--no-thinking", action="store_true", help="explicitly disable Qwen3 thinking")
    parser.add_argument(
        "--skip-tokenizer-validation",
        action="store_true",
        help="skip Qwen3 <think>/</think> vocabulary validation",
    )
    parser.add_argument("--format", choices=("lines", "shell", "json", "nul"), default="lines")
    parser.add_argument("--training-prefill", action="store_true",
                        help="use the evaluation assistant prefill in verl training")
    args = parser.parse_args()

    model_source: Any = {"model_type": args.model_type} if args.model_type else args.model
    profile = get_model_profile(model_source, thinking=not args.no_thinking)
    if (
        profile.family is ModelFamily.QWEN3
        and args.model
        and not args.skip_tokenizer_validation
    ):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        validate_qwen3_think_tokens(tokenizer)

    if args.command == "family":
        print(profile.family.value)
        return
    if args.command == "response-budget":
        if not args.task:
            parser.error("response-budget requires --task")
        print(profile.max_response_tokens(args.task))
        return

    overrides = verl_hydra_overrides(profile, task=args.task, thinking=profile.thinking)
    if args.training_prefill:
        if not (args.tokenizer or args.model):
            parser.error("--training-prefill requires --model or --tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        template = training_chat_template(tokenizer, profile)
        overrides.append("+data.apply_chat_template_kwargs.chat_template=" + hydra_string(template))
    if args.format == "nul":
        sys.stdout.write("\0".join(overrides) + "\0")
    elif args.format == "json":
        print(json.dumps(overrides))
    elif args.format == "shell":
        print(shlex.join(overrides))
    elif overrides:
        print("\n".join(overrides))


if __name__ == "__main__":
    _main()

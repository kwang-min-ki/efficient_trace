#!/usr/bin/env python3
"""학습·평가 공통 모델 계열 판별, 프롬프트·응답 예산·샘플링·종료 토큰 설정"""

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

# 추론 시작 태그를 assistant 접두사로 제공
LEGACY_ASSISTANT_PREFILL = "Let me solve this step by step.\n<think>"

TASKS = ("math", "code")


class ModelFamily(str, Enum):
    """지원하는 모델 계열, Qwen2에 Qwen2.5 포함"""

    QWEN2 = "qwen2"
    LLAMA = "llama"

    def __str__(self) -> str:
        """모델 계열의 문자열 값 반환"""
        return self.value


class UnsupportedModelError(ValueError):
    """지원하지 않는 모델 계열의 설정 오류"""


@dataclass(frozen=True)
class SamplingSettings:
    """생성 경로에서 공유하는 top-p·top-k·min-p 설정

    온도는 모델 속성과 분리해 호출부에서 전달
    """

    top_p: float
    top_k: int
    min_p: float = 0.0

    def as_kwargs(self) -> dict[str, Any]:
        """샘플링 필터 설정을 키워드 인자 사전으로 반환"""
        return {"top_p": self.top_p, "top_k": self.top_k, "min_p": self.min_p}

    # 호출부에서 생성 엔진을 명시할 수 있도록 별칭 제공
    hf_kwargs = as_kwargs
    vllm_kwargs = as_kwargs


# 현재 실험은 온도만 적용하고 top-p·top-k·min-p 필터 비활성화
PROTOCOL_SAMPLING = SamplingSettings(top_p=1.0, top_k=0, min_p=0.0)


# Qwen2.5 기준 응답 예산으로 비교 조건 통일
# Llama의 추론 잘림 비율은 별도 확인 필요
RESPONSE_BUDGETS: dict[ModelFamily, dict[str, int]] = {
    ModelFamily.QWEN2: {"math": 1024, "code": 600},
    ModelFamily.LLAMA: {"math": 1024, "code": 600},
}


@dataclass(frozen=True)
class ModelProfile:
    """모델 계열별 추론 접두사·응답 예산·샘플링 설정"""

    family: ModelFamily
    thinking: bool
    legacy_prefill: bool
    response_budget: Mapping[str, int]
    sampling: SamplingSettings = PROTOCOL_SAMPLING

    @property
    def chat_template_kwargs(self) -> dict[str, bool]:
        """지원 템플릿에 별도 thinking 옵션 없이 빈 사전 반환"""
        return {}

    def max_response_tokens(self, task: str) -> int:
        """도메인의 응답 토큰 한도 반환"""
        try:
            return self.response_budget[task]
        except KeyError:
            raise ValueError(
                f"no response budget for task {task!r}; expected one of {sorted(self.response_budget)}"
            ) from None


@dataclass(frozen=True)
class GenerationTokenIds:
    """로드된 모델·토크나이저에서 결정한 종료·패딩 토큰 ID"""

    eos: tuple[int, ...]
    pad: int | None

    @property
    def eos_for_backend(self) -> int | list[int] | None:
        """종료 토큰 수에 따라 None, 단일 ID 또는 목록 반환"""
        if not self.eos:
            return None
        return self.eos[0] if len(self.eos) == 1 else list(self.eos)

    def hf_kwargs(self) -> dict[str, Any]:
        """HF 생성에 전달할 종료·패딩 토큰 인자 구성"""
        result: dict[str, Any] = {}
        eos = self.eos_for_backend
        if eos is not None:
            result["eos_token_id"] = eos
        if self.pad is not None:
            result["pad_token_id"] = self.pad
        return result


@dataclass(frozen=True)
class ResponseParts:
    """응답을 추론 접두사·추론·답으로 나눈 결과

    prefix_before_cot + reasoning + THINK_END + after_think로 원문 복원
    생성된 think 시작 태그는 추론 길이에 포함하지 않고 접두사에 보관
    """

    prefix_before_cot: str
    reasoning: str
    after_think: str
    opening_marker: str

    @property
    def cot(self) -> str:
        """추론 텍스트 반환"""
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
}


_FAMILY_ALIASES = {
    "qwen2": ModelFamily.QWEN2,
    "qwen2.5": ModelFamily.QWEN2,
    "qwen2_5": ModelFamily.QWEN2,
    "llama": ModelFamily.LLAMA,
}


def family_from_model_type(model_type: str) -> ModelFamily:
    """AutoConfig의 model_type을 지원 모델 계열로 변환"""

    if not isinstance(model_type, str) or not model_type.strip():
        raise UnsupportedModelError("config.model_type must be a non-empty string")
    normalized = model_type.lower().replace("-", "_").replace(".", "_")
    if normalized == "llama":
        return ModelFamily.LLAMA
    if normalized == "qwen2" or normalized.startswith("qwen2_"):
        return ModelFamily.QWEN2
    raise UnsupportedModelError(
        f"unsupported model_type {model_type!r}; expected llama or qwen2/Qwen2.5"
    )


def _config_model_type(config: Any) -> str:
    """사전 또는 설정 객체에서 model_type 추출"""
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
    """체크포인트 이름 대신 AutoConfig의 model_type으로 계열 판별"""

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
    """모델 계열의 불변 설정 반환, 기본 thinking 활성화"""

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
    """체크포인트 설정에서 모델 계열을 판별해 실행 설정 구성"""

    profile = profile_for_family(
        detect_model_family(model_or_config, config_kwargs=config_kwargs),
        thinking=thinking,
    )
    return profile


resolve_model_profile = get_model_profile


def _coerce_profile(
    profile_or_family: ModelProfile | ModelFamily | str | None,
    tokenizer: Any,
    thinking: bool | None,
) -> ModelProfile:
    """설정 객체·계열·경로 입력을 ModelProfile로 통일"""
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
    """모델 고유 대화 템플릿과 필요한 추론 시작 접두사로 입력 구성"""

    profile = _coerce_profile(profile_or_family, tokenizer, thinking)
    prefill = profile.legacy_prefill if legacy_assistant_prefill is None else legacy_assistant_prefill

    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        **profile.chat_template_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(tokenize=False) did not return text")
    return rendered + LEGACY_ASSISTANT_PREFILL if prefill else rendered


render_prompt = render_chat_prompt


def render_records(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    profile: ModelProfile,
) -> list[str]:
    """각 레코드의 messages를 모델 입력 문자열로 변환"""

    return [render_chat_prompt(tokenizer, record["messages"], profile) for record in records]


def split_reasoning_response(
    response: str,
    *,
    prompt: str = "",
    family: ModelFamily | str | None = None,
) -> ResponseParts | None:
    """첫 think 종료 태그를 기준으로 추론과 답 분리

    생성된 시작 태그는 접두사로 이동해 추론 토큰 비율에서 제외
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
    """종료·패딩 ID 입력을 중복 없는 정수 튜플로 변환"""
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
    """모델·토크나이저 설정에서 종료·패딩 ID 결정

    EOS는 generation_config 우선, 패딩은 토크나이저 우선
    패딩 설정이 없으면 첫 EOS 사용
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
    """온도·모델 설정을 HF 또는 vLLM 생성 인자로 변환

    HF에서 온도가 0 이하면 do_sample=False 적용
    """

    sampling = profile.sampling
    if backend == "vllm":
        # vLLM은 온도가 0이면 greedy 처리를 내부 적용
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
    """모델별 응답 토큰 한도를 verl 옵션으로 변환, 학습 샘플링은 기본값 유지"""

    if isinstance(model_or_profile, ModelProfile):
        profile = replace(model_or_profile, thinking=thinking)
    elif isinstance(model_or_profile, ModelFamily):
        profile = profile_for_family(model_or_profile, thinking=thinking)
    elif isinstance(model_or_profile, str) and model_or_profile.lower() in _FAMILY_ALIASES:
        profile = profile_for_family(_FAMILY_ALIASES[model_or_profile.lower()], thinking=thinking)
    else:
        profile = get_model_profile(model_or_profile, thinking=thinking)

    overrides: list[str] = []
    if task is not None:
        overrides.append(f"data.max_response_length={profile.max_response_tokens(task)}")
    return overrides


def training_chat_template(tokenizer: Any, profile: ModelProfile) -> str:
    """평가와 같은 추론 접두사를 학습 템플릿에 추가

    원본 토크나이저는 변경하지 않고 assistant 종료 토큰도 추가하지 않음
    """
    template = tokenizer.get_chat_template()
    if not profile.legacy_prefill:
        return template
    return template + "{% if add_generation_prompt %}{{ " + json.dumps(
        LEGACY_ASSISTANT_PREFILL
    ) + " }}{% endif %}"


def hydra_string(value: str) -> str:
    """줄바꿈을 유지하며 Hydra 문자열의 따옴표·역슬래시 이스케이프"""
    escaped = re.sub(r'(\\*)"', lambda m: "\\" * (2 * len(m[1]) + 1) + '"', value)
    trailing = len(escaped) - len(escaped.rstrip("\\"))
    return '"' + escaped + "\\" * trailing + '"'


def _main() -> None:
    """학습용 모델 옵션·대화 템플릿을 CLI에서 출력"""
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
    parser.add_argument("--format", choices=("lines", "shell", "json", "nul"), default="lines")
    parser.add_argument("--training-prefill", action="store_true",
                        help="use the evaluation assistant prefill in verl training")
    args = parser.parse_args()

    model_source: Any = {"model_type": args.model_type} if args.model_type else args.model
    profile = get_model_profile(model_source)
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

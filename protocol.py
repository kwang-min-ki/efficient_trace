"""TRACE 공통 추론 절단 비율, 답 생성·중단 규칙과 AUC 계산"""

import os

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# 코드 테스트마다 자식 프로세스·파이프·임시 디렉터리를 열기 때문에 CPU가 많은
# 서버에서도 파일 디스크립터 한도를 소진하지 않도록 기본 동시성을 제한
REWARD_WORKERS = max(
    1,
    int(os.environ.get("REWARD_WORKERS", min(16, os.cpu_count() or 4))),
)


FORCE = {
    "math": "</think>\n<answer>",
    "code": "</think>\n```python\n",
}


REOPEN = {"math": "<answer>", "code": "```python\n"}


STOP = {"math": ["</answer>"], "code": ["```"]}


# 절단점별 답 생성 설정, 원본 응답 예산은 ModelProfile에서 관리
TASK_CFG = {
    "math": dict(n_samples=5, temp=0.7, ans_tokens=32),
    "code": dict(n_samples=1, temp=0.0, ans_tokens=600),
}


# 원본 응답 온도 0.7은 논문 지정값이 아닌 저장소 설정
# 방법 간 비교는 재생성 대신 --records로 같은 응답 사용
ROLLOUT_TEMPERATURE = 0.7


def auc(values):
    """절단 비율별 사다리꼴 적분을 구간 길이로 나누고 100배한 점수 반환"""
    area = sum((FRACS[i + 1] - FRACS[i]) * (values[i] + values[i + 1]) / 2
               for i in range(len(FRACS) - 1))
    return 100.0 * area / (FRACS[-1] - FRACS[0])


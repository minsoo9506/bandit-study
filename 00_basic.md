# 용어

MAB에서는 모든 행동이 순서대로 발생한다 가정합니다. 즉, 시점을 고려합니다.

- action/arm $A_t$: 액션 (recommendation candidates)
- reward $R_t$: 한 번의 행동에 따른 수치화된 결과 (customer interaction from a single trial, such as a click or purchase)
- value $q_{*}(a) = E[R_t | A_t=a]$: 행동에 대한 기대 보상 (estimated long-term reward of an arm over multiple trials)
  - 이는 우리가 알 수 없는 값이라서 데이터를 통해 추정합니다. $Q_t(a)$
- policy: algorithm/agent that chooses actions based on learned values



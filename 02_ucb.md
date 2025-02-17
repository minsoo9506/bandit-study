추정한 보상(value)의 신뢰구간의 상항이 최대가 되는 액션(action)을 선택하는 방법입니다.

$$UCB_t(a) = Q_t(a) + c \sqrt{\frac{\log t}{N_t(a)}}$$

- $N_t(a)$: $t$시점 까지 $a$ 액션을 선택한 횟수

따라서 $t$ 시점에서 우리가 취할 액션은 다음과 같습니다.

$$a_t = \argmax_a [ Q_t(a) + c \sqrt{\frac{\log t}{N_t(a)}} ]$$

초반에는 모든 액션들의 $N_t(a)$ 가 작아서 UCB 가 크고 변동성도 크기 때문에 exploration 이 더 많이 발생합니다. 하지만 시간이 지나면서 모든 액션에 대한 $N_t(a)$ 는 선형적으로 증가하고, $t$ 는 log 함수라서 뒤로 갈수록 증가하지 않게 됩니다. 후반에는 점점 0으로 수렴하게 되어 지금까지 구해진 표본평균 $Q_t(a)$ 으로 exploitation 을 진행하게 됩니다.
- 현재 상황에서 가장 적절한 액션을 선택하는 방법은 다양합니다.
- 아래 수식처럼 추정 value 를 최대화하는 액션을 선택하는 것을 greedy 한 방법이라고 합니다.

$$A_t = \argmax_a{Q_t(a)}$$

- 하지만 우리는 value 값을 모르거나 추정한 값에는 uncertainty 가 존재합니다. 따라서 적절한 exploitation 과 exploration 을 통해 uncertainty 를 줄여나가야합니다.
- 그렇다면 언제 exploitation 을 하고 언제 exploration 을 해야할까요?
- 가장 간단한 방법 중 하나인 $\epsilon$-greedy 방법에 대해 알아보겠습니다.
- 간단합니다. 먼저 $\epsilon$ 의 확률 만큼은 exploration 하고 1-$\epsilon$ 만큼은 exploitation 을 합니다.
  - 현재까지 추정한 데이터를 토대로 가장 value 를 높게 주는 arm 을 선택합니다. (exploitation)
  - 랜덤하게 arm 들 중 하나를 선택합니다. (exploration)
- 그리고 어느정도 time step 이 지나 수렵하는 단계가 되면 exploration 을 줄이기도 합니다.
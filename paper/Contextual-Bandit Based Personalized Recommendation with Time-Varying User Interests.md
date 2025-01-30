- 시간이 지남에 따라 변하지 않는 유저의 취향과 변하는 취향을 구분해서 접근

### 수식
- $u_t$: user at time $t$
- $x_{u_t}$: $d$-dim user's feature vector
- $y_{a_t}$: $m$-dim feature vector of arm $a_t$
- $W(t) \in R^{d*m}$: time-varying unknown weight matrix representing the preferences of users towards items in the feature space
- conditional expectation of the reward $r_{u_t, a_t}(t)$:

$$E[r_{u_t, a_t}(t) | x_{u_t}, y_{a_t}, W(t)] = x_{u_t}^T W(t) y_{a_t}$$

- 이때 stationary 인 상황에서는 $W(t)$ 가 시간에 상관없이 $W$ 로 fix 되어 있다. 이런 경우가 일반적인 contextual bandit model 이다.

## Disjoint Payoff Model
- $\theta_a (t) = W(t) y_a$ 를 $t$ 시간에 arm $a$ 에 대한 unknown preference vector 라고 한다.
- 따라서 특정 arm 을 추천했을 때 expected reward 는 $x_u$ 와 $\theta_a (t)$ 의 내적

$$E[r_{u,a}(t) | x_u, \theta_a(t)] = x_u^T \theta_a (t)$$

- $\theta_a (t)$ 에 piecewise-stationary assumption 을 적용한다.
- time horizon 이 $M_a$ 개의 stationary segments 로 나누어져 있다.
- change points 는 0 부터 $T$ 까지 $T+1$ 개 가 존재한다.
- 각 segment 에서 $\theta_a(t)$ 는 fix 되어 있다.
- arm 마다 change point 는 다르기 때문에 arm 에 대한 유저의 선호는 비동기적으로 반영된다.

## Hybrid Payoff Model
- 일단 skip

## Piecewise-Stationary LinUCB Algorithm under the Disjoint Payoff Model
- 
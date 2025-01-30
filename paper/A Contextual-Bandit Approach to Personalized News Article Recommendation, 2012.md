## Intro
- 유저마다 개인화된 web-based news content 를 제공하는 것이 목적이다.
- 일반적인 추천 모델은 cold start 등 몇가지 문제점을 가진다.
- 논문에서 LinUCB 라는 contextual bandit 알고리즘을 소개한다.

## formulation, related work

### multi-armed bandit formulation
- contextual bandit 알고리즘에 대해 알아보자.
- 용어 정리
  - $u_t$: $t$ 시점 user
  - vector $x_{t,a}$: user $u_t$, arm $a$ 에 대한 정보, context
  - $r_{t, a_t}$: payoff
  - $E[\sum_{t=1}^T r_{t, a_t^*}]$: optimal expected T-trial payoff, $a_t^*$ 는 $t$ 시점에서 expected payoff 를 최대화하는 arm
- 우리의 목표는 아래 수식이 의미하는 arm selection을 잘해서 regret 을 최소화하는 것이다.

$$R_A(T) = E[\sum_{t=1}^T r_{t, a_t^*}] - E[\sum_{t=1}^T r_{t, a_t}]$$

- article recommendation 에서 각 article 이 arm 이고 article 클릭이 발생하면 payoff 는 1 아니면 0 이다. 따라서 expected payoff 는 CTR 이라고 할 수 있다.

## algorithm

### LinUCB with disjoint linear models
-  각 arm $a$ 마다 $d$ 차원의 feature $x_{t,a}$ 를 통해 expected payoff 에 대해 ridge regression 을 만든다.
- disjoint 라는 것은 이 때 fit 하는 parameters 들이 arm 끼리 공유되는 것이 아니라는 의미이다. 따라서 모델 fitting 시 사용하는 data 도 arm 마다 구분될 것이다.
- linear model 이기 때문에 confidence interval 을 어렵지 않게 구할 수 있고 이를 이용해 UCB type arm-selection 방법으로 arm 을 선택한다.
- 구체적인 수식은 논문참조

### LinUCB with hybrid linear models
- arm 끼리 parameter 를 share 하는 것을 의미한다. linear 모델을 만들때 기존 모델에 추가로 user/item combination feature $z_{t,a}$를 사용하고 이 feature 의 paramter 는 arm 마다 동일한 값을 가진다.

$$E[r_{t,a} | x_{t,a}] = z_{t,a}^T \beta + x_{t,a}^T \theta_a$$

## evalutation
- bandit 알고리즘은 일반 ML 과 다르게 평가하기 어렵다. online (live) data 에 대해 적용해야 진짜 평가가 가능하기 때문이다. 하지만 우리는 offline data 만 가능하면 경우가 많고 이는 다른 policy 에 의해 나온 결과이다. 이런 경우는 강화학습에서 off-policy evaluation problem 이라고 부른다.

## experiments
- 뉴스 추천

### data collection
- 특정 일자, 유저와 뉴스 각각 random 하게 뽑기
- position bias 를 최소화하기 위해 특정한 위치 1개의 뉴스에 대해 평가 진행
- 특정 일자 이후 n일 동안 데이터를 평가 데이터로 사용

### feature construction
- 유저 feature 는 1000개
  - 성
  - 연령 10개 세그먼트
  - geographic features
  - behavioral categories
- 뉴스 feature 는 100개
  - url category
  - editor category
- 전부 categorical binary feature
- feature selection, dimension reduction 까지 해서 feature 수를 더 줄였다. 그래서 유저 군집화를 진행했다.
- 유저 $u$가 뉴스 $a$ 클릭할 확률 logistic regression fit (bilinear model) $\phi_u^T W \phi_a$
- 유저 feature 를 project $\phi_u^T W$ 하면 이는 유저가 $i^{th}$ 뉴스 category 를 선호한다는 것을 의미
- 여기에 5 cluster k-means 진행하고 각 group 에 속하는 것으로 feature 만들기
  - gaussian kernel 로 값을 만들고 sum 1이 되게 normalized -> 그룹에 속한 정도를 표현
- 그래서 최종 유저 feature 는 6차원 (6번쨰는 constant feature 1)
- 뉴스도 동일하게 진행해서 disjoint model 을 만든다.
- hybrid model 은 user x item 으로 36차원의 feature 를 모델에 추가하고 shared parameter 를 사용한다.

## compared algorithm
- 웹페이지에서 다양한 component 들이 있는데 이들은 독립적이지 않다. 이를 고려해서 최적의 layout 을 선택하고 싶다.

### problem setting
- $D$ 개의 slot 이 있고 각각 $N$ 개의 variation 이 있다고 가정한다.
- 그럼 layout $A$ 는 $D$ 차원의 vector 로 표현할 수 있다.
- 그리고 context $X$ 가 있고 $X,A$ 를 표현하는 vector 를 $B_{X,A}$ 라고 한다.
- 그리고 이들을 이용하여 reward 는 generalized linear model 을 통해서 모델링한다.

$$E[R|X,A] = g(B_{X,A}^T \mu)$$

- $R$ 는 -1,1 binary, probit regression 사용
- 각 weight 는 mutually independent random variables following a Gaussian posterior distributions, with Bayesian updates over an initial Gaussian prior of $N (0, 1)$
- feature 는 0,1 binary 이고 (당연히 numeric 확장이 어렵지는 않음) 2차원의 interaction feature 까지 사용

### multivariate test
- Thompson Sampling
  1. $t-1$ 시점까지의 데이터들을 이용해서 posterior 분포에서 weight 추정값 $W$ 을 뽑는다.
  2. 이를 통해 가장 reward 가 큰 $A_t$ 를 정한다.
  3. layout $A_t$ 를 보여주고 $R_t$ 를 본다.
- 위 과정에서 2번 연산에서 argmax 시 $O(N^D)$ 이기 때문에 efficient approximation 을 해야한다.
- Hill climbing optimization
  - 랜덤하게 하나의 slot 을 고르고 나머지를 fix 한 채로 $N$개의 경우의 수마다 max reward 인 값을 정한다.
  - 이를 $K$ 번 반복한다.
  - 이를 $S$ 번 반복한다.
  - 따라서 $SKN$ 만큼의 연산이 필요
  - local optimum 을 사용하는 것
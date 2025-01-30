- This paper aims at addressing the position bias and the pseudo-exposure problem to improve the performance in the online recommendation

## LINUCB WITH USER BROWSING MODEL
- 아이템 클릭을 하면 그 뒤로 나오는 아이템을 examination 할 확률이 높아진다.
  - 클릭햇다는 것은 추천 결과가 유저 취향에 맞았다는 것이므로
- 따라서 we introduce position weights relating to both positions and clicks to address the position bias and the pseudo-exposure issue
- assume that the click through rate $\gamma_{t,a} = w_{k,k'}\gamma (a)$
  - $\gamma (a)$: the attractiveness of an item $a$
  - $w_{k,k'}$: the examination probability meaning the probability that a user views an item
  - We assume that $w_{k,k'}$ depends not only on the rank of an item ($k$), but also on the rank of the previously clicked item ($k'$)
- the attractiveness of items 에 대해 linear reward model 을 사용한다.

$$E[\gamma_{a_k}] = \theta^T (w_{k,k'} \cdot x_{a_k})$$

- $w_{k,k'}$ 값은 fixed constant 이고 따로 구할 수 있다.
- 해당 모델은 ridge regression 으로 추정한다.

$$\sum_{t=1}^T \sum_{k=1}^K [\gamma_{a_{k,t}} - \theta^T (w_{k,k'} \cdot x_{a_{k,t}})]^2 + \sum_{j=1}^d \theta_j ^2$$

- UCB 방법을 통해 best recommended set 을 뽑는다.

## E-commerce Recommendation
- 12개 제품을 추천하는 상황
  - 12개 이외 더 추천은 없음
  - 유저가 들어오면 1번 제품, 2번제품의 일부가 보이는 상황, 슬라이드해서 나머지 제품을 구경할 수 있음
- feature
  - our dataset contains 180k records
  - while each record contains a user ID, IDs of 12 commodities
  - context vectors of 12 commodities and a click vector corresponding to the 12 commodities.
  - The context vector is a 56×1 array including some features of both a commodity and a user, such as the commodity’s price and the user’s purchasing power.
- feature dimension reduction
  - denoising AE 로 56 -> 10 차원 축소
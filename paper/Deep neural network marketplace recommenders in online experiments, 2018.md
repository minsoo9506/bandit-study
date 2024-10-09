## Background
- Norwegian marketplace FINN.no 이라는 서비스에 적용
- 유사 item 추천, 개인화 item 추천 2가지 서비스

## Experimentation platform
- offline, online 실험을 진행한다.
- offline test 의 경우 여러 후보 모델들 중에서 확연하게 성능이 떨어지는 모델을 제외하는 역할로 사용한다. 이로 인해 online test 리소스를 효율적으로 사용하고 유저들이 좋지 않은 경험을 하는 경우를 줄일 수 있다.
- offline metric 으로는 hit rate 를 online conversion metric 의 proxy 로 사용한다.

## Hybrid item representation model
- 목적: 유사한 item 추천
- 다양한 item representation 을 독립적으로 만들고 ensemble 합니다.
  - user behavior: 20 일전 유저 행동 데이터를 이용하여 ALS 모델
  - text: item 의 title, desc 로 item 카테고리 분류 모델을 만들고 top layer 를 이용
  - image: image 로 word2vec 으로 만든 item title 의 embedding 을 예측하는 모델 (Inception-v3 에 layer 추가)
  - location: 위치 데이터는 위도경도와 더불어 인구수, 유저 행동 데이터 등 다양한 데이터를 이용
- 위처러 만들어진 item embedding 은 concat 하고 attention layer 를 지나고 cosine similarity 로 co-converted item pairs 를 예측한다.
  - co-converted? two items that get conversions from the same user on the same day, and negative sampling item pairs that are unlikely to co-convert

![img](../Images/Deep%20neural%20network%20marketplace%20recommenders%20in%20online%20experiments,%202018/01.png)

## Sequence-based model
- 유저의 과거 클릭 history 를 통해 다음 k step 의 클릭을 예측하는 모델이다.
- 위 hybid 에서 구한 item representation 을 사용한다.
- LSTM 모델 계열 사용한다.

## Multi-armed bandit model
- 개인화 추천 item feed 에 서빙된다.
- bandit 은 추천모델이라기 보다는 re-ranker 라고 할 수 있다.
- 다양한 종류의 6-10 개의 추천 submodel 결과를 이용한다. epsilon-greedy policy 방법을 사용했고 5% random items 을 추천 list 에 추가했다 (to avoid local minimal).

### Row-separated feed
- simple baseline 이라고 할 수 있다.
- 각 submodel 의 결과를 row 마다 1개씩 사용한다.

### Regression bandit
- submodel score, submodel type, contextual information 으로 click probability 를 regression 모델로 예측한다.
  - Submodel scores are binned into 10 buckets
  - 나머지는 모두 categorical and one-hot encoded
- target 은 위 feature 들로 만들어진 group 별 average number of clicks 을 사용한다.
- ridge model 을 사용했다.

### Deep classification bandit
- click 여부에 대한 classification 모델이다.
- input: mix of scalars (submodel score, item position, hour of day, etc.) and categorical variables (submodel type, location, device, weekday, etc.)
  - scalars are normalized through batch normalization
  - categorical variables are one-hot encoded
- dataset 에서 click 과 view(non click) 의 비율이 많이 다르기 때문에 loss function 에 class weight 를 사용했다. model weight 에 L2 norm 을 적용했다.

## Result
- online 평가 (CTR)
  - item-item: content-based < matrix factorization < hybrid
  - user-item: matrix factorization < sequence-based
  - multi-armed: row-separated feed < regression bandit < deep bandit
- 재밌는 점
  - matrix factorization 의 점수와 CTR 이 monotonic 하게 비례하지는 않았다. 0.8 이 넘어가면 CTR 이 급격하게 떨어졌는데 이는 개인의 취향을 반영하지 못하고 viral tendencies 때문에 그럴 것으로 예상된다.
- item representation 에서 collaborative filtering, content feature 를 함께 사용하는 것이 효과적이다.
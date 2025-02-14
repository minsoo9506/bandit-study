# bandit-study
추천시스템을 중점으로 두고 bandit 알고리즘 공부한 내용을 정리하는 레포

## 설명
- epsilong greedy [`review`](./01_epsilon_greedy.md) [`code`](./notebook/epsilon_greedy.ipynb)
- ucb [`reveiw`](./02_ucb.md) [`code`](./notebook/ucb.ipynb)
- thomson sampling [`review`](./03_thomson_sampling.md) [`code`](./notebook/thomson_sampling.ipynb)

## Reference

### paper
- [Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits, 2018 spotify](https://static1.squarespace.com/static/5ae0d0b48ab7227d232c2bea/t/5ba849e3c83025fa56814f45/1537755637453/BartRecSys.pdf)
  - `epsilon-greedy`, `explanation`
- [Deep neural network marketplace recommenders in online experiments, 2018](https://arxiv.org/abs/1809.02130)
  - `epsilon-greedy`, `hybrid item-representation`
- [A Batched Multi-Armed Bandit Approach to News Headline Testing, 2019](https://arxiv.org/pdf/1908.06256)
  - `thomson-sampling`, `batched MAB`
- [A Contextual-Bandit Approach to Personalized News Article Recommendation, 2012](https://arxiv.org/abs/1003.0146)
  - `LinUCB`
- [Contextual User Browsing Bandits for Large-Scale Online Mobile Recommendation, 2020 Alibaba](https://arxiv.org/pdf/2008.09368)
  - `UBM-LinUCB`, `contextual combinatorial bandit`
- [An Empirical Evaluation of Thompson Sampling, 2011 Yahoo](https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)
- [An Efficient Bandit Algorithm for Realtime Multivariate Optimization, 2018 Amazon](https://arxiv.org/abs/1810.09558)
  - `thomson-sampling`
- [Carousel Personalization in Music Streaming Apps with Contextual Bandits, 2020 Deezer](https://arxiv.org/pdf/2009.06546)
  - `casecade bandit`, `semi-personalized`

### blog
- [카카오 AI추천 : 토픽 모델링과 MAB를 이용한 카카오 개인화 추천, 2021](https://tech.kakao.com/posts/445)
- [Personalized Cuisine Filter, 2020 doordash](https://careersatdoordash.com/blog/personalized-cuisine-filter/)

### etc
- [RecSys 2020 Tutorial: Introduction to Bandits in Recommender Systems, 2020](https://www.youtube.com/watch?v=rDjCfQJ_sYY&t=6s)
- [Korea Summer Workshop on Causal Inference 2022, [Industry] 멀티암드밴딧과 톰슨샘플링](https://www.youtube.com/watch?v=ffdazvIDfTM)
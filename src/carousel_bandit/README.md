## reference
- https://github.com/deezer/carousel_bandits

## dataset
- `user_feature.csv`
  - 975,960 명 유저의 96-dim embedding vector
- `playlist_features.csv`
  - 862 개 플레이리스트의 97-dim weight vector
- user-embedding vector 에 bias 추가하고 user-playlist vector 랑 내적하면 ground truth (display-to-stream 확률)
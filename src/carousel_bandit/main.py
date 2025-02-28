import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.carousel_bandit.env import ContextualEnvironment
from src.carousel_bandit.policy import (EpsilonGreedySegmentPolicy,
                                        RandomPolicy, TSSegmentPolicy)

user_features_df = pd.read_csv("./data/carousel_bandit/user_features.csv")
playlist_feature_df = pd.read_csv("./data/carousel_bandit/playlist_features.csv")

print(f"user_features_df shape: {user_features_df.shape}")
print(f"playlist_feature_df shape: {playlist_feature_df.shape}")

n_users = user_features_df.shape[0]
n_playlists = playlist_feature_df.shape[0]
n_recos = 12
l_init = 3

user_segment = np.array(user_features_df["segment"])
user_features = np.concatenate(
    [user_features_df.drop(["segment"], axis=1), np.ones((n_users, 1))], axis=1
)
playlist_features = np.array(playlist_feature_df)

# 각 round 마다 random 하게 유저를 선택
n_rounds = 20
n_users_per_round = 20000

policies = [
    RandomPolicy(n_playlists),
    EpsilonGreedySegmentPolicy(user_segment, n_playlists, 0.1),
    TSSegmentPolicy(user_segment, n_playlists),
]
n_policies = len(policies)
policies_name = ["Random", "EpsilonGreedySegment", "TSSegmentPolicy"]

cont_env = ContextualEnvironment(
    user_features, playlist_features, user_segment, n_recos
)

overall_rewards = np.zeros((n_policies, n_rounds))
overall_optimal_reward = np.zeros(n_rounds)

for i in tqdm(range(n_rounds)):
    user_ids = np.random.choice(range(n_users), n_users_per_round)
    overall_optimal_reward[i] = np.take(cont_env.th_rewards, user_ids).sum()
    # 각 policy 별로 reward 계산
    for j, policy in tqdm(enumerate(policies)):
        recos = policy.recommend_to_users_batch(user_ids, n_recos, l_init)
        rewards = cont_env.simulate_batch_users_reward(user_ids, recos.astype(np.int64))
        policy.update_policy(user_ids, recos, rewards, l_init)
        overall_rewards[j, i] = rewards.sum()
    if i % 5 == 0:
        print(f"[ Round {i} cumulative regret ]")
        for i in range(n_policies):
            print(
                f"{policies_name[i]}: {round(np.sum(overall_optimal_reward - overall_rewards[i]))}"
            )
        print("")

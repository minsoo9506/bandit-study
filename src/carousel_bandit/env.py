import numpy as np


class ContextualEnvironment:
    def __init__(self, user_features, playlist_features, user_segment, n_recos):
        self.user_features = user_features
        self.playlist_features = playlist_features
        self.user_segment = user_segment
        self.n_recos = n_recos
        self.th_rewards = np.zeros(user_features.shape[0])
        self.th_segment_rewards = np.zeros(user_features.shape[0])
        self.compute_optimal_theoretical_rewards()
        self.compute_segment_optimal_theoretical_rewards()

    # Computes highest expected reward for each user
    def compute_optimal_theoretical_rewards(self):
        n_users = self.user_features.shape[0]
        step = 100000
        for n_step in range(0, n_users, step):
            users_ids = range(n_step, min(n_step + step, n_users))
            opt_recos = self.compute_optimal_recos(users_ids, self.n_recos)
            opt_rewards = self.compute_theoretical_rewards(users_ids, opt_recos)
            self.th_rewards[n_step : n_step + step] = opt_rewards

    # Computes list of n recommendations with highest expected reward for each user
    def compute_optimal_recos(self, users_ids, n_recos):
        users_ids_features = np.take(self.user_features, users_ids, axis=0)
        probs = np.dot(users_ids_features, self.playlist_features.T)
        optimal_recos = np.argsort(-probs)[:, :n_recos]
        return optimal_recos

    # Computes highest expected reward for each user
    def compute_theoretical_rewards(self, users_ids, recos):
        users_ids_features = np.take(self.user_features, users_ids, axis=0)
        recos_features = np.take(self.playlist_features, recos, axis=0)
        th_reward = np.zeros(len(users_ids))
        for i in range(len(users_ids)):
            probs = self.sigmoid(
                np.dot(users_ids_features[i], recos_features[i].T)
            )  # |probs| = (n_recos,)
            th_reward[i] = 1 - np.prod(1 - probs)
        return th_reward

    def compute_segment_optimal_recos(self, n_recos):
        n_seg = len(np.unique(self.user_segment))
        seg_recos = np.zeros((n_seg, n_recos), dtype=np.int64)
        for i in range(n_seg):
            users_ids = np.where(self.user_segment == i)[0]
            users_id_features = np.take(self.user_features, users_ids, axis=0)
            probs = self.sigmoid(np.dot(users_id_features, self.playlist_features.T))
            probs_mean = np.mean(
                probs, axis=0
            )  # |probs_mean| = (self.playlist_features.shape[0],)
            seg_recos[i] = np.argsort(-probs_mean)[:n_recos]
        return seg_recos

    def compute_segment_optimal_theoretical_rewards(self):
        n_users = self.user_features.shape[0]
        seg_opt_recos = self.compute_segment_optimal_recos(self.n_recos)
        step = 100000
        for n_step in range(0, n_users, step):
            users_ids = range(n_step, min(n_step + step, n_users))
            user_seg = np.take(self.user_segment, users_ids)
            opt_recos = np.take(seg_opt_recos, user_seg)
            opt_rewards = self.compute_theoretical_rewards(users_ids, opt_recos)
            self.th_segment_rewards[n_step : n_step + step] = opt_rewards

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def simulate_batch_users_reward(self, batch_user_ids, batch_recos):
        batch_user_features = np.take(self.user_features, batch_user_ids, axis=0)
        batch_playlist_features = np.take(self.playlist_features, batch_recos, axis=0)
        n_users = len(batch_user_ids)
        n_playlists = len(batch_recos[0])
        probs = np.zeros((n_users, n_playlists), dtype=np.float64)
        for i in range(n_users):
            probs[i] = self.sigmoid(
                np.dot(batch_user_features[i], (batch_playlist_features[i].T))
            )
        rewards = np.zeros((n_users, n_playlists), dtype=np.int64)
        rewards_uncascaded = np.random.binomial(1, probs)
        positive_rewards = set()

        # Then, for each user, positive rewards after the first one are set to 0 (and playlists as "unseen" subsequently) to imitate a cascading browsing behavior
        # (nonetheless, users can be drawn several times in the batch of a same round ; therefore, each user
        # can have several positive rewards - i.e. stream several playlists - in a same round, consistently with
        # the multiple-plays framework from the paper)
        # 유저가 처음 positive 가 발생한 것에 대해서만 reward 를 주고, 그 이후에는 0 으로 설정 하는 과정
        nz = (
            rewards_uncascaded.nonzero()
        )  # nonzero 의 index, 2d 일떄 [0] 이 x, [1] 이 y 좌표
        for i in range(len(nz[0])):
            if nz[0][i] not in positive_rewards:
                rewards[nz[0][i]][nz[1][i]] = 1
                positive_rewards.add(nz[0][i])
        return rewards

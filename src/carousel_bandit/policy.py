import numpy as np


class Policy:

    # Returns a list of size n_recos of playlist ids
    def recommend_to_users_batch(self, batch_users, n_recos=12):
        return

    # Updates policies parameters
    def update_policy(self, user_ids, recos, rewards, l_init=3):
        return


# A simple baseline that randomly recommends n_recos playlists to each user.
class RandomPolicy(Policy):
    def __init__(self, n_playlists, cascade_model=True):
        self.cascade_model = cascade_model
        self.n_playlists = n_playlists

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        n_users = len(batch_users)
        recos = np.zeros((n_users, n_recos))
        r = np.arange(self.n_playlists)
        for i in range(n_users):
            np.random.shuffle(r)
            recos[i] = r[:n_recos]
        return recos


# Segment-based Epsilon-Greedy strategy: recommends playlists randomly with probability epsilon,
# otherwise recommends the top n_recos with highest mean observed rewards.
class EpsilonGreedySegmentPolicy(Policy):
    def __init__(self, user_segment, n_playlists, epsilon, cascade_model=True):
        self.user_segment = user_segment
        n_segments = len(np.unique(self.user_segment))
        self.playlist_score = np.ones((n_segments, n_playlists))
        self.epsilon = epsilon
        self.cascade_model = cascade_model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        n_users = len(batch_users)
        user_segment = np.take(self.user_segment, batch_users)
        user_scores = np.take(self.playlist_score, user_segment, axis=0)
        user_random_score = np.random.random(
            user_scores.shape
        )  # |user_random_score| = (n_users, n_playlists)
        user_greedy = np.random.binomial(1, [1 - self.epsilon] * n_users)
        new_scores = user_scores * user_greedy[:, np.newaxis]
        user_choice = np.lexsort((user_random_score, -new_scores))[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice


# Segment-based Thompson Sampling strategy, with Beta(alpha_zero,beta_zero) priors
class TSSegmentPolicy(Policy):
    def __init__(
        self, user_segment, n_playlists, alpha_zero=1, beta_zero=99, cascade_model=True
    ):
        self.user_segment = user_segment
        n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((n_segments, n_playlists))
        self.playlist_success = np.zeros((n_segments, n_playlists))
        self.alpha_zero = alpha_zero
        self.beta_zero = beta_zero
        self.t = 0
        self.cascade_model = cascade_model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_segment = np.take(self.user_segment, batch_users)
        user_displays = np.take(self.playlist_display, user_segment, axis=0).astype(
            float
        )
        user_success = np.take(self.playlist_success, user_segment, axis=0)
        user_score = np.random.beta(
            self.alpha_zero + user_success,
            self.beta_zero + user_displays - user_success,
        )
        user_choice = np.argsort(-user_score)[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice

    def update_policy(self, user_ids, recos, rewards, l_init=3):
        batch_size = len(user_ids)
        for i in range(batch_size):
            user_segment = self.user_segment[user_ids[i]]
            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for p, r in zip(recos[i], rewards[i]):
                nb_display += 1
                self.playlist_success[user_segment][p] += r
                self.playlist_display[user_segment][p] += 1
                if self.cascade_model and (
                    (total_stream == 0 and nb_display == l_init) or (r == 1)
                ):
                    break
        return

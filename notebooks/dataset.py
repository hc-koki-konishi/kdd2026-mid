from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from utils import (
    sample_action_fast,
    softmax,
    eps_greedy_policy,
    sigmoid,
    normal,
    sorted_normal
)

def calc_q_x_a(
    x: np.ndarray,
    a_feat: np.ndarray,
    x_coef: np.ndarray,
    a_coef: np.ndarray,
    x_a_coef: np.ndarray,
    mu: float = None,
    sigma: float = None,
    sparsity_factor: float = 0.0,
    reward_type: str = "continuous",
) -> np.ndarray:
    q_x_a = (x @ x_coef)[:, np.newaxis, :] + (a_feat @ a_coef)[np.newaxis, :, :]
    q_x_a[:, :, 0] += x @ x_a_coef[:, :, 0] @ a_feat.T
    q_x_a = np.squeeze(q_x_a, -1)
    
    if mu is None:
        mu, sigma = q_x_a.mean(), q_x_a.std()
        q_x_a = (q_x_a - mu) / sigma
        
        if reward_type == "binary":
            q_x_a -= q_x_a.min()
            q_x_a = sigmoid(q_x_a * sparsity_factor)
        
        return q_x_a * 0.1, mu, sigma
    else:
        q_x_a = (q_x_a - mu) / sigma
        
        if reward_type == "binary":
            q_x_a -= q_x_a.min()
            q_x_a = sigmoid(q_x_a * sparsity_factor)
        
        return q_x_a * 0.1


@dataclass
class SyntheticDataset:
    n_actions: int
    x_dim: int
    a_dim: int
    reward_type: str = "continuous"
    sparsity_factor: float = 0.1
    reward_std: float = 2.0
    n_all_users: int = 1000
    coef_std: float = 1.0
    random_state: int = 12345
    
    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.a_feat = self.random_.normal(size=(self.n_actions, self.a_dim))
        self._generate_user_features()
        self._generate_coefficients()
        self.q_x_a_all, self.mu, self.sigma = self._calc_q_x_a_for_all_users()
    
    def _generate_user_features(self) -> None:
        self.x_all = self.random_.normal(size=(self.n_all_users, self.x_dim))
    
    def _generate_coefficients(self) -> None:
        # coefficients for q_x_a
        self.x_coef = self.random_.normal(size=(self.x_dim, 1), scale=self.coef_std)
        self.a_coef = self.random_.normal(size=(self.a_dim, 1), scale=self.coef_std)
        self.x_a_coef = self.random_.normal(size=(self.x_dim, self.a_dim, 1), scale=self.coef_std)
    
    def _calc_q_x_a_for_all_users(self) -> np.ndarray:
        q_x_a, mu, sigma = calc_q_x_a(
            x=self.x_all,
            a_feat=self.a_feat,
            x_coef=self.x_coef,
            a_coef=self.a_coef,
            x_a_coef=self.x_a_coef,
            sparsity_factor=self.sparsity_factor,
            reward_type=self.reward_type,
        )
        
        return q_x_a, mu, sigma
    
    def generate_dataset(
        self,
        n_data: int,
        logging_policy_params: dict,
        evaluation_policy_params: dict,
    ) -> np.ndarray:
        
        """Generate a synthetic dataset."""
        data_idx = self.random_.choice(
            self.n_all_users,
            size=n_data,
            replace=True
        )
        x = self.x_all[data_idx]
        q_x_a = calc_q_x_a(
            x=x,
            a_feat=self.a_feat,
            x_coef=self.x_coef,
            a_coef=self.a_coef,
            x_a_coef=self.x_a_coef,
            mu=self.mu,
            sigma=self.sigma,
        )
        
        params = logging_policy_params
        if params["policy_kind"] == "softmax":
            logging_policy = softmax(params["beta"] * q_x_a)
        elif params["policy_kind"] == "eps-greedy":
            logging_policy = eps_greedy_policy(q_x_a, params["k"], params["eps"])
        elif params["policy_kind"] == "normal":
            logging_policy = normal(q_x_a, mu_rate=params["mu_rate"], sigma=params["sigma"])
        elif params["policy_kind"] == "sorted_normal":
            logging_policy = sorted_normal(q_x_a, mu_rate=params["mu_rate"], sigma=params["sigma"])
        actions = sample_action_fast(logging_policy, random_state=self.random_state)
        
        params = evaluation_policy_params
        if params["policy_kind"] == "softmax":
            evaluation_policy = softmax(params["beta"] * q_x_a)
        elif params["policy_kind"] == "eps-greedy":
            evaluation_policy = eps_greedy_policy(q_x_a, params["k"], params["eps"])
        elif params["policy_kind"] == "normal":
            evaluation_policy = normal(q_x_a, mu_rate=params["mu_rate"], sigma=params["sigma"])
        elif params["policy_kind"] == "sorted_normal":
            evaluation_policy = sorted_normal(q_x_a, mu_rate=params["mu_rate"], sigma=params["sigma"])
        
        q_x_a_factual = q_x_a[np.arange(n_data), actions]
        if self.reward_type == "binary":
            rewards = self.random_.binomial(n=1, p=q_x_a_factual)
        elif self.reward_type == "continuous":
            rewards = self.random_.normal(q_x_a_factual, self.reward_std)
        
        return dict(
            n_data=n_data,
            n_actions=self.n_actions,
            x=x,
            a_feat=self.a_feat,
            actions=actions,
            r=rewards,
            q_x_a=q_x_a,
            q_x_a_factual=q_x_a_factual,
            logging_policy=logging_policy,
            pscore=logging_policy[np.arange(n_data), actions],
            evaluation_policy=evaluation_policy,
        )
    
    def calc_policy_value(self, policy_params: dict) -> float:
        if policy_params["policy_kind"] == "softmax":
            policy = softmax(policy_params["beta"] * self.q_x_a_all)
        elif policy_params["policy_kind"] == "eps-greedy":
            policy = eps_greedy_policy(self.q_x_a_all, policy_params["k"], policy_params["eps"])
        elif policy_params["policy_kind"] == "normal":
            policy = normal(self.q_x_a_all, policy_params["mu_rate"], policy_params["sigma"])
        elif policy_params["policy_kind"] == "sorted_normal":
            policy = sorted_normal(self.q_x_a_all, policy_params["mu_rate"], policy_params["sigma"])
        
        return (self.q_x_a_all * policy).sum(1).mean()


@dataclass
class KuaiRecDataset:
    n_actions: int
    reward_std: float = 2.0
    random_state: int = 12345
    
    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.q_x_a_all = self._read_q_x_a_for_all_users()
    
    def _read_q_x_a_for_all_users(self) -> np.ndarray:
        small_matrix = pd.read_csv("../data/small_matrix.csv")
        
        # ユーザリストを作成する
        user_id_set = set(small_matrix["user_id"].unique())
        
        # 全ユーザとインタラクションが発生した動画のリストを作成する
        video_id_set = None
        for user_id in user_id_set:
            mask_user_id = small_matrix["user_id"] == user_id
            now_video_id_set = set(small_matrix[mask_user_id]["video_id"].unique())
            
            if video_id_set is None:
                video_id_set = now_video_id_set
            else:
                video_id_set = video_id_set.intersection(now_video_id_set)
        
        # アクションを間引く（ランダムにアクションを選択する）
        action_rnd = check_random_state(12345)
        action_indice = action_rnd.choice(
            len(video_id_set),
            size=self.n_actions,
            replace=False
        )
        video_id_set = np.array(list(video_id_set))[action_indice]
        
        # ユーザIDとビデオIDをインデックスに変換するテーブルを作成する
        user_id2idx_dict = {user_id: i for i, user_id in enumerate(user_id_set)}
        video_id2idx_dict = {video_id: i for i, video_id in enumerate(video_id_set)}
        
        # 期待報酬テーブルを作成する
        mask_video_id = small_matrix["video_id"].isin(video_id_set)
        
        q_x_a = np.zeros((len(user_id_set), len(action_indice)))
        for idx, row in small_matrix[mask_video_id].iterrows():
            user_id, video_id = row["user_id"], row["video_id"]
            
            if video_id not in video_id2idx_dict:
                continue
            
            user_idx = user_id2idx_dict[user_id]
            video_idx = video_id2idx_dict[video_id]
            reward = row["watch_ratio"]
            
            q_x_a[user_idx, video_idx] = reward
        
        return q_x_a
    
    def generate_dataset(
        self,
        n_data: int,
        logging_policy_params: dict,
        evaluation_policy_params: dict,
    ) -> np.ndarray:
        """Generate a KuaiRec dataset."""
        data_idx = self.random_.choice(
            self.q_x_a_all.shape[0],
            size=n_data,
            replace=True
        )
        q_x_a = self.q_x_a_all[data_idx]
        
        params = logging_policy_params
        if params["policy_kind"] == "softmax":
            logging_policy = softmax(params["beta"] * q_x_a)
        elif params["policy_kind"] == "eps-greedy":
            logging_policy = eps_greedy_policy(q_x_a, params["k"], params["eps"])
        elif params["policy_kind"] == "normal":
            logging_policy = normal(q_x_a, params["mu_rate"], params["sigma"])
        elif params["policy_kind"] == "sorted_normal":
            logging_policy = sorted_normal(q_x_a, params["mu_rate"], params["sigma"])
        actions = sample_action_fast(logging_policy, random_state=self.random_state)
        
        params = evaluation_policy_params
        if params["policy_kind"] == "softmax":
            evaluation_policy = softmax(params["beta"] * q_x_a)
        elif params["policy_kind"] == "eps-greedy":
            evaluation_policy = eps_greedy_policy(q_x_a, params["k"], params["eps"])
        elif params["policy_kind"] == "normal":
            evaluation_policy = normal(q_x_a, params["mu_rate"], params["sigma"])
        elif params["policy_kind"] == "sorted_normal":
            evaluation_policy = sorted_normal(q_x_a, params["mu_rate"], params["sigma"])
        
        q_x_a_factual = q_x_a[np.arange(n_data), actions]
        rewards = self.random_.normal(q_x_a_factual, self.reward_std)
        
        return dict(
            n_data=n_data,
            n_actions=self.n_actions,
            actions=actions,
            r=rewards,
            q_x_a=q_x_a,
            q_x_a_factual=q_x_a_factual,
            logging_policy=logging_policy,
            pscore=logging_policy[np.arange(n_data), actions],
            evaluation_policy=evaluation_policy,
        )
    
    def calc_policy_value(self, policy_params: dict) -> float:
        if policy_params["policy_kind"] == "softmax":
            policy = softmax(policy_params["beta"] * self.q_x_a_all)
        elif policy_params["policy_kind"] == "eps-greedy":
            policy = eps_greedy_policy(self.q_x_a_all, policy_params["k"], policy_params["eps"])
        elif policy_params["policy_kind"] == "normal":
            policy = normal(self.q_x_a_all, policy_params["mu_rate"], policy_params["sigma"])
        elif policy_params["policy_kind"] == "sorted_normal":
            policy = sorted_normal(self.q_x_a_all, policy_params["mu_rate"], policy_params["sigma"])
        
        return (self.q_x_a_all * policy).sum(1).mean()
    
    def calc_mid_policy_value(self, policy_0_params: dict, policy_1_params: dict) -> float:
        policy_0 = sorted_normal(self.q_x_a_all, policy_0_params["mu_rate"], policy_0_params["sigma"])
        policy_1 = sorted_normal(self.q_x_a_all, policy_1_params["mu_rate"], policy_1_params["sigma"])
        policy_m = (2 * policy_0 * policy_1) / (policy_0 + policy_1)
        
        return (self.q_x_a_all * policy_m).sum(1).mean()
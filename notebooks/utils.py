import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
from sklearn.utils import check_random_state


####################################################################################################
# 実験結果を保存するディレクトリの命名ルール
####################################################################################################
def increment_path(path: Path, exist_ok=True, sep='') -> str:
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    else:
        dirs = glob.glob(f"{path}{sep}*")   # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m] # indice
        n = max(i) + 1 if i else 2 # increment number
        return f"{path}{sep}{n}"


####################################################################################################
# 推定量の性能を4つの指標で評価する関数
# ・方策選択の正解率
# ・差の推定量のバイアスの2乗
# ・差の推定量のバリアンス
# ・差の推定量のMSE
####################################################################################################
def evaluate_performance(result: pd.DataFrame):
    # 真の方策性能の差を算出する
    result["true_delta_value"] = result["true_pi_1_value"] - result["true_pi_0_value"]
    # 推定値の差を算出する
    result["delta_value"] = result["pi_1_value"] - result["pi_m_D_1_value"] + result["pi_m_D_0_value"] - result["pi_0_value"]
    
    # 方策選択の正解率を算出する
    true_selection = result["true_delta_value"] > 0
    est_selection = result["delta_value"] > 0
    accuracy = np.mean(true_selection == est_selection)
    # バイアスの二乗を算出する
    squared_bias = (result["true_delta_value"].mean() - result["delta_value"].mean()) ** 2
    # バリアンスを算出する
    variance = ((result["delta_value"] - result["delta_value"].mean()) ** 2).mean()
    # MSEを算出する
    mse = ((result["true_delta_value"] - result["delta_value"]) ** 2).mean()
    
    return {
        "accuracy": accuracy,
        "squared_bias": squared_bias,
        "variance": variance,
        "MSE": mse,
    }


####################################################################################################
#　統計的検定を行う関数
####################################################################################################
def welch_t_test(sample_a, sample_b):
    """
    Welch's t-test を計算する関数
    """
    n_a, n_b = len(sample_a), len(sample_b)
    mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
    var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1) # ddof=1 is for unbiased estimator
    
    se = np.sqrt(var_a / n_a + var_b / n_b)
    t_stat = np.abs(mean_a - mean_b) / se
    
    df_num = np.power(var_a/n_a + var_b/n_b, 2)
    df_den = np.power(var_a, 2) / (np.power(n_a, 2) * (n_a - 1)) + np.power(var_b, 2) / (np.power(n_b, 2) * (n_b - 1))
    df = df_num / df_den
    
    p = 2 * t.sf(t_stat, df)
    
    return p


####################################################################################################
# 方策に従って行動をサンプリングする関数
####################################################################################################
def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """Sampling actions from a given policy quickly."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    
    return flg.argmax(axis=1)


####################################################################################################
# 以降，方策を定義する関数
####################################################################################################

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function, used when defining a stochastic policy."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def eps_greedy_policy(
    q_x_a: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
) -> np.ndarray:
    """Define an epsilon-greedy policy based on the expected reward function."""
    is_topk = rankdata(-q_x_a, axis=1) <= k
    pi = ((1.0 - eps) / k) * is_topk
    pi += eps / q_x_a.shape[1]
    pi /= pi.sum(1)[:, np.newaxis]
    
    return pi

def normal(
    q_x_a: np.ndarray,
    mu_rate: float = 0.5,
    sigma: float = 4.0
) -> np.array:
    assert 0 <= mu_rate and mu_rate <= 1.0
    
    """Descrete Normal Distribution"""
    # パラメータ
    x_num, action_num = q_x_a.shape
    
    mu = int(action_num * mu_rate) # 中心（μ）
    sigma = sigma                  # 標準偏差（σ）
    range_of_values = np.arange(0, action_num)  # 離散的な値の範囲
    
    # ガウス関数を算出する
    pdf_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((range_of_values - mu) / sigma) ** 2)
    # 確率質量関数として正規化
    pi = pdf_values / np.sum(pdf_values)
    
    return np.broadcast_to(pi, (x_num, action_num))

def sorted_normal(
    q_x_a: np.ndarray,
    mu_rate: float = 0.5,
    sigma: float = 4.0
) -> np.array:
    assert 0 <= mu_rate and mu_rate <= 1.0
    
    normal_pi = normal(q_x_a, mu_rate=mu_rate, sigma=sigma)
    
    sort_indices = np.argsort(q_x_a, axis=1)
    inv_sort_indice = np.argsort(sort_indices, axis=1)
    sorted_normal_pi = np.take_along_axis(normal_pi, inv_sort_indice, axis=1)
    
    return sorted_normal_pi

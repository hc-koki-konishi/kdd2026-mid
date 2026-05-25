from typing import List

import numpy as np

from utils import welch_t_test


def run_AB_test(D_E_list: List[dict]):
    
    D_E_0, D_E_1 = D_E_list
    
    value_of_pi_0_D_0 = D_E_0["r"].mean()
    value_of_pi_1_D_1 = D_E_1["r"].mean()
    
    return {
        "pi_1_D_1": value_of_pi_1_D_1,
        "pi_m_D_1": 0,
        "pi_m_D_0": 0,
        "pi_0_D_0": value_of_pi_0_D_0,
        "p_value": welch_t_test(D_E_1["r"], D_E_0["r"]),
    }


def run_typical_ips(D_E_list: List[dict]):
    
    D_E_0, _ = D_E_list
    
    value_of_pi_0_D_0 = D_E_0["r"].mean()
    
    n_data = D_E_0["n_data"]
    r = D_E_0["r"]
    logging_policy = D_E_0["logging_policy"]
    evaluation_policy = D_E_0["evaluation_policy"]
    actions = D_E_0["actions"]
    
    iw = evaluation_policy[np.arange(n_data), actions] / logging_policy[np.arange(n_data), actions]
    value_of_pi_1_D_0 = (iw * r).mean()
    
    return {
        "pi_1_D_1": value_of_pi_1_D_0,
        "pi_m_D_1": 0,
        "pi_m_D_0": 0,
        "pi_0_D_0": value_of_pi_0_D_0,
        "p_value": welch_t_test(r, iw * r),
    }


def run_average_ips(D_E_list: List[dict]):
    
    each_estimated_values = {}
    
    for E_idx, D_E in enumerate(D_E_list):
        
        n_data = D_E["n_data"]
        r, actions = D_E["r"], D_E["actions"]
        logging_policy = D_E["logging_policy"]
        
        estimated_values = dict()
        for i, policy in enumerate(["logging_policy", "evaluation_policy"]):
            # IPS estimator
            pi_i = D_E[policy]
            iw = pi_i[np.arange(n_data), actions] / logging_policy[np.arange(n_data), actions]
            estimated_values[f"ips{i}"] = (iw * r).mean()
        
        each_estimated_values[E_idx] = estimated_values
    
    estimated_values = {
        0: (each_estimated_values[0]["ips0"] + each_estimated_values[1]["ips1"]) / 2,
        1: (each_estimated_values[0]["ips1"] + each_estimated_values[1]["ips0"]) / 2
    }
    
    return estimated_values


def run_mid_policy(D_E_list: List[dict], mode="mean", mu_rate_0=None, mu_rate_1=None):
    D_E_0, D_E_1 = D_E_list
    
    pi_0_D0, pi_1_D0 = D_E_0["logging_policy"], D_E_0["evaluation_policy"]
    pi_0_D1, pi_1_D1 = D_E_1["evaluation_policy"], D_E_1["logging_policy"]
    
    assert mode in ["mean", 'compare', 'opt']
    
    if mode == "mean":
        pi_mid_D1 = (pi_0_D1 + pi_1_D1) / 2
        pi_mid_D0 = (pi_0_D0 + pi_1_D0) / 2
    elif mode == "compare":
        pi_mid_D1 = np.where(pi_1_D1 < pi_0_D1, pi_1_D1, pi_0_D1)
        pi_mid_D0 = np.where(pi_1_D0 < pi_0_D0, pi_1_D0, pi_0_D0)
    elif mode == "opt":
        pi_mid_D1 = (2 * pi_1_D1 * pi_0_D1) / (pi_1_D1 + pi_0_D1)
        pi_mid_D0 = (2 * pi_1_D0 * pi_0_D0) / (pi_1_D0 + pi_0_D0)
    
    # V(pi_1, D1) - V(pi_m, D1)
    n1_data = D_E_1["n_data"]
    actions1 = D_E_1["actions"]
    r1 = D_E_1["r"]
    value_of_pi_1_D_1 = r1.mean()
    iw1 = pi_mid_D1[np.arange(n1_data), actions1] / pi_1_D1[np.arange(n1_data), actions1]
    value_of_pi_m_D_1 = (iw1 * r1).mean()
    
    # V(pi_m, D0) - V(pi_0, D0)
    n0_data = D_E_0["n_data"]
    actions0 = D_E_0["actions"]
    r0 = D_E_0["r"]
    value_of_pi_0_D_0 = r0.mean()
    iw0 = pi_mid_D0[np.arange(n0_data), actions0] / pi_0_D0[np.arange(n0_data), actions0]
    value_of_pi_m_D_0 = (iw0 * r0).mean()
    
    return {
        "pi_1_D_1": np.round(value_of_pi_1_D_1, 10),
        "pi_m_D_1": np.round(value_of_pi_m_D_1, 10),
        "pi_m_D_0": np.round(value_of_pi_m_D_0, 10),
        "pi_0_D_0": np.round(value_of_pi_0_D_0, 10),
        "p_value": welch_t_test(
            r1 * (1 - iw1),
            r0 * (1 - iw0),
        ),
    }


def run_experiment(D_E_list: List[dict]) -> List[dict]:
    """A/Bテストの推定"""
    AB_test = run_AB_test(D_E_list)
    typical_ips = run_typical_ips(D_E_list)
    average_ips = run_mid_policy(D_E_list, mode="mean")
    mid_policy = run_mid_policy(D_E_list, mode="compare")
    opt_policy = run_mid_policy(D_E_list, mode="opt")
    
    values_of_pi_1 = {
        "AVG": AB_test["pi_1_D_1"],
        "IPS": typical_ips["pi_1_D_1"],
        "AVG_IPS": average_ips["pi_1_D_1"],
        "MIN": mid_policy["pi_1_D_1"],
        "MID": opt_policy["pi_1_D_1"],
    }
    values_of_pi_m_D_1 = {
        "AVG": AB_test["pi_m_D_1"],
        "IPS": typical_ips["pi_m_D_1"],
        "AVG_IPS": average_ips["pi_m_D_1"],
        "MIN": mid_policy["pi_m_D_1"],
        "MID": opt_policy["pi_m_D_1"],
    }
    values_of_pi_m_D_0 = {
        "AVG": AB_test["pi_m_D_0"],
        "IPS": typical_ips["pi_m_D_0"],
        "AVG_IPS": average_ips["pi_m_D_0"],
        "MIN": mid_policy["pi_m_D_0"],
        "MID": opt_policy["pi_m_D_0"],
    }
    values_of_pi_0 = {
        "AVG": AB_test["pi_0_D_0"],
        "IPS": typical_ips["pi_0_D_0"],
        "AVG_IPS": average_ips["pi_0_D_0"],
        "MIN": mid_policy["pi_0_D_0"],
        "MID": opt_policy["pi_0_D_0"],
    }
    p_value = {
        "AVG": AB_test["p_value"],
        "IPS": typical_ips["p_value"],
        "AVG_IPS": average_ips["p_value"],
        "MIN": mid_policy["p_value"],
        "MID": opt_policy["p_value"],
    }
    
    policy_comparison = dict()
    for method in values_of_pi_1:
        estimated_delta_value = (
            values_of_pi_1[method]
            - values_of_pi_m_D_1[method]
            + values_of_pi_m_D_0[method]
            - values_of_pi_0[method]
        )
        
        is_pi_1_better = np.int32(estimated_delta_value > 0)
        policy_comparison[method] = is_pi_1_better - (1 - is_pi_1_better)
    
    return (
        values_of_pi_1,
        values_of_pi_m_D_1,
        values_of_pi_m_D_0,
        values_of_pi_0,
        policy_comparison,
        p_value,
    )

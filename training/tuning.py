from __future__ import annotations

from copy import deepcopy


def build_trial_configs(base_config: dict) -> list[dict]:
    trials = []
    candidate_lrs = [base_config["training"].get("learning_rate_stage1", 1e-3), 5e-4, 2e-4]
    candidate_dropouts = [base_config["model"].get("dropout_rate", 0.3), 0.4, 0.25]
    candidate_dense = [base_config["model"].get("dense_units", 256), 128, 384]
    max_trials = int(base_config["training"].get("trials", 3))

    for lr, dr, dense in zip(candidate_lrs, candidate_dropouts, candidate_dense):
        cfg = deepcopy(base_config)
        cfg["training"]["learning_rate_stage1"] = lr
        cfg["model"]["dropout_rate"] = dr
        cfg["model"]["dense_units"] = dense
        trials.append(cfg)
        if len(trials) >= max_trials:
            break
    return trials

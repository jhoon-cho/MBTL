from statistics import mean
from typing import Dict, List


def merge_rollout_metrics(metrics_per_rollout: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Averages the given dicts, assuming they have the same keys
    """
    return {
        k: mean(metric_dict[k] for metric_dict in metrics_per_rollout if k in metric_dict)
        for k in max(metrics_per_rollout, key=lambda d: len(d))
    }
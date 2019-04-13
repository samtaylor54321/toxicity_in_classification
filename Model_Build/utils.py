import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def get_identities():
    return [
        'asian',
        'atheist',
        'bisexual',
        'black',
        'buddhist',
        'christian',
        'female',
        'heterosexual',
        'hindu',
        'homosexual_gay_or_lesbian',
        'intellectual_or_learning_disability',
        'jewish',
        'latino',
        'male',
        'muslim',
        'other_disability',
        'other_gender',
        'other_race_or_ethnicity',
        'other_religion',
        'other_sexual_orientation',
        'physical_disability',
        'psychiatric_or_mental_illness',
        'transgender',
        'white'
    ]

def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, y_true, y_pred):
    mask = df[subgroup]
    return compute_auc(y_true[mask], y_pred[mask])


def compute_bpsn_auc(df, subgroup, y_true, y_pred):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    mask = (df[subgroup] & ~y_true) | (~df[subgroup] & y_true)
    return compute_auc(y_true[mask], y_pred[mask])


def compute_bnsp_auc(df, subgroup, y_true, y_pred):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    mask = (df[subgroup] & y_true) | (~df[subgroup] & ~y_true)
    return compute_auc(y_true[mask], y_pred[mask])


def compute_bias_metrics_for_model(dataset, subgroups, y_true,  y_pred):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {'subgroup': subgroup,
                  'subgroup_size': len(dataset[dataset[subgroup]]),
                  'subgroup_auc': compute_subgroup_auc(dataset, subgroup, y_true, y_pred),
                  'bpsn_auc': compute_bpsn_auc(dataset, subgroup, y_true, y_pred),
                  'bnsp_auc': compute_bnsp_auc(dataset, subgroup, y_true, y_pred)}
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, power=-5, model_weight=0.25):
    bias_score = np.nanmean([
        power_mean(bias_df['subgroup_auc'], power),
        power_mean(bias_df['bpsn_auc'], power),
        power_mean(bias_df['bnsp_auc'], power)
    ])
    return (model_weight * overall_auc) + ((1 - model_weight) * bias_score)

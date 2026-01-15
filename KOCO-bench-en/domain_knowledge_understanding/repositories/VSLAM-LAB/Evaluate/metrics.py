import numpy as np


def rmse_ate(trajectory_1, trajectory_2):
    alignment_error = trajectory_1 - trajectory_2
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 1))
    rmse_ate_value = float(np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
    return rmse_ate_value


def recall_ate(trajectory_1, trajectory_2, recall=0.1):
    alignment_error = trajectory_1 - trajectory_2
    e_t = np.sqrt((alignment_error['tx'] ** 2 + alignment_error['ty'] ** 2 + alignment_error['tz'] ** 2))

    total_elements = e_t.size
    smaller_than_th = (e_t < recall).sum().sum()
    return (smaller_than_th / total_elements) * 100

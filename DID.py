import numpy as np
import pandas as pd

def DID(M, mask):

    """Difference in difference estimation using numpy"""
    M = M * mask
    treated_rows = np.where(np.mean(mask, axis=1) < 1)[0]
    control_rows = np.setdiff1d(np.arange(M.shape[0]), treated_rows)
    num_control = len(control_rows)
    M_control_rows = M[control_rows, :]
    M_pred = M.copy()
    
    tr_row_miss_cols = np.where(mask[treated_rows, :] == 0)
    control_cols = np.setdiff1d(np.arange(M.shape[1]), tr_row_miss_cols[1])
    W = (1 / num_control) * np.ones((num_control, 1))
    
    mu = (np.mean(M[np.ix_(treated_rows, control_cols)], axis=1) - 
          np.mean(M[np.ix_(control_rows, control_cols)], axis=1).reshape(-1, 1))

    M_pred_this_row = (M_control_rows.T @ W).T + mu
    M_pred[treated_rows, :] = M_pred_this_row
    return M_pred

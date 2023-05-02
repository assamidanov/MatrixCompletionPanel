import numpy as np

def my_synth(A, b, niter, rel_tol):
    row_m = np.mean(A, axis=0)
    A -= row_m
    b -= row_m
    max_norm = np.max(np.abs(A))
    A /= max_norm
    b /= max_norm
    m, n = A.shape
    w = np.full((n, 1), 1/n)
    J = A.T @ A
    g = A.T @ b
    obj_val = w.T @ J @ w - 2 * w.T @ g + b.T @ b
    alpha = 1

    for t in range(niter):
        step_size = alpha
        grad = 2 * (J @ w - g)
        w_np = mirror_dec(w, step_size, grad)
        obj_val_n = w_np.T @ J @ w_np - 2 * w_np.T @ g + b.T @ b
        rel_imp = (obj_val - obj_val_n) / obj_val
        if obj_val_n < 1e-14:
            w = w_np
            break
        if rel_imp <= 0:
            alpha *= 0.95
        else:
            w = w_np
            obj_val = obj_val_n
        if rel_imp > 0 and rel_imp < rel_tol:
            w = w_np
            break

    return w

def mirror_dec(v, alpha, grad):
    h = v * np.exp(-alpha * grad)
    h /= np.sum(h)
    return h

def rows(M, mask, niter=10000, rel_tol=1e-8):
    M *= mask
    treated_rows = np.where(np.mean(mask, axis=1) < 1)[0]
    control_rows = np.setdiff1d(np.arange(M.shape[0]), treated_rows)
    num_treated = len(treated_rows)
    num_control = len(control_rows)
    M_control_rows = M[control_rows, :]
    M_pred = M.copy()

    for l in range(num_treated):
        tr_row_pred = treated_rows[l]
        tr_row_miss = np.where(mask[treated_rows[l], :] == 0)[0]
        A = M[control_rows, :][:, np.setdiff1d(np.arange(M.shape[1]), tr_row_miss)].T
        b = M[tr_row_pred, :][np.setdiff1d(np.arange(M.shape[1]), tr_row_miss)].reshape(-1, 1)
        if num_treated > 50:
            niter = 200
        W = my_synth(A, b, niter, rel_tol)
        M_pred_this_row = M[control_rows, :].T @ W
        M_pred[treated_rows[l], :] = M_pred_this_row.squeeze()

    return M_pred

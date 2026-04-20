
import numpy as np

def apply_piecewise_linear(S, p_dict):
    """
    Calculates the piecewise linear sag model in counts.
    p_dict must contain 'knots', 'values', and 'sigma_tail'.
    """
    knots = p_dict['knots']
    values = p_dict['values']
    sigma_tail = p_dict['sigma_tail']
    
    k1, k2, k3 = knots
    v1, v2, v3 = values
    res = np.zeros_like(S)
    
    # Region 1: <= k1
    m1 = S <= k1
    res[m1] = v1
    # Region 2: k1 to k2
    m2 = (S > k1) & (S <= k2)
    if np.any(m2):
        res[m2] = np.interp(S[m2], [k1, k2], [v1, v2])
    # Region 3: k2 to k3
    m3 = (S > k2) & (S <= k3)
    if np.any(m3):
        res[m3] = np.interp(S[m3], [k2, k3], [v2, v3])
    # Region 4: > k3 (linear tail)
    m4 = S > k3
    if np.any(m4):
        res[m4] = v3 + sigma_tail * (S[m4] - k3)
    return res

def apply_multiplicative_correction(img, p):
    """
    Applies the full joint multiplicative model and robust bias estimation.
    p must be a dictionary with 'top' and 'bot' halves.
    """
    half = 512
    S_obs = np.sum(img, axis=1)
    st, sb = S_obs[:half], S_obs[half:]
    
    # Unitless P = model(counts) / 2500
    ps_t = apply_piecewise_linear(st, p['top']) / 2500.0
    ps_b = apply_piecewise_linear(sb, p['bot']) / 2500.0
    
    # Activation masks
    mask_t = (st > p['top']['knots'][0]).astype(float)
    mask_b = (sb > p['bot']['knots'][0]).astype(float)
    
    # 1. Top half including bottom echo
    dt = np.maximum(1 - ps_t[:, None] * mask_t[:, None] - p['top']['echo_scale'] * ps_b[:, None] * mask_b[:, None], 0.1)
    zt = (img[:half] + p['top']['beta'] * mask_t[:, None]) / dt
    
    # 2. Bottom half including top echo
    db = np.maximum(1 - ps_b[:, None] * mask_b[:, None] - p['bot']['echo_scale'] * ps_t[:, None] * mask_t[:, None], 0.1)
    zb = (img[half:] + p['bot']['beta'] * mask_b[:, None]) / db
    
    # 3. Robust bias estimation from corrected data z
    is_unsag_t = (st <= p['top']['knots'][0]) & (sb <= p['bot']['knots'][0])
    is_unsag_b = is_unsag_t # Row matched
    
    if np.any(is_unsag_t):
        bt = np.median(zt[is_unsag_t], axis=0)
        bb = np.median(zb[is_unsag_b], axis=0)
    else:
        # Fallback to standard safe rows
        bt = np.median(zt[150:363], axis=0)
        bb = np.median(zb[150:363], axis=0)
    
    return np.concatenate([zt - bt[None, :], zb - bb[None, :]], axis=0)

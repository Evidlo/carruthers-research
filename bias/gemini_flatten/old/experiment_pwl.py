
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import scipy.optimize as opt
import json

# We will implement a Piecewise Linear (PWL) sag model
# instead of a single threshold.

def apply_pwl_correction(img, knots, top_values, bot_values):
    # knots: list of S values
    # top_values: multipliers for each knot (top half)
    # bot_values: multipliers for each knot (bot half)
    half = img.shape[0] // 2
    S_obs = np.sum(img, axis=1)
    
    res = np.zeros_like(img)
    
    # Interp function for multipliers
    # Sag(S) = interp(S, knots, values)
    # We want z = y / (1 - Sag(S)) ? No, let's keep it simple: 
    # y = z * (1 - Sag_primary(S) - Sag_echo(S')) - Beta * Indicator
    # Actually let's use a simpler additive/multiplicative combo
    
    mult_top = np.interp(S_obs[:half], knots, top_values)
    mult_bot = np.interp(S_obs[half:], knots, bot_values)
    
    # Mirror echo
    echo_top = np.interp(S_obs[half:], knots, top_values) # Echo uses same model?
    echo_bot = np.interp(S_obs[:half], knots, bot_values)

    # Let's use the exact model from AGENT.md but replace sigma*S + beta with PWL(S)
    # y = (x+b) - PWL_primary(S) * (x+b) - PWL_echo(S') * (x+b)
    # z = y / (1 - PWL_primary(S) - PWL_echo(S'))
    
    def get_denom(S, S_p, v_p, v_e):
        p_sag = np.interp(S, knots, v_p)
        e_sag = np.interp(S_p, knots, v_e)
        return np.maximum(1 - p_sag - e_sag, 0.1)

    # Need 4 PWL functions? PrimaryTop, PrimaryBot, EchoTop, EchoBot
    # Let's start with just 2: Primary and Echo (shared across halves)
    # No, top and bottom are different physically.
    
    # Simple version for now to test if PWL helps:
    # Just primary sag per half, no echo for a second
    res[:half] = img[:half] / (1 - np.interp(S_obs[:half], knots, top_values)[:, None])
    res[half:] = img[half:] / (1 - np.interp(S_obs[half:], knots, bot_values)[:, None])
    
    return res

# I will stick to the AGENT.md formula but refine the sigma*S part.
# Variation 2.5 counts is small, maybe we just need a better alpha/sigma fit.

if __name__ == "__main__":
    # Let's try to fit a more flexible model to the row profile
    print("Beginning PWL experiment...")
    # ...


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import json

data = np.load('gemini_flatten/sag_curve_halves_data.npz')

def model_ramp(S, alpha, beta, sigma):
    return - (beta * (S > alpha) + sigma * np.maximum(0, S - alpha))

def model_linear(S, beta, sigma):
    return - (beta + sigma * S)

results = {}

plt.figure(figsize=(12, 6))

for i, side in enumerate(['top', 'bot']):
    S_data = data[f'S_{side}']
    res_data = data[f'res_{side}']
    
    mask = ~np.isnan(res_data)
    S_data = S_data[mask]
    res_data = res_data[mask]
    
    def obj_ramp(p):
        alpha, beta, sigma = p
        pred = model_ramp(S_data, alpha, beta, sigma)
        return np.mean((res_data - pred)**2)
        
    def obj_linear(p):
        beta, sigma = p
        pred = model_linear(S_data, beta, sigma)
        return np.mean((res_data - pred)**2)

    s_min, s_max = S_data.min(), S_data.max()
    print(f"  Side {side} S range: {s_min:.2f} to {s_max:.2f}")
    
    res_ramp = opt.differential_evolution(obj_ramp, [(s_min * 0.5, s_max), (0, 30), (0, 5e-5)], popsize=20, maxiter=60)
    res_lin = opt.differential_evolution(obj_linear, [(-100, 100), (0, 5e-5)], popsize=20, maxiter=60)
    
    print(f"  {side} Ramp Loss: {res_ramp.fun:.4f}")
    print(f"  {side} Linear Loss: {res_lin.fun:.4f}")
    
    # Select better model
    if res_ramp.fun < res_lin.fun:
        popt = res_ramp.x
        model_type = 'ramp'
        print(f"Side {side} (Ramp): alpha={popt[0]:.2f}, beta={popt[1]:.2f}, sigma={popt[2]:.2e}")
    else:
        popt = res_lin.x
        model_type = 'linear'
        print(f"Side {side} (Linear): beta={popt[0]:.2f}, sigma={popt[1]:.2e}")
    
    results[side] = {'type': model_type, 'params': list(popt)}
    
    plt.subplot(1, 2, i+1)
    plt.scatter(S_data, res_data, alpha=0.1, s=1)
    S_test = np.linspace(S_data.min(), S_data.max(), 1000)
    if model_type == 'ramp':
        plt.plot(S_test, model_ramp(S_test, *popt), 'r-', label='Ramp Fit')
    else:
        plt.plot(S_test, model_linear(S_test, *popt), 'g-', label='Linear Fit')
    plt.title(f'{side.capitalize()} Half Fit')
    plt.legend()

plt.tight_layout()
plt.savefig('/www/gemini/sag_model_halves_fit_robust.png')
with open('gemini_flatten/params_halves.json', 'w') as f:
    json.dump(results, f, indent=4)

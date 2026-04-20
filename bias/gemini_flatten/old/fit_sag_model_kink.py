
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import json

data = np.load('gemini_flatten/sag_curve_halves_data.npz')

def model_kink(S, alpha, beta, sigma):
    # beta: additive jump at alpha
    # sigma: linear slope after alpha
    # baseline is 0
    res = np.zeros_like(S)
    mask = S > alpha
    res[mask] = - (beta + sigma * (S[mask] - alpha))
    return res

results = {}

plt.figure(figsize=(12, 6))

for i, side in enumerate(['top', 'bot']):
    S_data = data[f'S_{side}']
    res_data = data[f'res_{side}']
    mask = ~np.isnan(res_data)
    S_data, res_data = S_data[mask], res_data[mask]
    
    # Heuristic initial guesses from binned plot
    if side == 'top':
        a_guess = 2.83e6
    else:
        a_guess = 2.39e6

    def obj(p):
        alpha, beta, sigma = p
        pred = model_kink(S_data, alpha, beta, sigma)
        return np.mean((res_data - pred)**2)

    # Use DE around the suspected threshold
    bounds = [(S_data.min(), S_data.min() + 0.1*(S_data.max()-S_data.min())), (0, 30), (0, 2e-5)]
    res = opt.differential_evolution(obj, bounds, popsize=20, maxiter=50)
    
    popt = res.x
    results[side] = list(popt)
    print(f"Side {side} Kink: alpha={popt[0]:.2f}, beta={popt[1]:.2f}, sigma={popt[2]:.2e}")
    
    plt.subplot(1, 2, i+1)
    plt.scatter(S_data, res_data, alpha=0.1, s=1)
    S_test = np.linspace(S_data.min(), S_data.max(), 1000)
    plt.plot(S_test, model_kink(S_test, *popt), 'r-', label='Kink Fit')
    plt.axvline(popt[0], color='k', linestyle='--', label='Alpha')
    plt.title(f'{side.capitalize()} Half Kink Fit')
    plt.legend()

plt.tight_layout()
plt.savefig('/www/gemini/sag_model_halves_kink_fit.png')
with open('gemini_flatten/params_halves_kink.json', 'w') as f:
    json.dump(results, f, indent=4)

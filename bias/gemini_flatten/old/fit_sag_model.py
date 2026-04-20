
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = np.load('gemini_flatten/sag_curve_halves_data.npz')

def model_ramp(S, alpha, beta, sigma):
    diff = S - alpha
    return - (beta * (S > alpha) + sigma * np.maximum(0, diff))

plt.figure(figsize=(12, 6))

for i, side in enumerate(['top', 'bot']):
    S = data[f'S_{side}']
    res = data[f'res_{side}']
    
    mask = ~np.isnan(res)
    S = S[mask]
    res = res[mask]
    
    # Fit Ramp Model
    popt, _ = opt.curve_fit(model_ramp, S, res, p0=[2.8e6, 8, 3e-6])
    print(f"Side {side}: alpha={popt[0]:.2f}, beta={popt[1]:.2f}, sigma={popt[2]:.2e}")
    
    plt.subplot(1, 2, i+1)
    plt.scatter(S, res, alpha=0.1, s=1)
    S_test = np.linspace(S.min(), S.max(), 1000)
    plt.plot(S_test, model_ramp(S_test, *popt), 'r-', label='Fit')
    plt.title(f'{side.capitalize()} Half Fit')
    plt.xlabel('Row Sum (S)')
    plt.ylabel('Residual (y - b)')
    plt.legend()

plt.tight_layout()
plt.savefig('/www/gemini/sag_model_halves_fit.png')
print("Halves fit complete.")

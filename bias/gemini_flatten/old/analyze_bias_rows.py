
import numpy as np
import os
import sys
# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load
import matplotlib.pyplot as plt

if __name__ == "__main__":
    files = [
        'images_20260111/oob_nfi_l0.pkl', 
        'images_20260113/oob_nfi_l0.pkl',
        'images_20260115/oob_nfi_l0.pkl', 
        'images_20260117/oob_nfi_l0.pkl', 
        'images_20260119/oob_nfi_l0.pkl'
    ]
    
    plt.figure(figsize=(10, 6))
    for f in files:
        img = np.asarray(load(f))
        S = np.sum(img, axis=1)
        plt.plot(S, label=os.path.basename(f), alpha=0.5)
    
    plt.axvline(150, color='k', linestyle='--')
    plt.axvline(363, color='k', linestyle='--')
    plt.axvline(662, color='k', linestyle='--')
    plt.axvline(875, color='k', linestyle='--')
    plt.title('Row Sums (S) and Static Bias Regions')
    plt.xlabel('Row Index')
    plt.ylabel('Row Sum (S)')
    plt.legend()
    plt.savefig('/www/gemini/bias_regions_S.png')
    
    print("Static bias regions (150-363, 662-875) marked on plot.")
    # Check if S in these regions is truly the floor
    img = np.asarray(load(files[0]))
    S = np.sum(img, axis=1)
    S_nonsag = np.concatenate([S[150:363], S[662:875]])
    print(f"Mean S in nonsag regions: {np.mean(S_nonsag):.2f}")
    print(f"Min S overall: {np.min(S):.2f}")

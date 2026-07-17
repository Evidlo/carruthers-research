import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    npz_path = '/home/evan/sync/research/carruthers/bias/opencode_cjrelate/cj.npz'
    d = np.load(npz_path, allow_pickle=True)
    data = d['arr_0'].item()

    top = data['top']
    dates = ['0316', '0317', '0318', '0319']

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ci = top[i]
            cj = top[j]

            if i == j:
                vals = ci[np.isfinite(ci)]
                ax.hist(vals, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
                ax.set_title(f'{dates[i]}', fontsize=11)
                ax.tick_params(axis='both', labelsize=7)
            else:
                mask = np.isfinite(ci) & np.isfinite(cj)
                x = cj[mask]
                y = ci[mask]

                if len(x) > 0:
                    vmin = min(x.min(), y.min())
                    vmax = max(x.max(), y.max())
                    pad = (vmax - vmin) * 0.05
                    if pad == 0:
                        pad = 1e-3

                    ax.scatter(x, y, s=5, alpha=0.4, c='royalblue', edgecolors='none')
                    ax.set_xlim(vmin - pad, vmax + pad)
                    ax.set_ylim(vmin - pad, vmax + pad)
                    ax.set_aspect('equal')

                ax.set_xlabel(f'{dates[j]}', fontsize=9)
                ax.set_ylabel(f'{dates[i]}', fontsize=9)
                ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    out_path = '/www/cj_scatter.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()

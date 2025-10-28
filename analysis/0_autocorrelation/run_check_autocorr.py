#!/usr/bin/env python3
"""Run the autocorrelation checks from the notebook as a script.
# To run from terminal: python3 0_check_autocorr.py

Creates two PNG figures in the current directory:
- 0_fig_autocorr_gw.png
- 0_fig_autocorr_que.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths (repo-local)
GW_CLEAN_PATH = Path('../../data/cr2sub/cr2sub_v1.1_gwl_mon_clean.csv')
Q_MON_PATH = Path('../../data/camels/q_mm_dga_mon_ts_1960_2025.csv')
FIG_GW_PATH = Path('0_fig_autocorr_gw.png')
FIG_Q_PATH = Path('0_fig_autocorr_q.png')

# --- Helper functions ---
def load_monthly_wide(csv_path: Path, date_col: str = 'date') -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df

def compute_autocorr_n_eff(df: pd.DataFrame):
    autocorrs, n_effective, n_obs = [], [], []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 3 or np.isclose(np.std(s), 0):
            continue
        s = s.sort_index()
        r1 = s.autocorr(lag=1)
        n = len(s)
        n_eff = n * (1 - r1) / (1 + r1) if not np.isclose(1 + r1, 0) else np.nan
        autocorrs.append(r1)
        n_effective.append(n_eff)
        n_obs.append(n)
    return np.array(autocorrs), np.array(n_effective), np.array(n_obs)

def deseasonalize(df: pd.DataFrame) -> pd.DataFrame:
    return df - df.groupby(df.index.month).transform('mean')

def plot_autocorr_panels(autocorr_raw, n_eff_raw, n_obs_raw,
                         autocorr_an, n_eff_an, n_obs_an,
                         title_prefix: str):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].hist(n_obs_raw, bins=30, alpha=0.5, label='N_obs (raw)', color='grey', edgecolor='k', lw=0.4)
    axs[0, 0].hist(n_eff_raw, bins=30, alpha=0.7, label='N_eff (raw)', color='orange', edgecolor='k', lw=0.4)
    axs[0, 0].set_title(f"Raw {title_prefix} Series: N_obs vs N_eff")
    axs[0, 0].set_xlabel("Number of observations")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].legend()

    axs[0, 1].hist(autocorr_raw, bins=30, color='skyblue', edgecolor='k', lw=0.5)
    axs[0, 1].set_title(f"Raw {title_prefix} Series: Lag-1 Autocorrelation")
    axs[0, 1].set_xlabel("Autocorrelation (lag=1)")
    axs[0, 1].set_ylabel("Count")

    axs[1, 0].hist(n_obs_an, bins=30, alpha=0.5, label='N_obs (anomalies)', color='grey', edgecolor='k', lw=0.4)
    axs[1, 0].hist(n_eff_an, bins=30, alpha=0.7, label='N_eff (anomalies)', color='orange', edgecolor='k', lw=0.4)
    axs[1, 0].set_title(f"{title_prefix} anomalies: N_obs vs N_eff")
    axs[1, 0].set_xlabel("Number of observations")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].legend()

    axs[1, 1].hist(autocorr_an, bins=30, color='lightgreen', edgecolor='k', lw=0.5)
    axs[1, 1].set_title(f"{title_prefix} anomalies: Lag-1 Autocorrelation")
    axs[1, 1].set_xlabel("Autocorrelation (lag=1)")
    axs[1, 1].set_ylabel("Count")

    for ax in axs.flat:
        ax.grid(ls='--', alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # === Groundwater (GW) ===
    print('Loading GW data from', GW_CLEAN_PATH)
    gw_raw = load_monthly_wide(GW_CLEAN_PATH)
    if 'date' in gw_raw.columns:
        raise ValueError('Unexpected "date" column after indexing GW data.')

    gw_an = deseasonalize(gw_raw)

    gw_autocorr_raw, gw_n_eff_raw, gw_n_obs_raw = compute_autocorr_n_eff(gw_raw)
    gw_autocorr_an, gw_n_eff_an, gw_n_obs_an = compute_autocorr_n_eff(gw_an)

    fig_gw = plot_autocorr_panels(
        gw_autocorr_raw, gw_n_eff_raw, gw_n_obs_raw,
        gw_autocorr_an, gw_n_eff_an, gw_n_obs_an,
        title_prefix='GW'
    )
    fig_gw.savefig(FIG_GW_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig_gw)

    # === Streamflow (Q) ===
    print('Loading Q data from', Q_MON_PATH)
    q_raw = pd.read_csv(Q_MON_PATH, parse_dates=['Index'], index_col='Index').sort_index()
    q_an = deseasonalize(q_raw)

    q_autocorr_raw, q_n_eff_raw, q_n_obs_raw = compute_autocorr_n_eff(q_raw)
    q_autocorr_an, q_n_eff_an, q_n_obs_an = compute_autocorr_n_eff(q_an)

    fig_q = plot_autocorr_panels(
        q_autocorr_raw, q_n_eff_raw, q_n_obs_raw,
        q_autocorr_an, q_n_eff_an, q_n_obs_an,
        title_prefix='Q'
    )
    fig_q.savefig(FIG_Q_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig_q)

    print(f'Saved GW figure to {FIG_GW_PATH.resolve()}')
    print(f'Saved Q figure to {FIG_Q_PATH.resolve()}')


if __name__ == '__main__':
    main()

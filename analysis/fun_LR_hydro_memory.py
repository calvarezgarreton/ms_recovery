#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from matplotlib import rc
import os
from statsmodels.api import OLS, add_constant
from collections import OrderedDict

rc('mathtext', default='regular')

# ========================== FUNCTIONS ============================

# === Functions to get basin_id (gauge_id) for corresponding well from join tables ===
def get_gauge_id(cod, path_join_table_camels, path_join_table_bna):
    # Try CAMELS first
    join_camels = pd.read_csv(path_join_table_camels, dtype=str)
    subset = join_camels[join_camels["well_id"] == cod]
    if not subset.empty:
        gauge_id = str(subset.sort_values("area", ascending=True, key=lambda x: pd.to_numeric(x, errors='coerce'))["gauge_id"].iloc[0])
        predictor_source = "camels"
        return gauge_id, predictor_source
    # Try BNA if CAMELS failed
    join_bna = pd.read_csv(path_join_table_bna, dtype=str)
    subset = join_bna[join_bna["well_id"] == cod]
    if not subset.empty:
        gauge_id = str(subset.sort_values("area", ascending=True, key=lambda x: pd.to_numeric(x, errors='coerce'))["bna_id"].iloc[0])
        predictor_source = "bna"
        return gauge_id, predictor_source
    return None

from collections import OrderedDict

def make_lag_ranges(lag_increase=1, n_windows=10, incr_type = 1):
    """
    Build an expanding set of lag windows.
    Example (lag_increase=1): (0,0), (1,2), (3,5), (6,9), ...
    Returns:
        - ranges: OrderedDict with names and (start, end) tuples
        - widths: list of window widths (end - start + 1)
    """
    ranges = OrderedDict()
    widths = []
    start, width = 0, 1

    for i in range(n_windows):
        end = start + width - 1
        name = f"lag_{start}_{end}"
        ranges[name] = (start, end)
        widths.append(width)
        start = end + 1
        
        if incr_type == 1:
            width = width + lag_increase #lineal
            incr_name = 'w(i+1) = w(i) + incr'
        
        if incr_type == 2:        
            width = width + lag_increase + 1 #lineal incremental
            incr_name = 'w(i+1) = w(i) + incr + 1'
        
        if incr_type == 3:
            width = width * lag_increase #multiplicativo
            incr_name = 'w(i+1) = w(i)*incr'
               
    return ranges, widths, incr_name


def rolling_predictors(series, lag_ranges=None, standardize=True):
    """
    Given a (monthly) pd.Series, build predictors as rolling means over each
    lag window, shifted to start at the given lag. Returns a DataFrame with
    one column per window (same names as lag_ranges keys).
    Ensures only full windows (no NaNs) are used in calculations.
    """
    if lag_ranges is None:
        lag_ranges = make_lag_ranges()

    s = pd.Series(series).sort_index()
    preds = {}

    for name, (lag_start, lag_end) in lag_ranges.items():
        win = lag_end - lag_start + 1
        preds[name] = s.rolling(window=win, min_periods=win).mean().shift(lag_start)

    df = pd.DataFrame(preds, index=s.index)

    if standardize:
        df = (df - df.mean()) / df.std()

    return df


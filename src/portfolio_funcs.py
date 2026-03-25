"""
Phase 2: Portfolio Engine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.interest_funcs import calculate_risk_premium


def get_portfolio_pd(model_pipeline, data: pd.DataFrame):
    """
    Inference helper: Gets PDs for the entire portfolio.
    """
    # The pipeline handles all preprocessing, no manual cleaning needed
    return model_pipeline.predict_proba(data)[:, 1]


def expected_profit_list(pds, r, ead, lgd, max_threshold=0.7):
    """
    Calculates the EP curve.
    Formula: EP = EAD * r * (1 - PD) - EAD * LGD * PD
    """
    thresholds = np.arange(0, max_threshold, 0.01)
    expected_profits = []
    
    for max_pd_allowed in thresholds:
        # Mask for loans that meet our risk appetite
        approved = pds < max_pd_allowed
        
        # Calculate EP for the portfolio segment
        ep = np.sum((1 - pds[approved]) * r[approved] * ead) - np.sum(pds[approved] * lgd * ead)
        expected_profits.append(ep)
        
    return expected_profits, thresholds


def portfolio_max_profit_and_threshold(expected_profits, thresholds):
    """
    Identifies the optimal risk cutoff point.
    """
    expected_profits_array = np.array(expected_profits)
    max_index = np.argmax(expected_profits_array)
    return expected_profits_array[max_index], thresholds[max_index]


def plot_expected_profit(thresholds, expected_profits, y_value, x_value, label='Max Profit'):
    """
    Visualizes the portfolio profit curve.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#121212')
    ax.set_facecolor('#121212')

    # plot EP curve and the point of max profit
    ax.plot(thresholds * 100, expected_profits, color='#89CFF0', linewidth=2, label="Expected Profit")
    ax.scatter(x_value * 100, y_value, color='#FFB347', s=75, edgecolors='white', zorder=5, label=label)
    
    # title
    fig.suptitle("Portfolio Expected Profit vs. Policy Threshold", fontsize=16, fontweight='normal', color='white', x=0.2, ha='left', y=0.93)       
    plt.subplots_adjust(top=0.85)
    # axes labels
    ax.set_xlabel("PD Threshold (%)", fontsize=14, labelpad=15)
    ax.set_ylabel("Expected Profit (R)", fontsize=14, labelpad=15)
    # y-axis units
    formatter = ticker.EngFormatter(unit='', places=1)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_visible(False)
    # grid and legend
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.2)
    sns.despine(ax=ax, top=False, right=False)
    ax.legend(frameon=False, loc='upper left')
    
    return fig

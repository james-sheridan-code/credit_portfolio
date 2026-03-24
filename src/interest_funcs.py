"""
Interest rate calcualtions
"""

import numpy as np

def calculate_risk_premium(pd):
    """Risk premium for all risk types"""
    conditions = [pd < 0.01, pd < 0.03, pd < 0.08, pd < 0.15]
    choices = [-0.005, 0.02, 0.05, 0.10]
    return np.select(conditions, choices, default=0.15)


def get_break_even_rate(pd, lgd):
    """Calculates the interest rate where the bank's Expected Profit is zero."""
    return (lgd * pd) / (1 - pd)
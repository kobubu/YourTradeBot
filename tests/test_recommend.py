import pandas as pd
import numpy as np
from core.recommend import generate_recommendations
from datetime import datetime

def test_recommendations_and_profit():
    idx = pd.bdate_range(start='2024-01-01', periods=30)
    # make a simple wave with clear mins/maxs
    y = 100 + np.sin(np.linspace(0, 6.0, 30))*5
    df = pd.DataFrame({'forecast': y}, index=idx)
    summary, profit = generate_recommendations(df, 1000.0)
    assert isinstance(summary, str)
    assert isinstance(profit, float)

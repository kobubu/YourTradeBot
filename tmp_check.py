from core.data import load_ticker_history
for t in ("AAPL","META"):
    df = load_ticker_history(t)
    if df is None:
        print(f"{t} -> None")
    else:
        print(f"{t} -> {len(df)} rows, last index={df.index[-1]}")

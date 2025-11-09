import io
import os
import json
import hashlib
import time
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from dataclasses import asdict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from .models import select_and_fit, refit_and_forecast_30d
from . import model_cache

MODEL_VERSION = "v1"
WF_HORIZON = int(os.getenv("WF_HORIZON", str(5)))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(0)))

def _make_data_signature(df: pd.DataFrame) -> str:
    """Создает сигнатуру данных для проверки изменений"""
    last_ts = str(df.index[-1])
    tail = df['Close'].tail(10).tolist()
    payload = json.dumps({"last_ts": last_ts, "tail": tail}, default=str, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def train_select_and_forecast(df: pd.DataFrame,
                              ticker: Optional[str] = None,
                              force_retrain: bool = False) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Обучает/выбирает лучшую модель и строит 30-дневный прогноз.
    
    Пытается использовать кэш если доступен, иначе обучает с нуля.
    Возвращает: (best_dict, metrics, fcst_df)
    """
    y = df['Close'].copy()
    val_steps = min(30, max(10, len(y)//10))

    if ticker and not force_retrain:
        data_sig = _make_data_signature(df)
        params = {"val_steps": val_steps, "disable_lstm": os.getenv('DISABLE_LSTM','0')=='1'}
        cache_key = model_cache.make_cache_key(ticker, "auto", params, data_sig)

        skl_model, skl_meta = model_cache.load_sklearn_model(cache_key)
        if skl_model is not None:
            try:
                fresh = (skl_meta.get('model_version') == MODEL_VERSION and
                         (time.time() - int(skl_meta.get('trained_at', 0)) <= CACHE_TTL_SECONDS))
                if not fresh:
                    model_cache.remove_key(cache_key)
                else:
                    lag = int(skl_meta.get('extra', {}).get('lag', 30))
                    arr = y.values.astype(float)
                    if len(arr) >= lag + 1:
                        last_window = arr[-lag:].copy()
                        preds = []
                        for _ in range(30):
                            yhat = skl_model.predict(last_window.reshape(1, -1))[0]
                            preds.append(float(yhat))
                            last_window = np.roll(last_window, -1)
                            last_window[-1] = yhat
                        future_idx = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
                        fcst_df = pd.DataFrame({'forecast': np.array(preds)}, index=future_idx)
                        best_dict = {"name": skl_meta.get('name', 'cached_sklearn')}
                        metrics = skl_meta.get('metrics', {"rmse": None})
                        print(f"DEBUG: cache HIT (sklearn) for {ticker}, model={best_dict['name']}")
                        return best_dict, metrics, fcst_df
                    else:
                        model_cache.remove_key(cache_key)
            except Exception:
                try: model_cache.remove_key(cache_key)
                except Exception: pass

        sm_res, sm_meta = model_cache.load_statsmodels_result(cache_key)
        if sm_res is not None:
            try:
                fresh = (sm_meta.get('model_version') == MODEL_VERSION and
                         (time.time() - int(sm_meta.get('trained_at', 0)) <= CACHE_TTL_SECONDS))
                if not fresh:
                    model_cache.remove_key(cache_key)
                else:
                    fcst = sm_res.get_forecast(steps=30).predicted_mean.values
                    future_idx = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
                    fcst_df = pd.DataFrame({'forecast': fcst}, index=future_idx)
                    best_dict = {"name": sm_meta.get('name', 'cached_sarimax')}
                    metrics = sm_meta.get('metrics', {"rmse": None})
                    print(f"DEBUG: cache HIT (sarimax) for {ticker}, model={best_dict['name']}")
                    return best_dict, metrics, fcst_df
            except Exception:
                try: model_cache.remove_key(cache_key)
                except Exception: pass

        tf_model, tf_meta = model_cache.load_tf_model(cache_key)
        if tf_model is not None:
            try:
                fresh = (tf_meta.get('model_version') == MODEL_VERSION and
                         (time.time() - int(tf_meta.get('trained_at', 0)) <= CACHE_TTL_SECONDS))
                if not fresh:
                    model_cache.remove_key(cache_key)
                else:
                    mu = float(tf_meta['extra']['mu'])
                    sigma = float(tf_meta['extra']['sigma'])
                    window = int(tf_meta['extra']['window'])
                    arr = y.values.astype('float32').reshape(-1, 1)
                    norm = (arr - mu) / sigma
                    last_seq = norm[-window:, 0].reshape(1, window, 1)
                    preds = []
                    for _ in range(30):
                        yhat = tf_model.predict(last_seq, verbose=0).reshape(-1)[0]
                        preds.append(float(yhat * sigma + mu))
                        last_seq = np.concatenate([last_seq[0, 1:, 0], [yhat]]).reshape(1, window, 1)
                    future_idx = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
                    fcst_df = pd.DataFrame({'forecast': np.array(preds)}, index=future_idx)
                    best_dict = {"name": tf_meta.get('name', 'cached_lstm')}
                    metrics = tf_meta.get('metrics', {"rmse": None})
                    print(f"DEBUG: cache HIT (lstm) for {ticker}, model={best_dict['name']}")
                    return best_dict, metrics, fcst_df
            except Exception:
                try: model_cache.remove_key(cache_key)
                except Exception: pass

        print(f"DEBUG: cache MISS for {ticker}, training from scratch")
    else:
        if force_retrain:
            print(f"DEBUG: force_retrain=True → skip cache for {ticker or 'N/A'}")

    y = df['Close'].copy()
    best = select_and_fit(
        y, 
        val_steps=min(30, max(10, len(y)//10)),
        horizon=WF_HORIZON,
        eval_tag=ticker,
        save_plots=True
    )

    y_fcst_30 = refit_and_forecast_30d(y, best)
    last_date = df.index[-1]
    future_idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
    fcst_df = pd.DataFrame({'forecast': y_fcst_30.values}, index=future_idx)

    rmse = mean_squared_error(y.iloc[-val_steps:], best.yhat_val[-val_steps:], squared=False)
    try:
        mape = mean_absolute_percentage_error(y.iloc[-val_steps:], best.yhat_val[-val_steps:])
    except Exception:
        mape = np.nan

    best_dict = {"name": best.name}
    metrics = {"rmse": float(rmse), "mape": float(mape) if mape == mape else None}

    try:
        if ticker:
            data_sig = _make_data_signature(df)
            params = {"val_steps": val_steps, "disable_lstm": os.getenv('DISABLE_LSTM','0')=='1'}
            cache_key = model_cache.make_cache_key(ticker, "auto", params, data_sig)
            meta = {
                "name": best.name, "trained_at": int(time.time()),
                "metrics": metrics, "extra": getattr(best, 'extra', {}),
                "model_version": MODEL_VERSION, "data_sig": data_sig
            }

            if best.extra.get('type') == 'ridge':
                model_cache.save_sklearn_model(cache_key, best.model_obj, meta)
            elif best.extra.get('type') == 'sarimax':
                import statsmodels.api as sm
                order, _ = best.model_obj
                model = sm.tsa.SARIMAX(y, order=order,
                                       enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                meta['extra'] = {"order": order}
                model_cache.save_statsmodels_result(cache_key, res, meta)
            elif best.extra.get('type') == 'lstm':
                model, (mu, sigma), window = best.model_obj
                meta['extra'] = {"mu": float(mu), "sigma": float(sigma), "window": int(window)}
                model_cache.save_tf_model(cache_key, model, meta)
    except Exception:
        pass

    print(f"DEBUG: trained from scratch. winner={best_dict['name']}, rmse={metrics['rmse']:.4f}")
    return best_dict, metrics, fcst_df


def make_plot_image(history_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str, markers: list = None) -> io.BytesIO:
    """Создает изображение графика в памяти и возвращает байтовый буфер"""
    plt.figure(figsize=(10,5))
    plt.plot(history_df.index, history_df['Close'], label='History')
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast')
    plt.title(f"{ticker}: History & 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def export_plot_pdf(history_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str, out_path: str) -> None:
    """Экспортирует график в PDF файл"""
    plt.figure(figsize=(10,5))
    plt.plot(history_df.index, history_df['Close'], label='History')
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast')
    plt.title(f"{ticker}: History & 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
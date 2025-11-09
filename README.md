# Telegram Stock Forecast Bot (Time Series Project)

A Telegram bot that:
- Loads 2 years of daily stock prices from Yahoo Finance.
- Trains **three** model families on the close price:
  1) **Ridge Regression** with lag features (classic ML)
  2) **SARIMAX** (statistical)
  3) **LSTM** (neural network)
- Selects the best model by RMSE on a rolling validation window.
- Forecasts the next **30 trading days**.
- Plots history + forecast.
- Generates buy/sell day hints (local minima/maxima) and an **estimated profit** for a notional capital.
- Writes a CSV log per user request.

> ⚠️ Educational project. Not financial advice.

## Quick start

1. **Python 3.10+** recommended.
2. Create a virtual env and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set your Telegram bot token:
   - Copy `.env.example` to `.env` and put your token there.

4. Run the bot:
   ```bash
   .\.venv\Scripts\python.exe bot.py
   ```

## Usage in Telegram

- `/start` — help.
- `/forecast <TICKER> <AMOUNT>` — e.g. `/forecast AAPL 1000`

Bot will reply with:
- Plot image with history + forecast.
- Delta vs. today, chosen model & RMSE.
- Text summary of buy/sell suggestions.
- Estimated profit assuming you trade forecasted local minima/maxima using the full amount per leg (fractional shares allowed).

## Files

- `core/data.py` — data loading & validation.
- `core/models.py` — models, training, selection.
- `core/forecast.py` — backtesting, evaluation, 30D forecast, plotting.
- `core/recommend.py` — local extrema & profit calc.
- `core/logging_utils.py` — structured CSV logging.
- `logs/` — logs CSV.
- `artifacts/` — generated plots.

## Notes

- LSTM can be slow on first run (TensorFlow). You can set `USE_LSTM=False` in `core/models.py` if needed.
- Trading strategy here is heuristic and **for study only**.

## Docker

Build and run with Docker:

```bash
docker compose up --build -d
```

Put your token into `.env`. Optional toggles:
- `SAVE_CSV=1` — сохранять загруженную историю в `artifacts/`.
- `DISABLE_LSTM=1` — не использовать LSTM при выборе моделей (ускоряет холодный старт).

## Примеры для отчёта

В папке `docs/` лежит файл **ExampleDialog.docx** с шаблоном скриншотов/диалога (вставьте скриншоты из Telegram).


## Тесты
Запуск unit-тестов (локально):
```bash
pytest -q
```

## Логирование в Google Sheets (опционально)
1. Создайте Service Account в Google Cloud и скачайте JSON ключ.
2. Поделитесь нужной таблицей Google (Drive) с email сервис-аккаунта (роль Editor).
3. В `.env` укажите:
```
GSHEETS_ENABLED=1
GSHEETS_CRED_JSON=/app/service_account.json  # или локальный путь
GSHEETS_SPREADSHEET_ID=<ID_таблицы>
GSHEETS_WORKSHEET=logs
```
4. Если используете Docker, добавьте volume с JSON или COPY в Dockerfile.

При ошибках записи в Sheets бот продолжит работу, запись уйдёт только в `logs/logs.csv`.


## Качество кода и CI
- **Ruff** — линтер/форматирование (`ruff check .`)
- **Mypy** — статическая типизация (`mypy .`)
- **Pydocstyle** — стиль docstring
- **GitHub Actions** — запуск тестов и линтеров на каждом пуше/PR (см. `.github/workflows/ci.yml`).

## Кэширование данных
Чтобы не дергать Yahoo Finance при каждом запросе, включено кэширование CSV в `artifacts/`:
- `CACHE_DAYS=1` (по умолчанию) — использовать последние загруженные котировки, если им меньше суток.
- `SAVE_CSV=1` — дополнительно хранить исторические выгрузки `history_*.csv` для отчётов.

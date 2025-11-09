import os

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes

from core.data import load_ticker_history
from core.forecast import export_plot_pdf, make_plot_image, train_select_and_forecast
from core.logging_utils import log_request
from core.recommend import generate_recommendations

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

DEFAULT_AMOUNT = 1000.0

CAPTION_MAX = 1024
TEXT_MAX = 4096

SUPPORTED_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "JPM",
    "BAC",
    "NFLX",
    "DIS",
]


async def _run_forecast_for(ticker: str, amount: float, reply_text_fn, reply_photo_fn, user_id=None):
    """Shared forecast workflow used by command and callback handlers"""
    try:
        await reply_text_fn(f"Загружаю данные для {ticker} и считаю прогноз…")
        df = load_ticker_history(ticker)
        if df is None or df.empty:
            await reply_text_fn("Не удалось загрузить данные. Проверьте тикер.")
            return

        best, metrics, fcst_df = train_select_and_forecast(df, ticker=ticker)
        rec_summary, profit_est, markers = generate_recommendations(fcst_df, amount, model_rmse=metrics.get('rmse') if metrics else None)
        img_buf = make_plot_image(df, fcst_df, ticker, markers=markers)

        try:
            from datetime import datetime
            art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(art_dir, exist_ok=True)
            pdf_path = os.path.join(art_dir, f"{ticker}_forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
            export_plot_pdf(df, fcst_df, ticker, pdf_path)
        except Exception:
            pass

        delta_pct = ((fcst_df['forecast'].iloc[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100.0
        msg = (
            f"Тикер: {ticker}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены к последнему дню: {delta_pct:+.2f}%\n\n"
            f"{rec_summary}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_est:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )

        if len(msg) <= CAPTION_MAX:
            await reply_photo_fn(photo=img_buf, caption=msg)
        else:
            await reply_photo_fn(photo=img_buf)
            for i in range(0, len(msg), TEXT_MAX):
                await reply_text_fn(msg[i:i+TEXT_MAX])

        log_request(
            user_id=user_id,
            ticker=ticker,
            amount=amount,
            best_model=best['name'],
            metric_name='RMSE',
            metric_value=metrics['rmse'],
            est_profit=profit_est,
        )
    except Exception as e:
        await reply_text_fn(f"Ошибка: {e}")

HELP_TEXT = (
    "Привет! Я бот прогноза акций.\n\n"
    "Команды:\n"
    "/forecast <TICKER> — пример: /forecast AAPL\nЧтобы увидеть быстрые кнопки с популярными тикерами, используйте /tickers\n"
    "Я загружу котировки за 2 года, обучу несколько моделей и пришлю прогноз на 30 дней,\n"
    "рекомендации по дням покупки/продажи и оценку условной прибыли.\n\n"
    "⚠️ Учебный проект, не является инвестсоветом."
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(HELP_TEXT)

async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /forecast <TICKER>"""
    try:
        if len(context.args) < 1:
            await update.message.reply_text("Использование: /forecast <TICKER>\nНапример: /forecast AAPL")
            return
        try:
            print("DEBUG: received message_text=", update.message.text if update.message else None)
            print("DEBUG: context.args=", context.args)
        except Exception:
            pass

        ticker = context.args[0].upper().strip()
        amount = DEFAULT_AMOUNT

        await update.message.reply_text(f"Загружаю данные для {ticker} и считаю прогноз…")

        df = load_ticker_history(ticker)
        if df is None or df.empty:
            await update.message.reply_text("Не удалось загрузить данные. Проверьте тикер.")
            return

        force_retrain = False
        if len(context.args) >= 2 and context.args[1].lower() in ("retrain", "force", "fresh"):
            force_retrain = True

        best, metrics, fcst_df = train_select_and_forecast(df, ticker=ticker, force_retrain=force_retrain)
        rec_summary, profit_est, markers = generate_recommendations(fcst_df, amount, model_rmse=metrics.get('rmse') if metrics else None)
        img_buf = make_plot_image(df, fcst_df, ticker, markers=markers)

        try:
            from datetime import datetime
            art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(art_dir, exist_ok=True)
            pdf_path = os.path.join(art_dir, f"{ticker}_forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
            export_plot_pdf(df, fcst_df, ticker, pdf_path)
        except Exception:
            pass

        delta_pct = ((fcst_df['forecast'].iloc[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100.0
        msg = (
            f"Тикер: {ticker}\n"
            f"Лучшая модель: {best['name']} (RMSE={metrics['rmse']:.2f})\n"
            f"Изменение цены к последнему дню: {delta_pct:+.2f}%\n\n"
            f"{rec_summary}\n\n"
            f"Ориентировочная прибыль при капитале {amount:.2f} USD: {profit_est:.2f} USD\n"
            "⚠️ Не является инвестсоветом."
        )

        if len(msg) <= CAPTION_MAX:
            await update.message.reply_photo(photo=img_buf, caption=msg)
        else:
            await update.message.reply_photo(photo=img_buf)
            for i in range(0, len(msg), TEXT_MAX):
                await update.message.reply_text(msg[i:i+TEXT_MAX])

        user_id = update.effective_user.id if update.effective_user else None
        log_request(
            user_id=user_id,
            ticker=ticker,
            amount=amount,
            best_model=best['name'],
            metric_name='RMSE',
            metric_value=metrics['rmse'],
            est_profit=profit_est,
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /tickers - показывает кнопки с популярными тикерами"""
    buttons, row = [], []
    for t in SUPPORTED_TICKERS:
        row.append(InlineKeyboardButton(t, callback_data=f"forecast:{t}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    await update.message.reply_text(
        "Выберите тикер (нажмите кнопку):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )

async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на инлайн-кнопки"""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = (query.data or "").strip()

    if data.startswith("forecast:"):
        ticker = data.split(":", 1)[1].strip().upper()
        if SUPPORTED_TICKERS and ticker not in SUPPORTED_TICKERS:
            await query.message.reply_text(f"Тикер {ticker} не в списке доступных. Нажмите /tickers.")
            return

        amount = DEFAULT_AMOUNT

        async def reply_text(text):
            await query.message.reply_text(text)

        async def reply_photo(photo, caption=None):
            await query.message.reply_photo(photo=photo, caption=caption)

        user_id = query.from_user.id if query.from_user else None
        await _run_forecast_for(ticker=ticker, amount=amount,
                                reply_text_fn=reply_text, reply_photo_fn=reply_photo,
                                user_id=user_id)



def main():
    """Основная функция запуска бота"""
    if not BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("tickers", tickers))
    app.add_handler(CallbackQueryHandler(_on_callback))
    print("Bot is running…")
    app.run_polling()

if __name__ == '__main__':
    main()
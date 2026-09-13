AI STOCK MARKET PREDICTION PROJECT

Project features:
1. Login page
2. Stock dashboard
3. Four stocks: Reliance, TCS, HDFC Bank, Infosys
4. Live/historical data using yfinance
5. LSTM AI prediction
6. Current price vs predicted price chart
# The project uses PyTorch for LSTM prediction and supports Python 3.14.

Demo login:
Username: admin
Password: 1234

RUN:
python app.py

NOTE:
AI-generated forecasts and market signals are based on historical price and technical-indicator patterns.They involve uncertainty
& don't guarantee future performance."

final architecture:
python app.py
        ↓
Streamlit starts automatically
        ↓
Yahoo Finance / yfinance
        ↓
OHLCV Market Data
        ↓
Data Cleaning
        ↓
13-Feature Engineering(MinMax Scaling,60-Day Sequences,80% Train / 20% Test,Multivariate PyTorch LSTM,LSTM 64 → LSTM 32,Price Forecast,MAE / RMSE / MAPE / R²)
        ↓
Technical Analysis(RSI + MACD + SMA + EMA + Volatility)
        ↓
AI Decision Engine
        ↓
BUY / HOLD / SELL + Confidence Score + Explanation
        ↓
Interactive Streamlit Dashboard
        ↓
🏆 AI Ranking
        ↓
📅 Multi-Day Forecast
        ↓
📄 Prediction Report
        ↓
ctrl + c (to end)
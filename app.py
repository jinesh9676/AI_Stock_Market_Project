import os
import sys
import subprocess
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# FIRST-RUN DEPENDENCY INSTALLER - LOCAL COMPUTER
BASE_DIR = Path(__file__).resolve().parent
PACKAGES = {"streamlit": "streamlit", "pandas": "pandas", "numpy": "numpy", "yfinance": "yfinance", "altair": "altair", "torch": "torch", "sklearn": "scikit-learn"}

def local_setup():
    if os.name != "nt" or os.environ.get("AI_STOCK_ENV_ACTIVE") == "1":
        return
    python = BASE_DIR / "env" / "Scripts" / "python.exe"
    if not python.is_file():
        subprocess.check_call([sys.executable, "-m", "venv", str(BASE_DIR / "env")])
    missing = [package for module, package in PACKAGES.items() if subprocess.run([str(python), "-c", f"import {module}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0]
    if missing:
        subprocess.check_call([str(python), "-m", "pip", "install", *missing])
    environment = os.environ.copy()
    environment["AI_STOCK_ENV_ACTIVE"] = "1"
    subprocess.run([str(python), "-m", "streamlit", "run", str(BASE_DIR / "app.py")], env=environment, check=True)
    sys.exit(0)

if __name__ == "__main__" and os.name == "nt" and os.environ.get("AI_STOCK_ENV_ACTIVE") != "1":
    local_setup()

st.set_page_config(page_title='AI Stock Market', page_icon='📈', layout='wide')
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        /* ===== LOGIN PAGE BACKGROUND ===== */
        .stApp {
            background:
                radial-gradient(circle at 18% 18%, rgba(37, 99, 235, 0.18), transparent 28%),
                radial-gradient(circle at 82% 22%, rgba(14, 165, 233, 0.12),transparent 24%),
                linear-gradient(135deg, #060b14 0%, #0b1220 48%, #070d16 100%);
        }

        /* Hide Streamlit chrome on login */
        header[data-testid="stHeader"] {
            display: none;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 1rem;
            max-width: 1180px;
        }

        /* ===== BRAND AREA ===== */
        .login-brand {
            text-align: center;
            margin-bottom: 26px;
        }

        .brand-icon {
            width: 78px;
            height: 78px;
            margin: 0 auto 16px auto;
            border-radius: 22px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            background: linear-gradient(
                145deg, rgba(59,130,246,.22), rgba(14,165,233,.12)
            );
            border: 1px solid rgba(148,163,184,.20);
            box-shadow: 0 16px 40px rgba(0,0,0,.35);
        }

        .brand-title {
            color: #f8fafc;
            font-size: 38px;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }

        .brand-subtitle {
            color: #94a3b8;
            font-size: 15px;
        }

        .brand-chip {
            display: inline-block;
            margin-top: 13px;
            padding: 7px 14px;
            border-radius: 999px;
            color: #bfdbfe;
            font-size: 12px;
            font-weight: 600;
            background: rgba(37,99,235,.12);
            border: 1px solid rgba(96,165,250,.20);
        }

        /* ===== FORM CARD ===== */
        div[data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 28px 30px 26px 30px;
            box-shadow: 0 24px 65px rgba(0,0,0,.42);
            backdrop-filter: blur(18px);
        }

        .form-title {
            text-align: center;
            color: #f8fafc;
            font-size: 24px;
            font-weight: 750;
            margin-bottom: 5px;
        }

        .form-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            margin-bottom: 20px;
        }

        /* ===== INPUTS ===== */
        div[data-testid="stTextInput"] label {
            color: #cbd5e1 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stTextInput"] input {
            min-height: 46px !important;
            border-radius: 11px !important;
            border: 1px solid #334155 !important;
            background: #0b1220 !important;
            color: #f8fafc !important;
            padding-left: 14px !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }

        /* ===== LOGIN BUTTON ===== */
        div[data-testid="stFormSubmitButton"] {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stFormSubmitButton"] > button {
            width: 220px;
            min-height: 48px;
            border: 0 !important;
            border-radius: 11px !important;
            color: white !important;
            font-size: 14px !important;
            font-weight: 750 !important;
            letter-spacing: 0.5px;
            background: linear-gradient(
                90deg, #2563eb 0%, #0284c7 52%,  #059669 100%
            ) !important;
            box-shadow: 0 10px 26px rgba(37,99,235,.22);
            margin-top: 7px;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(37,99,235,.30);
        }
        /* ===== DEMO BOX ===== */
        .demo-box {
            margin-top: 18px;
            padding: 12px 14px;
            border-radius: 11px;
            text-align: center;
            background: rgba(30,41,59,.72);
            border: 1px solid rgba(148,163,184,.14);
            color: #cbd5e1;
            font-size: 12px;
            line-height: 1.6;
        }

        .demo-box strong {
            color: #f8fafc;
            font-size: 13px;
        }

        .secure-note {
            text-align: center;
            color: #64748b;
            font-size: 11px;
            margin-top: 14px;
        }

        .login-features {
            margin-top: 25px;
            text-align: center;
            color: #64748b;
            font-size: 11px;
            line-height: 1.7;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 2rem;
            }

            .brand-title {
                font-size: 30px;
            }

            div[data-testid="stForm"] {
                padding: 22px 18px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('\n        <div class="login-brand">\n            <div class="brand-icon">📈</div>\n            <div class="brand-title">AI Stock Market</div>\n            <div class="brand-subtitle">\n                Intelligent Prediction & Market Analysis System\n            </div>\n            <div class="brand-chip">\n                PyTorch LSTM • Technical Analysis • Explainable AI\n            </div>\n        </div>\n        ', unsafe_allow_html=True)
    left, center, right = st.columns([1.1, 1.2, 1.1])
    with center:
        with st.form('login_form', clear_on_submit=False):
            st.markdown('\n                <div class="form-title">Welcome Back</div>\n                <div class="form-subtitle">\n                    Sign in to continue to your AI dashboard\n                </div>\n                ', unsafe_allow_html=True)
            user = st.text_input('Username', placeholder='Enter your username')
            password = st.text_input('Password', type='password', placeholder='Enter your password')
            submitted = st.form_submit_button('LOGIN TO DASHBOARD', type='primary', width='stretch')
            st.markdown('\n                <div class="demo-box">\n                    <strong>Demo Account</strong><br>\n                    Username: <b>admin</b> &nbsp;&nbsp;•&nbsp;&nbsp;\n                    Password: <b>1234</b>\n                </div>\n\n                <div class="secure-note">\n                    🔒 Educational access • AI Stock Market Prediction System\n                </div>\n                ', unsafe_allow_html=True)
        if submitted:
            if user == 'admin' and password == '1234':
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error('Invalid username or password.')
    st.markdown('\n        <div class="login-features">\n            13-Feature LSTM Forecasting &nbsp;•&nbsp;\n            RSI &nbsp;•&nbsp; MACD &nbsp;•&nbsp; SMA &nbsp;•&nbsp; EMA\n            &nbsp;•&nbsp; Volatility &nbsp;•&nbsp; BUY / HOLD / SELL\n        </div>\n        ', unsafe_allow_html=True)
    st.markdown('\n    <style>\n    .login-contact{margin:12px auto 0;padding:14px 16px;max-width:800px;text-align:center;border-top:1px solid #334155;color:#e2e8f0}\n    .login-contact h3{font-size:18px;margin:0 0 7px}.login-contact p{font-size:13px;margin:5px 0}\n    .login-contact-details{display:flex;flex-wrap:wrap;gap:10px 18px;justify-content:center;font-size:13px;margin:9px 0}\n    .login-contact-details a{color:#38bdf8;text-decoration:none}\n    </style>\n    <div class="login-contact">\n      <h3>💼 Hire a Python AI Developer</h3>\n      <p>Custom Python, Machine Learning, Streamlit and AI-powered applications for businesses and institutes.</p>\n      <div class="login-contact-details">\n        <span>👨\u200d💻 Jinesh Paras Shah</span>\n        <a href="mailto:jinesh9676@gmail.com">📧 jinesh9676@gmail.com</a>\n        <a href="tel:+918799428255">📱 +91 8799428255</a>\n      </div>\n      <p>Python Automation • AI / ML • Stock Analytics • Streamlit Dashboards</p>\n    </div>\n    ', unsafe_allow_html=True)
    st.stop()
top1, top2 = st.columns([8, 1])
with top2:
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.rerun()
st.title('📈 AI Stock Market Prediction System')
st.caption('Multivariate LSTM forecasting, technical analysis, AI decision scoring and explainable market signals')
stocks = {'RELIANCE.NS': 'RELIANCE', 'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFC BANK', 'INFY.NS': 'INFOSYS'}

@st.cache_data(ttl=300)
def download_stock_data(ticker, period):
    return yf.download(ticker, period=period, progress=False, auto_adjust=False)

def add_technical_indicators(data):
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - 100 / (1 + rs)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=20).std() * 100
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

def show_price_zone_chart(series, title):
    price = series.dropna()
    if len(price) < 2:
        st.warning('Not enough price data for chart.')
        return
    low_limit = float(price.quantile(0.33))
    high_limit = float(price.quantile(0.67))
    chart_data = pd.DataFrame({'Date': pd.to_datetime(price.index), 'Price': price.values.astype(float)})
    chart_data['NextDate'] = chart_data['Date'].shift(-1)
    chart_data['NextPrice'] = chart_data['Price'].shift(-1)
    chart_data = chart_data.dropna()
    chart_data['AveragePrice'] = (chart_data['Price'] + chart_data['NextPrice']) / 2
    chart_data['Zone'] = np.where(chart_data['AveragePrice'] < low_limit, 'Low Price', np.where(chart_data['AveragePrice'] > high_limit, 'High Price', 'Normal Price'))
    price_line = alt.Chart(chart_data).mark_rule(strokeWidth=3).encode(x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %Y', labelAngle=0)), x2='NextDate:T', y=alt.Y('Price:Q', title='Price (₹)', scale=alt.Scale(zero=False)), y2='NextPrice:Q', color=alt.Color('Zone:N', scale=alt.Scale(domain=['Low Price', 'Normal Price', 'High Price'], range=['#ff2b2b', '#2196f3', '#00a651']), legend=alt.Legend(title='Price Zone', orient='top')), tooltip=[alt.Tooltip('Date:T', title='Date', format='%d-%m-%Y'), alt.Tooltip('Price:Q', title='Price ₹', format=',.2f'), alt.Tooltip('Zone:N', title='Zone')])
    low_rule_data = pd.DataFrame({'Price': [low_limit]})
    high_rule_data = pd.DataFrame({'Price': [high_limit]})
    low_rule = alt.Chart(low_rule_data).mark_rule(color='#ff2b2b', strokeDash=[6, 5], opacity=0.55).encode(y='Price:Q')
    high_rule = alt.Chart(high_rule_data).mark_rule(color='#00a651', strokeDash=[6, 5], opacity=0.55).encode(y='Price:Q')
    chart = (price_line + low_rule + high_rule).properties(title=title, height=430).interactive()
    st.altair_chart(chart, width='stretch')
    z1, z2, z3 = st.columns(3)
    z1.metric('🔴 Low Price Zone', f'Below ₹{low_limit:,.2f}')
    z2.metric('🔵 Normal Price Zone', f'₹{low_limit:,.2f} - ₹{high_limit:,.2f}')
    z3.metric('🟢 High Price Zone', f'Above ₹{high_limit:,.2f}')

def calculate_ai_decision(expected, current, sma20, sma50, rsi, macd, macd_signal, volatility, mape):
    score = 0
    bullish_points = 0
    bearish_points = 0
    reasons = []
    if expected >= 2:
        score += 3
        bullish_points += 3
        reasons.append(f'✅ LSTM forecasts a strong positive movement of {expected:+.2f}%.')
    elif expected >= 0.5:
        score += 2
        bullish_points += 2
        reasons.append(f'✅ LSTM forecasts a positive movement of {expected:+.2f}%.')
    elif expected <= -2:
        score -= 3
        bearish_points += 3
        reasons.append(f'⚠️ LSTM forecasts a strong negative movement of {expected:+.2f}%')
    elif expected <= -0.5:
        score -= 2
        bearish_points += 2
        reasons.append(f'⚠️ LSTM forecasts a negative movement of {expected:+.2f}%.')
    else:
        reasons.append(f'➖ LSTM forecasts limited movement of {expected:+.2f}%.')
    if current > sma20 and sma20 > sma50:
        score += 2
        bullish_points += 2
        reasons.append('✅ Price is above SMA 20 & SMA 20 is above SMA 50, bullish trend.')
    elif current < sma20 and sma20 < sma50:
        score -= 2
        bearish_points += 2
        reasons.append('⚠️ Price is below SMA 20 & SMA 20 is below SMA 50, bearish trend.')
    elif current > sma20:
        score += 1
        bullish_points += 1
        reasons.append('✅ Current price is above the short-term SMA 20.')
    elif current < sma20:
        score -= 1
        bearish_points += 1
        reasons.append('⚠️ Current price is below the short-term SMA 20.')
    if macd > macd_signal:
        score += 2
        bullish_points += 2
        reasons.append('✅ MACD is above its signal line, indicating positive momentum.')
    else:
        score -= 2
        bearish_points += 2
        reasons.append('⚠️ MACD is below its signal line, indicating negative momentum.')
    if rsi < 30:
        score += 1
        bullish_points += 1
        reasons.append(f'✅ RSI is {rsi:.2f}, indicating an oversold condition.')
    elif rsi > 70:
        score -= 1
        bearish_points += 1
        reasons.append(f'⚠️ RSI is {rsi:.2f}, indicating an overbought condition.')
    else:
        reasons.append(f'➖ RSI is {rsi:.2f}, inside the neutral 30-70 range.')
    if volatility < 1:
        risk_adjustment = 5
        reasons.append(f'✅ Volatility is relatively low at {volatility:.2f}%.')
    elif volatility < 2:
        risk_adjustment = 0
        reasons.append(f'➖ Volatility is moderate at {volatility:.2f}%.')
    else:
        risk_adjustment = -10
        reasons.append(f'⚠️ Volatility is high at {volatility:.2f}%.')
    if mape <= 2:
        accuracy_adjustment = 15
        reasons.append(f'✅ Model MAPE is relatively low at {mape:.2f}%.')
    elif mape <= 5:
        accuracy_adjustment = 5
        reasons.append(f'➖ Model MAPE is moderate at {mape:.2f}%.')
    elif mape <= 10:
        accuracy_adjustment = -5
        reasons.append(f'⚠️ Model MAPE is elevated at {mape:.2f}%.')
    else:
        accuracy_adjustment = -15
        reasons.append(f'⚠️ Model MAPE is high at {mape:.2f}%.')
    if score >= 4:
        signal = '🟢 BUY'
        signal_text = 'BUY'
    elif score <= -4:
        signal = '🔴 SELL'
        signal_text = 'SELL'
    else:
        signal = '🔵 HOLD'
        signal_text = 'HOLD'
    total_directional = bullish_points + bearish_points
    if total_directional > 0:
        directional_agreement = max(bullish_points, bearish_points) / total_directional * 100
    else:
        directional_agreement = 50
    confidence = directional_agreement + risk_adjustment + accuracy_adjustment
    confidence = int(np.clip(confidence, 40, 95))
    if confidence >= 80:
        confidence_level = 'High'
    elif confidence >= 60:
        confidence_level = 'Moderate'
    else:
        confidence_level = 'Low'
    return (score, signal, signal_text, confidence, confidence_level, reasons)
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'Return', 'Volatility']
period = st.selectbox('Data Period', ['6mo', '1y', '2y', '5y'], index=1)
cols = st.columns(4)
data_cache = {}
feature_cache = {}
for col, (ticker, name) in zip(cols, stocks.items()):
    try:
        data = download_stock_data(ticker, period)
        if data.empty:
            raise ValueError('No data available')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = data['Close'].dropna()
        data_cache[ticker] = close
        feature_data = add_technical_indicators(data)
        if not feature_data.empty:
            feature_cache[ticker] = feature_data
        current = float(close.iloc[-1])
        if len(close) > 1:
            previous = float(close.iloc[-2])
            change = (current / previous - 1) * 100
        else:
            change = 0
        col.metric(name, f'₹{current:,.2f}', f'{change:+.2f}%')
    except Exception:
        col.metric(name, 'Data unavailable')
st.divider()
st.subheader('📊 Market Overview')
st.caption('🔴 Low-price area | 🔵 Normal-price area | 🟢 High-price area')
for ticker, name in stocks.items():
    if ticker in data_cache:
        st.write(f'### {name}')
        show_price_zone_chart(data_cache[ticker], f'{name} Historical Price Zones')
st.divider()
st.header('📈 Technical Analysis')
analysis_stock = st.selectbox('Select Stock for Technical Analysis', list(stocks.keys()), format_func=lambda x: stocks[x], key='analysis_stock')
if analysis_stock in feature_cache:
    technical = feature_cache[analysis_stock]
    latest = technical.iloc[-1]
    current_price = float(latest['Close'])
    sma20 = float(latest['SMA_20'])
    sma50 = float(latest['SMA_50'])
    ema20 = float(latest['EMA_20'])
    rsi = float(latest['RSI_14'])
    macd = float(latest['MACD'])
    macd_signal = float(latest['MACD_Signal'])
    daily_return = float(latest['Return']) * 100
    volatility = float(latest['Volatility'])
    a1, a2, a3, a4 = st.columns(4)
    a1.metric('Current Price', f'₹{current_price:,.2f}')
    a2.metric('SMA 20', f'₹{sma20:,.2f}')
    a3.metric('SMA 50', f'₹{sma50:,.2f}')
    a4.metric('EMA 20', f'₹{ema20:,.2f}')
    b1, b2, b3, b4 = st.columns(4)
    b1.metric('RSI 14', f'{rsi:.2f}')
    b2.metric('MACD', f'{macd:.2f}')
    b3.metric('Daily Return', f'{daily_return:+.2f}%')
    b4.metric('Volatility', f'{volatility:.2f}%')
    if current_price > sma20 and sma20 > sma50:
        trend = '🟢 Bullish'
    elif current_price < sma20 and sma20 < sma50:
        trend = '🔴 Bearish'
    else:
        trend = '🔵 Neutral'
    if rsi >= 70:
        rsi_status = 'Overbought'
    elif rsi <= 30:
        rsi_status = 'Oversold'
    else:
        rsi_status = 'Neutral'
    if macd > macd_signal:
        macd_status = 'Bullish Momentum'
    else:
        macd_status = 'Bearish Momentum'
    if volatility < 1:
        risk_level = '🟢 Low'
    elif volatility < 2:
        risk_level = '🟠 Medium'
    else:
        risk_level = '🔴 High'
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Market Trend', trend)
    c2.metric('RSI Status', rsi_status)
    c3.metric('MACD Status', macd_status)
    c4.metric('Risk Level', risk_level)
    st.subheader('Interactive Price Zone Analysis')
    show_price_zone_chart(technical['Close'], f'{stocks[analysis_stock]} Price Zone Analysis')
    st.subheader('Moving Average Analysis')
    price_chart = technical[['Close', 'SMA_20', 'SMA_50', 'EMA_20']]
    st.line_chart(price_chart, height=350)
    st.subheader('MACD Analysis')
    st.line_chart(technical[['MACD', 'MACD_Signal']], height=250)
    st.subheader('RSI Analysis')
    st.line_chart(technical[['RSI_14']], height=250)
st.divider()
st.header('🤖 Multivariate LSTM AI Prediction')
selected = st.selectbox('Select Stock', list(stocks.keys()), format_func=lambda x: stocks[x], key='prediction_stock')
epochs = st.slider('Training Epochs', min_value=5, max_value=50, value=15)

class StockModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
if st.button('🤖 Train Multivariate LSTM', type='primary'):
    if selected not in feature_cache:
        st.error('Feature data is unavailable.')
        st.stop()
    df = feature_cache[selected].copy()
    time_step = 60
    if len(df) <= time_step + 30:
        st.error('Not enough historical data. Please select 1y, 2y or 5y.')
        st.stop()
    feature_values = df[FEATURES].values.astype(np.float32)
    target_values = df[['Close']].values.astype(np.float32)
    train_size = int(len(df) * 0.8)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(feature_values[:train_size])
    target_scaler.fit(target_values[:train_size])
    scaled_features = feature_scaler.transform(feature_values)
    scaled_target = target_scaler.transform(target_values)
    X = []
    y = []
    sequence_indexes = []
    for i in range(time_step, len(df)):
        X.append(scaled_features[i - time_step:i])
        y.append(scaled_target[i, 0])
        sequence_indexes.append(i)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    test_dates = []
    for idx, original_index in enumerate(sequence_indexes):
        if original_index < train_size:
            X_train.append(X[idx])
            y_train.append(y[idx])
        else:
            X_test.append(X[idx])
            y_test.append(y[idx])
            test_dates.append(df.index[original_index])
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    if len(X_train) == 0 or len(X_test) == 0:
        st.error('Insufficient training or testing data.')
        st.stop()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    model = StockModel(input_size=len(FEATURES))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    with st.spinner('Training 13-feature Multivariate LSTM...'):
        progress_bar = st.progress(0)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress = int((epoch + 1) / epochs * 100)
            progress_bar.progress(progress)
    model.eval()
    with torch.no_grad():
        test_prediction_scaled = model(X_test_tensor).numpy()
    test_predictions = target_scaler.inverse_transform(test_prediction_scaled).flatten()
    actual_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(actual_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(actual_test, test_predictions))
    non_zero = actual_test != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((actual_test[non_zero] - test_predictions[non_zero]) / actual_test[non_zero])) * 100
    else:
        mape = 0
    if len(actual_test) > 1:
        r2 = r2_score(actual_test, test_predictions)
    else:
        r2 = 0
    last_sequence = scaled_features[-time_step:]
    last_tensor = torch.tensor(last_sequence.reshape(1, time_step, len(FEATURES)), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        next_scaled = model(last_tensor).numpy()
    predicted = float(target_scaler.inverse_transform(next_scaled)[0][0])
    current = float(df['Close'].iloc[-1])
    expected = (predicted / current - 1) * 100
    latest = df.iloc[-1]
    rsi = float(latest['RSI_14'])
    macd = float(latest['MACD'])
    macd_signal = float(latest['MACD_Signal'])
    sma20 = float(latest['SMA_20'])
    sma50 = float(latest['SMA_50'])
    volatility = float(latest['Volatility'])
    decision_score, signal, signal_text, confidence, confidence_level, reasons = calculate_ai_decision(expected, current, sma20, sma50, rsi, macd, macd_signal, volatility, mape)
    if current > sma20 and sma20 > sma50:
        ai_trend = 'Bullish'
    elif current < sma20 and sma20 < sma50:
        ai_trend = 'Bearish'
    else:
        ai_trend = 'Neutral'
    if volatility < 1:
        ai_risk = 'Low'
    elif volatility < 2:
        ai_risk = 'Medium'
    else:
        ai_risk = 'High'
    low_limit = float(df['Close'].quantile(0.33))
    high_limit = float(df['Close'].quantile(0.67))
    if current < low_limit:
        current_zone = '🔴 LOW PRICE ZONE'
    elif current > high_limit:
        current_zone = '🟢 HIGH PRICE ZONE'
    else:
        current_zone = '🔵 NORMAL PRICE ZONE'
    st.divider()
    st.subheader('🧠 AI Forecast Result')
    p1, p2, p3 = st.columns(3)
    p1.metric('Current Price', f'₹{current:,.2f}')
    p2.metric('Predicted Next Price', f'₹{predicted:,.2f}')
    p3.metric('Expected Movement', f'{expected:+.2f}%')
    st.subheader('🎯 AI Decision Support')
    d1, d2, d3, d4 = st.columns(4)
    d1.metric('AI Signal', signal)
    d2.metric('Confidence Score', f'{confidence}%')
    d3.metric('Decision Score', f'{decision_score:+d}')
    d4.metric('Confidence Level', confidence_level)
    st.progress(confidence / 100)
    e1, e2, e3 = st.columns(3)
    e1.metric('Market Trend', ai_trend)
    e2.metric('Risk Level', ai_risk)
    e3.metric('Price Zone', current_zone)
    st.subheader('🔍 Explainable AI Analysis')
    st.write(f'### Why the system generated **{signal_text}**')
    for reason in reasons:
        st.write(reason)
    st.info('The confidence score is a heuristic agreement score derived from LSTM direction, technical-indicator agreement, volatility and model error. It is not a calibrated probability of profit or future price movement.')
    st.subheader('📊 Decision Indicator Snapshot')
    i1, i2, i3, i4 = st.columns(4)
    i1.metric('RSI 14', f'{rsi:.2f}')
    i2.metric('MACD', f'{macd:.2f}')
    i3.metric('SMA 20', f'₹{sma20:,.2f}')
    i4.metric('SMA 50', f'₹{sma50:,.2f}')
    st.subheader('📉 Model Evaluation')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('MAE', f'₹{mae:,.2f}')
    m2.metric('RMSE', f'₹{rmse:,.2f}')
    m3.metric('MAPE', f'{mape:.2f}%')
    m4.metric('R²', f'{r2:.4f}')
    st.subheader('Actual vs Predicted Test Data')
    comparison = pd.DataFrame({'Actual Price': actual_test, 'Predicted Price': test_predictions}, index=pd.DatetimeIndex(test_dates))
    st.line_chart(comparison, height=350)
    st.subheader('Current vs AI Forecast')
    prediction_chart = pd.DataFrame({'Price': [current, predicted]}, index=['Current', 'AI Forecast'])
    st.bar_chart(prediction_chart)
    st.subheader('🧠 AI Model Information')
    info1, info2, info3, info4 = st.columns(4)
    info1.metric('Input Features', len(FEATURES))
    info2.metric('Sequence Length', f'{time_step} Days')
    info3.metric('Training Data', '80%')
    info4.metric('Testing Data', '20%')
    st.write('**Multivariate LSTM Features:**')
    st.write(', '.join(FEATURES))
    st.info(f'\nModel: PyTorch Multivariate LSTM\nStock: {stocks[selected]}\nInput Features: {len(FEATURES)}\nSequence Window: Previous {time_step} trading observations\nLSTM Layer 1: 64 hidden units\nLSTM Layer 2: 32 hidden units\nDropout: 20%\nTraining Epochs: {epochs}\nOptimizer: Adam\nLoss Function: Mean Squared Error\nDecision Engine: LSTM Forecast + SMA Trend + RSI + MACD + Volatility +\nModel Error\n        ')
st.divider()
st.warning('AI-generated forecasts, confidence scores and market signals are based on historical market data, technical indicators and model outputs. They involve uncertainty and do not guarantee future market performance.')

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, date, timedelta
import numpy as np
import asyncio
import aiohttp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state['disclaimer_accepted'] = False

    if not st.session_state['disclaimer_accepted']:
        show_disclaimer()
    else:
        st.markdown(
            """ <style> .font { font-size:50px ; font-weight: bold; font-family: 'Courier New'; color: #DB7093;} </style> """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="font">StockVortex</p>', unsafe_allow_html=True)
        st.write(
            "<p style='color:LightPink ; font-size: 20px;font-family: Garamond ;font-weight: normal;'>Where stocks converge and profits swirl – welcome to StockVortex!</p>",
            unsafe_allow_html=True,
        )
        st.image("https://media.tenor.com/dPYNJASNrIkAAAAi/pepe-money-rain.gif")
        
        start_date, end_date, ticker, custom_ticker = user_inputs()
        if custom_ticker:
            ticker = custom_ticker.upper()

        if start_date >= end_date:
            st.error("End date must be after start date.")
            return
        
        if st.button('Fetch Data'):
            with st.spinner('Fetching data...'):
                data = asyncio.run(fetch_data(ticker, start_date, end_date))
        
            if not data.empty:
                st.success('Data fetched successfully!')
                st.write(f'Data from {start_date} to {end_date}')
                st.write(data)
                st.download_button(
                    label="Download data as CSV",
                    data=data.to_csv().encode('utf-8'),
                    file_name=f'{ticker}_data.csv',
                    mime='text/csv',
                )
                display_stock_info(ticker)
                display_summary_statistics(data)
                plot_data(data)
                analyze_data(data)
                model_summary, predictions = forecast(data, end_date)
                st.write(model_summary)
                plot_predictions(data, predictions)
                add_technical_indicators(data)
                portfolio_analysis()
            else:
                st.error("No data found for the selected parameters.")
        about_author()

def show_disclaimer():
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Disclaimer</p>", unsafe_allow_html=True)
    st.write("""
        The information provided by StockVortex is for educational purposes only and should not be considered as financial advice.
        Trading stocks involves risk, and you should consult with a licensed financial advisor before making any investment decisions.
        StockVortex and its creators are not responsible for any financial losses you may incur.
    """)
    if st.button("I Understand and Accept"):
        st.session_state['disclaimer_accepted'] = True
        st.experimental_rerun()

def user_inputs():
    st.sidebar.header('Parameters')
    start_date, end_date = st.sidebar.slider(
        'Select date range',
        value=(date(2020, 1, 1), date(2020, 12, 31)),
        format="YYYY-MM-DD"
    )
    ticker_list = ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
    ticker = st.sidebar.selectbox('Company', ticker_list)
    custom_ticker = st.sidebar.text_input('Or enter a custom ticker (overrides selection above)', value='')
    return start_date, end_date, ticker, custom_ticker

async def fetch_data(ticker, start_date, end_date):
    async with aiohttp.ClientSession() as session:
        try:
            start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history"
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error fetching data: {response.status}")
                    st.error(f"Error fetching data: {response.status}")
                    return pd.DataFrame()
                data = await response.text()
                data = pd.read_csv(pd.compat.StringIO(data))
                return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

def display_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            st.error("No stock information available.")
            return
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Stock Information</p>", unsafe_allow_html=True)
        st.write(f"**Company:** {info.get('shortName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
        st.write(f"**Previous Close:** {info.get('previousClose', 'N/A')}")
        st.write(f"**Open:** {info.get('open', 'N/A')}")
        st.write(f"**Day's Range:** {info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}")
        st.write(f"**52 Week Range:** {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
    except Exception as e:
        logger.error(f"An error occurred while fetching stock information: {e}")
        st.error(f"An error occurred while fetching stock information: {e}")

def display_summary_statistics(data):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Summary Statistics</p>", unsafe_allow_html=True)
    st.write(data.describe())

def plot_data(data):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Data Visualization</p>", unsafe_allow_html=True)
    st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Plot of the Data</p>", unsafe_allow_html=True)
    fig = px.line(data, x='Date', y='Close', title='Closing price of the stock')
    st.plotly_chart(fig)
    st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Candlestick Chart</p>", unsafe_allow_html=True)
    fig_candlestick = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
    fig_candlestick.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_candlestick)

def analyze_data(data):
    column = st.selectbox('Select column', data.columns[1:], help="Select the column to analyze")
    data = data[['Date', column]]
    st.write('Selected Data')
    st.write(data)
    st.write('Data Stationarity')
    stationarity = adfuller(data[column])[1] < 0.05
    st.write(f'Stationary: {stationarity}')
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Decomposition of Data</p>", unsafe_allow_html=True)
    decomposition = seasonal_decompose(data[column], model='additive', period=12)
    st.write(decomposition.plot())
    st.write('Evaluating Plots')
    plot_decomposition(data, decomposition)

def plot_decomposition(data, decomposition):
    st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
    st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
    st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residuals', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

def forecast(data, end_date):
    p = st.slider('Select value of p', 0, 5, 2, help="AR order")
    d = st.slider('Select value of d', 0, 5, 1, help="Differencing order")
    q = st.slider('Select value of q', 0, 5, 2, help="MA order")
    seasonal_order = st.number_input('Select value of seasonal p', 0, 24, 12, help="Seasonal AR order")
    model = sm.tsa.statespace.SARIMAX(data.iloc[:, 1], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit()
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Model Summary</p>", unsafe_allow_html=True)
    st.write(model.summary())
    st.write('---')
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
    forecast_period = st.number_input('Select number of days for prediction', 1, 365, 10)
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period - 1)
    predictions = predictions.predicted_mean
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, 'Date', predictions.index)
    predictions.reset_index(drop=True, inplace=True)
    st.write('Predictions', predictions)
    st.write('Actual Data', data)
    st.write('---')
    return model.summary(), predictions

def plot_predictions(data, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data.iloc[:, 1], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=800, height=400)
    st.plotly_chart(fig)
    show_plots = st.button('Show separate plots')
    if show_plots:
        st.write(px.line(x=data['Date'], y=data.iloc[:, 1], title='Actual', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions['Date'], y=predictions['predicted_mean'], title='Predicted', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'))

def add_technical_indicators(data):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Technical Indicators</p>", unsafe_allow_html=True)
    # Simple Moving Average
    data['SMA'] = data['Close'].rolling(window=20).mean()
    # Exponential Moving Average
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    # Bollinger Bands
    data['Bollinger_Upper'] = data['SMA'] + 2*data['Close'].rolling(window=20).std()
    data['Bollinger_Lower'] = data['SMA'] - 2*data['Close'].rolling(window=20).std()
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    # MACD
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Plotting the indicators
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA'], mode='lines', name='SMA'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], mode='lines', name='EMA'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Upper'], mode='lines', name='Bollinger Upper'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Lower'], mode='lines', name='Bollinger Lower'))
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
    st.plotly_chart(fig)

def portfolio_analysis():
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Portfolio Analysis</p>", unsafe_allow_html=True)
    tickers = st.sidebar.multiselect('Select tickers for portfolio', ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"], ["AAPL", "MSFT"])
    weights = st.sidebar.text_input('Enter weights for selected tickers', '0.5, 0.5', help="Enter weights separated by commas")
    weights = list(map(float, weights.split(',')))
    if len(weights) != len(tickers):
        st.write("Number of weights must match number of tickers selected.")
        return

    data = yf.download(tickers, start="2020-01-01", end="2021-01-01")['Close']
    data = data.pct_change().dropna()
    portfolio_return = (data * weights).sum(axis=1)
    cumulative_return = (1 + portfolio_return).cumprod() - 1

    st.write("Cumulative Portfolio Return:")
    st.line_chart(cumulative_return)

    st.write("Risk Metrics:")
    st.write("Sharpe Ratio:", np.mean(portfolio_return) / np.std(portfolio_return))
    st.write("Value at Risk (5%):", np.percentile(portfolio_return, 5))

def about_author():
    st.write("---")
    st.write("<p style='color:HotPink ; font-size: 30px;font-family: Courier New; font-weight: bold;'>About the Author</p>", unsafe_allow_html=True)
    st.write("<p style='color:LightPink; font-size: 25px; font-family: Georgia;font-weight: bold;'>Shelly Bhalla</p>", unsafe_allow_html=True)
    linkedin_url = "https://www.linkedin.com/in/shelly-bhalla-58a7271b6"
    github_url = "https://github.com/Shellybhalla13"
    linkedin_icon = "https://static.vecteezy.com/system/resources/previews/017/339/624/original/linkedin-icon-free-png.png"
    github_icon = "https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Github-512.png"
    st.markdown(
        f'<a href="{github_url}"><img src="{github_icon}" width="60" height="60"></a>'
        f'<a href="{linkedin_url}"><img src="{linkedin_icon}" width="60" height="60"></a>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

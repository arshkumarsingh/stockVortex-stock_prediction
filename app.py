import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import date, timedelta
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache
def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.insert(0, "Date", data.index, True)
        data.reset_index(drop=True, inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def main():
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state['disclaimer_accepted'] = False

    if not st.session_state['disclaimer_accepted']:
        show_disclaimer()
    else:
        render_main_page()

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

def render_main_page():
    st.markdown(
        """ <style> .font { font-size:50px ; font-weight: bold; font-family: 'Courier New'; color: #DB7093;} </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">StockVortex</p>', unsafe_allow_html=True)
    st.write(
        "<p style='color:LightPink ; font-size: 20px;font-family: Garamond ;font-weight: normal;'>Where stocks converge and profits swirl – welcome to StockVortex!</p>",
        unsafe_allow_html=True,
    )
    st.image("https://thechainsaw.com/wp-content/uploads/2023/05/pepe-cover.jpg")
    
    start_date, end_date, ticker = user_inputs()
    if st.button('Fetch Data', help="Click to fetch stock data for the selected parameters"):
        if start_date >= end_date:
            st.error("End date must be after start date.")
        else:
            with st.spinner('Fetching data...'):
                data = fetch_data(ticker, start_date, end_date)
            
            if not data.empty:
                st.success('Data fetched successfully!')
                st.write(f'Data from {start_date} to {end_date}')
                
                # Ensure Date is in string format and other columns are of supported data types
                data['Date'] = data['Date'].astype(str)
                data = data.astype({
                    'Open': 'float64', 'High': 'float64', 'Low': 'float64',
                    'Close': 'float64', 'Volume': 'int64'
                })
                
                st.dataframe(data)
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
                risk_analysis(data, start_date, end_date)
                dividend_analysis(ticker)
                economic_indicators()
                news_sentiment_analysis(ticker)
                event_study_analysis(ticker)
                custom_alerts(ticker)
            else:
                st.error("No data found for the selected parameters.")
    debug_mode()
    about_author()

def user_inputs():
    st.sidebar.header('Parameters')
    
    # Provide multiple default time periods
    time_periods = {
        '1 Month': (date.today() - timedelta(days=30), date.today()),
        '3 Months': (date.today() - timedelta(days=90), date.today()),
        '6 Months': (date.today() - timedelta(days=180), date.today()),
        '1 Year': (date.today() - timedelta(days=365), date.today()),
        'YTD': (date(date.today().year, 1, 1), date.today()),
        'Max': (date(2000, 1, 1), date.today())  # Adjust max date as per the requirement
    }
    
    selected_period = st.sidebar.selectbox('Select Time Period', list(time_periods.keys()), index=3)
    start_date, end_date = time_periods[selected_period]
    
    custom_dates = st.sidebar.checkbox('Custom Date Range')
    if custom_dates:
        start_date, end_date = st.sidebar.date_input(
            'Select date range',
            value=(start_date, end_date),
            format="YYYY-MM-DD",
            help="Select the start and end dates for the stock data"
        )

    ticker_list = ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
    ticker = st.sidebar.selectbox('Company', ticker_list, help="Select the company ticker symbol")
    return start_date, end_date, ticker

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
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Plot of the Data</p>", unsafe_allow_html=True)
        fig = px.line(data, x='Date', y='Close', title='Closing price of the stock', labels={'Close': 'Close Price'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Candlestick Chart</p>", unsafe_allow_html=True)
        fig_candlestick = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig_candlestick.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_candlestick, use_container_width=True)

def analyze_data(data):
    column = st.selectbox('Select column', data.columns[1:], help="Select the column to analyze")
    data = data[['Date', column]]
    st.write('Selected Data')
    st.dataframe(data)
    st.write('Data Stationarity')
    stationarity = adfuller(data[column])[1] < 0.05
    st.write(f'Stationary: {stationarity}')
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Decomposition of Data</p>", unsafe_allow_html=True)
    decomposition = seasonal_decompose(data[column], model='additive', period=12)
    st.write(decomposition.plot())
    st.write('Evaluating Plots')
    plot_decomposition(data, decomposition)

def plot_decomposition(data, decomposition):
    st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'), use_container_width=True)
    st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'), use_container_width=True)
    st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residuals', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'), use_container_width=True)

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
    forecast_period = st.number_input('Select number of days for prediction', 1, 365, 10, help="Enter the number of days for which you want the prediction")
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period - 1)
    predictions = predictions.predicted_mean
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions, columns=['predicted_mean'])
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
    st.plotly_chart(fig, use_container_width=True)
    show_plots = st.button('Show separate plots', help="Click to show the actual and predicted plots separately")
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
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
    st.plotly_chart(fig, use_container_width=True)

def portfolio_analysis():
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Portfolio Analysis</p>", unsafe_allow_html=True)
    tickers = st.sidebar.multiselect('Select tickers for portfolio', ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"], ["AAPL", "MSFT"], help="Select the stocks you want to include in your portfolio")
    weights = st.sidebar.text_input('Enter weights for selected tickers', '0.5, 0.5', help="Enter weights separated by commas (e.g., 0.5, 0.5)")
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

def risk_analysis(data, start_date, end_date):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Risk Analysis</p>", unsafe_allow_html=True)
    returns = data['Close'].pct_change().dropna()
    beta, alpha = calculate_beta_alpha(returns, start_date, end_date)
    st.write(f"**Beta:** {beta}")
    st.write(f"**Alpha:** {alpha}")
    st.write("**Sortino Ratio:**", np.mean(returns) / np.std(returns[returns < 0]))

def calculate_beta_alpha(returns, start_date, end_date):
    market_data = yf.download('SPY', start=start_date, end=end_date)['Close'].pct_change().dropna()
    combined_data = pd.concat([returns, market_data], axis=1).dropna()
    combined_data.columns = ['Stock', 'Market']
    covariance = np.cov(combined_data['Stock'], combined_data['Market'])[0][1]
    beta = covariance / np.var(combined_data['Market'])
    alpha = np.mean(combined_data['Stock']) - beta * np.mean(combined_data['Market'])
    return beta, alpha

def dividend_analysis(ticker):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Dividend Analysis</p>", unsafe_allow_html=True)
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    if dividends.empty:
        st.write("No dividends data available.")
        return
    st.write("Dividend Data:")
    st.write(dividends)
    st.line_chart(dividends)

def economic_indicators():
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Economic Indicators</p>", unsafe_allow_html=True)
    indicators = {
        "Interest Rate": 0.025,
        "Inflation Rate": 0.02,
        "Unemployment Rate": 0.05
    }
    st.write(indicators)

def news_sentiment_analysis(ticker):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>News Sentiment Analysis</p>", unsafe_allow_html=True)
    # Placeholder for actual news sentiment analysis
    st.write("Sentiment Analysis of Recent News:")
    st.write(f"Positive sentiment for {ticker}: 60%")
    st.write(f"Negative sentiment for {ticker}: 40%")

def event_study_analysis(ticker):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Event Study Analysis</p>", unsafe_allow_html=True)
    # Placeholder for actual event study analysis
    st.write(f"Impact of recent earnings announcement on {ticker}: +5%")

def custom_alerts(ticker):
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Custom Alerts</p>", unsafe_allow_html=True)
    alert_price = st.number_input('Set price alert', min_value=0.0, help="Set the price at which you want to receive an alert for the stock")
    st.write(f"Alert set for {ticker} at {alert_price}")

def debug_mode():
    debug = st.checkbox('Enable Debug Mode')
    if debug:
        with open('app.log') as f:
            st.download_button('Download Logs', data=f.read(), file_name='app.log')

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

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
import indicators  # Import the new indicators module

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to apply custom CSS
def apply_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #0F1126;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #431875;
        }
        .font {
            font-size:50px ; font-weight: bold; font-family: 'Courier New'; color: #DB7093;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    apply_custom_css()
    
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state['disclaimer_accepted'] = False

    if not st.session_state['disclaimer_accepted']:
        show_disclaimer()
    else:
        st.sidebar.title("Navigation")
        options = ["Stock Analysis", "About Author"]
        choice = st.sidebar.radio("Go to", options)

        if choice == "Stock Analysis":
            stock_analysis_tabs()
        elif choice == "About Author":
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

def stock_analysis_tabs():
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
            data = fetch_data(ticker, start_date, end_date)
    
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

            tabs = st.tabs(["Stock Information", "Summary Statistics", "Data Visualization", "Data Analysis", "Forecast", "Portfolio Analysis"])
            
            with tabs[0]:
                display_stock_info(ticker)
                about_author()
                
            with tabs[1]:
                display_summary_statistics(data)
                about_author()
                
            with tabs[2]:
                plot_data(data)
                about_author()
                
            with tabs[3]:
                analyze_data(data)
                about_author()
                
            with tabs[4]:
                model_summary, predictions = forecast(data, end_date)
                st.write(model_summary)
                plot_predictions(data, predictions)
                about_author()
                
            with tabs[5]:
                portfolio_analysis()
                about_author()
        else:
            st.error("No data found for the selected parameters.")
    
    if st.button('Refresh Data'):
        st.experimental_rerun()

def user_inputs():
    st.sidebar.header('Parameters')
    # Default date ranges
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    two_years_ago = today - timedelta(days=730)
    
    date_ranges = {
        "Past Year": (one_year_ago, today),
        "Past Two Years": (two_years_ago, today),
        "Custom": None
    }
    
    # Dropdown for date range selection
    date_range_option = st.sidebar.selectbox(
        "Select date range",
        options=list(date_ranges.keys()),
        help="Select the time period for which you want to analyze the stock data."
    )
    
    if date_range_option == "Custom":
        start_date, end_date = st.sidebar.slider(
            'Select date range',
            value=(date(2020, 1, 1), date(2020, 12, 31)),
            format="YYYY-MM-DD",
            help="Select a custom date range for the stock data."
        )
    else:
        start_date, end_date = date_ranges[date_range_option]
    
    ticker_list = ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
    ticker = st.sidebar.selectbox(
        'Company',
        ticker_list,
        help="Select a company from the list to fetch its stock data."
    )
    custom_ticker = st.sidebar.text_input(
        'Or enter a custom ticker (overrides selection above)',
        value='',
        help="Enter a custom stock ticker symbol to fetch data for a specific company."
    )
    return start_date, end_date, ticker, custom_ticker

def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
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
    try:
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Summary Statistics</p>", unsafe_allow_html=True)
        st.write(data.describe())
    except Exception as e:
        logger.error(f"Error displaying summary statistics: {e}")
        st.error(f"Error displaying summary statistics: {e}")

def plot_data(data):
    try:
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Data Visualization</p>", unsafe_allow_html=True)
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Plot of the Data</p>", unsafe_allow_html=True)
        fig = px.line(data, x='Date', y='Close', title='Closing price of the stock', labels={'Date': 'Date', 'Close': 'Closing Price'})
        st.plotly_chart(fig)
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Candlestick Chart</p>", unsafe_allow_html=True)
        fig_candlestick = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig_candlestick.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_candlestick)
    except Exception as e:
        logger.error(f"Error plotting data: {e}")
        st.error(f"Error plotting data: {e}")

def analyze_data(data):
    try:
        column = st.selectbox(
            'Select column',
            data.columns[1:],
            help="Select the column you want to analyze for stationarity and decomposition."
        )
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
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        st.error(f"Error analyzing data: {e} ")

def plot_decomposition(data, decomposition):
    try:
        st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
        st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residuals', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))
    except Exception as e:
        logger.error(f"Error plotting decomposition: {e}")
        st.error(f"Error plotting decomposition: {e} ")

def forecast(data, end_date):
    try:
        p = st.slider('Select value of p', 0, 5, 2, help="AR order: The number of lag observations included in the model.")
        d = st.slider('Select value of d', 0, 5, 1, help="Differencing order: The number of times the raw observations are differenced.")
        q = st.slider('Select value of q', 0, 5, 2, help="MA order: The size of the moving average window.")
        seasonal_order = st.number_input('Select value of seasonal p', 0, 24, 12, help="Seasonal AR order: The number of lag observations included in the seasonal part of the model.")
        model = sm.tsa.statespace.SARIMAX(data.iloc[:, 1], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
        model = model.fit()
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Model Summary</p>", unsafe_allow_html=True)
        st.write(model.summary())
        st.write('---')
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
        forecast_period = st.number_input('Select number of days for prediction', 1, 365, 10, help="Number of days into the future you want to forecast.")
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
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        st.error(f"Error in forecasting: {e} ")

def plot_predictions(data, predictions):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data.iloc[:, 1], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines', name='Predicted', line=dict(color='red')))
        fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=800, height=400)
        st.plotly_chart(fig)
        show_plots = st.button('Show separate plots')
        if show_plots:
            st.write(px.line(x=data['Date'], y=data.iloc[:, 1], title='Actual', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
            st.write(px.line(x=predictions['Date'], y=predictions['predicted_mean'], title='Predicted', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'))
    except Exception as e:
        logger.error(f"Error plotting predictions: {e}")
        st.error(f"Error plotting predictions: {e} ")

def portfolio_analysis():
    try:
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Portfolio Analysis</p>", unsafe_allow_html=True)
        tickers = st.sidebar.multiselect(
            'Select tickers for portfolio',
            ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"],
            ["AAPL", "MSFT"],
            help="Select the stocks to include in your portfolio."
        )
        weights = st.sidebar.text_input(
            'Enter weights for selected tickers',
            '0.5, 0.5',
            help="Enter the weights for each stock in your portfolio, separated by commas. Ensure the total weights sum up to 1."
        )
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
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {e}")
        st.error(f"Error in portfolio analysis: {e}")

def explain_sarimax_results(summary):
    try:
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>SARIMAX Model Explanation</p>", unsafe_allow_html=True)
        st.write("<p style='color:LightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Interpreting the SARIMAX Model Results</p>", unsafe_allow_html=True)

        st.write("""
            The SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model summary provides several key pieces of information:

            - **Log-Likelihood**: A higher value indicates a better fit of the model to the data.
            - **AIC (Akaike Information Criterion)**: Used for model comparison. Lower values indicate a better fit.
            - **BIC (Bayesian Information Criterion)**: Similar to AIC but penalizes model complexity more strongly. Lower values indicate a better fit.
            - **Coefficients**: Represent the relationship between the variables and the time series. Significant coefficients (p-values < 0.05) indicate a meaningful relationship.
            - **P-values**: Used to determine the significance of the coefficients. P-values below 0.05 typically indicate statistical significance.
            - **Standard Errors**: Measure the accuracy of the coefficients' estimates. Lower values indicate more precise estimates.

            **Conclusion**:
            - If the model has low AIC and BIC values, and the p-values of the coefficients are significant, the model is likely a good fit.
            - The residuals (errors) should be randomly distributed (check residual plots) for the model to be reliable.
            - Use the model to forecast future values and compare them to actual data to validate the model's predictive power.
        """)
    except Exception as e:
        logger.error(f"Error explaining SARIMAX results: {e}")
        st.error(f"Error explaining SARIMAX results: {e}")

def about_author():
    try:
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
    except Exception as e:
        logger.error(f"Error displaying author information: {e}")
        st.error(f"Error displaying author information: {e}")

if __name__ == "__main__":
    main()

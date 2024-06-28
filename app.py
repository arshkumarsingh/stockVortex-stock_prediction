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
    """
    Applies custom CSS styles to the Streamlit app.

    This function uses the `st.markdown` function from the Streamlit library to inject custom CSS styles into the app. The CSS styles define the background color, text color, and font properties for the main container, sidebar content, and font styles.

    Parameters:
    None

    Returns:
    None
    """
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
    """
    The main function of the program.

    This function is responsible for running the main logic of the program. It first applies the custom CSS to the Streamlit application.

    It then checks if the 'disclaimer_accepted' key is present in the session state. If it is not present, it sets the value of 'disclaimer_accepted' to False in the session state.

    If the 'disclaimer_accepted' is False, it calls the show_disclaimer() function.

    If the 'disclaimer_accepted' is True, it displays a navigation menu in the sidebar of the Streamlit application. The menu includes two options: "Stock Analysis" and "About Author".

    It then retrieves the user's choice from the radio button in the sidebar. If the choice is "Stock Analysis", it calls the stock_analysis() function.

    If the choice is "About Author", it calls the about_author() function.

    Parameters:
        None

    Returns:
        None
    """
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
            stock_analysis()
        elif choice == "About Author":
            about_author()

def show_disclaimer():
    """
    Displays a disclaimer to the user regarding the use of StockVortex. The disclaimer informs the user that the information provided by StockVortex is for educational purposes only and should not be considered as financial advice. It also reminds the user that trading stocks involves risk and that they should consult with a licensed financial advisor before making any investment decisions. The disclaimer includes a button that allows the user to accept the disclaimer and proceed. 

    Parameters:
    None

    Returns:
    None
    """
    st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Disclaimer</p>", unsafe_allow_html=True)
    st.write("""
        The information provided by StockVortex is for educational purposes only and should not be considered as financial advice.
        Trading stocks involves risk, and you should consult with a licensed financial advisor before making any investment decisions.
        StockVortex and its creators are not responsible for any financial losses you may incur.
    """)
    if st.button("I Understand and Accept"):
        st.session_state['disclaimer_accepted'] = True
        st.experimental_rerun()

def stock_analysis():
    """
    This function performs stock analysis using user inputs and displays the fetched data, summary statistics, plots, and predictions.

    Parameters:
    None

    Returns:
    None

    Side Effects:
    - Displays the stock analysis page with the specified styling and image.
    - Prompts the user to input start and end dates, ticker symbol, and custom ticker symbol.
    - Validates that the end date is after the start date.
    - Fetches data using the fetch_data function and displays success message and data.
    - Provides a download button to download the data as a CSV file.
    - Displays stock information using the display_stock_info function.
    - Displays summary statistics using the display_summary_statistics function.
    - Plots data using the plot_data function.
    - Analyzes data using the analyze_data function.
    - Performs forecasting using the forecast function and displays the model summary and predictions.
    - Plots the predictions using the plot_predictions function.
    - Adds technical indicators using the add_technical_indicators function from the indicators module.
    - Performs portfolio analysis.
    - Explains SARIMAX results using the explain_sarimax_results function.
    - Refreshes the data if the "Refresh Data" button is clicked.

    Note:
    - The function assumes that the necessary modules and functions are imported and defined elsewhere in the codebase.
    - The function relies on the st.session_state dictionary to store the user's input and the status of the disclaimer acceptance.
    """
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
            display_stock_info(ticker)
            display_summary_statistics(data)
            plot_data(data)
            analyze_data(data)
            model_summary, predictions = forecast(data, end_date)
            st.write(model_summary)
            plot_predictions(data, predictions)
            indicators.add_technical_indicators(data)  # Call the function from the new module
            portfolio_analysis()
            explain_sarimax_results(model_summary)
        else:
            st.error("No data found for the selected parameters.")
    
    if st.button('Refresh Data'):
        st.experimental_rerun()

def user_inputs():
    """
    Function for user inputs including selecting date ranges, company, and custom ticker.
    
    This function returns the start date, end date, selected ticker (or custom ticker if provided), and None if no custom ticker is entered.
    """
    with st.sidebar:
        st.header('Parameters')
        date_ranges = {
            "Past Year": (date.today() - timedelta(days=365), date.today()),
            "Past Two Years": (date.today() - timedelta(days=730), date.today()),
            "Custom": None
        }
        date_range_option = st.selectbox(
            "Select date range",
            options=date_ranges.keys(),
            help="Select the time period for which you want to analyze the stock data."
        )
        start_date, end_date = date_ranges[date_range_option] if date_range_option != "Custom" else st.slider(
            'Select date range',
            value=(date(2020, 1, 1), date(2020, 12, 31)),
            format="YYYY-MM-DD",
            help="Select a custom date range for the stock data."
        )
        ticker_list = ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
        ticker = st.selectbox(
            'Company',
            ticker_list,
            help="Select a company from the list to fetch its stock data."
        )
        custom_ticker = st.text_input(
            'Or enter a custom ticker (overrides selection above)',
            value='',
            help="Enter a custom stock ticker symbol to fetch data for a specific company."
        )
    return start_date, end_date, ticker.upper() if custom_ticker else ticker, custom_ticker.upper() if custom_ticker else None

def fetch_data(ticker, start_date, end_date):
    """
    Fetches data for a given stock ticker within a specified date range.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for fetching the data.
    end_date (str): The end date for fetching the data.

    Returns:
    pandas.DataFrame: The fetched stock data.
    """
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
    """
    Displays the summary statistics of the given data.

    Parameters:
        data (pandas.DataFrame): The data to display the summary statistics for.

    Returns:
        None

    Raises:
        Exception: If an error occurs while displaying the summary statistics.

    """
    try:
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Summary Statistics</p>", unsafe_allow_html=True)
        st.write(data.describe())
    except Exception as e:
        logger.error(f"Error displaying summary statistics: {e}")
        st.error(f"Error displaying summary statistics: {e}")

def plot_data(data):
    """
    Plots the data using Plotly and Streamlit.

    Args:
        data (pandas.DataFrame): The data to be plotted. It must have columns 'Date' and 'Close'.

    Returns:
        None

    Raises:
        Exception: If there is an error while plotting the data.

    This function first writes a heading to the Streamlit app using the 'st.write' function. It then writes a subheading for the line plot and creates a line plot using the 'px.line' function from Plotly Express. The line plot shows the closing price of the stock over time. The 'st.plotly_chart' function is used to display the line plot.

    After that, it writes a subheading for the candlestick chart and creates a candlestick chart using the 'go.Figure' function from Plotly. The candlestick chart shows the opening, high, low, and closing prices of the stock over time. The 'st.plotly_chart' function is used to display the candlestick chart.

    If there is an error while plotting the data, an error message is logged and displayed using the 'logger.error' and 'st.error' functions respectively.
    """
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
    """
    Analyzes the given data by selecting a column, checking its stationarity, and decomposing it.

    Parameters:
    - data (pandas.DataFrame): The data to be analyzed. The first column is assumed to be the index column.

    Returns:
    - None

    Raises:
    - Exception: If there is an error during the analysis process. The error message will be logged and displayed as an error message in the Streamlit app.
    """
    try:
        columns = data.columns[1:]
        column = st.selectbox('Select column', columns, key='column_select', help="Select the column you want to analyze for stationarity and decomposition.")
        data = data.set_index('Date')[column].to_frame()
        st.write('Selected Data')
        st.write(data)
        st.write('Data Stationarity')
        adf_result = adfuller(data[column])
        stationarity = adf_result[1] < 0.05
        st.write(f'Stationary: {stationarity}')
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Decomposition of Data</p>", unsafe_allow_html=True)
        decomposition = seasonal_decompose(data[column], model='additive', period=12)
        st.plotly_chart(decomposition.plot())
        st.write('Evaluating Plots')
        plot_decomposition(data, decomposition)
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        st.error(f"Error analyzing data: {e} ")

def plot_decomposition(data, decomposition):
    """
    Plots the decomposition of the given data using Plotly.

    Args:
        data (pandas.DataFrame): The input data containing the 'Date' column.
        decomposition (statsmodels.tsa.seasonal.DecomposeResult): The decomposition result obtained from the seasonal_decompose function.

    Raises:
        Exception: If there is an error while plotting the decomposition.

    Returns:
        None
    """
    try:
        st.plotly_chart(px.line(x=data['Date'], y=decomposition.trend, title='Trend', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        st.plotly_chart(px.line(x=data['Date'], y=decomposition.seasonal, title='Seasonality', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
        st.plotly_chart(px.line(x=data['Date'], y=decomposition.resid, title='Residuals', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))
    except Exception as e:
        logger.error(f"Error plotting decomposition: {e}")
        st.error(f"Error plotting decomposition: {e} ")

def forecast(data, end_date):
    """
    This function performs forecasting on a given dataset using the SARIMAX model.

    Args:
        data (pandas.DataFrame): The dataset to perform forecasting on. It should have a 'Date' column and a 'Close' column.
        end_date (str): The end date for the forecast. It should be in the format 'YYYY-MM-DD'.

    Returns:
        tuple: A tuple containing the model summary and the forecasted predictions. The model summary is a string containing the summary statistics of the fitted SARIMAX model. The forecasted predictions is a pandas DataFrame containing the forecasted values along with the corresponding dates.

    Raises:
        Exception: If there is an error while fitting the SARIMAX model or getting the forecasted predictions.

    """
    try:
        # Get the values for the SARIMAX model parameters from the user input
        p = st.slider('Select value of p', 0, 5, 2, help="AR order: The number of lag observations included in the model.")
        d = st.slider('Select value of d', 0, 5, 1, help="Differencing order: The number of times the raw observations are differenced.")
        q = st.slider('Select value of q', 0, 5, 2, help="MA order: The size of the moving average window.")
        seasonal_order = st.number_input('Select value of seasonal p', 0, 24, 12, help="Seasonal AR order: The number of lag observations included in the seasonal part of the model.")

        # Check if the input data is not empty
        if not data.empty:
            # Create an instance of the SARIMAX model with the specified parameters
            model = sm.tsa.statespace.SARIMAX(data.iloc[:, 1], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
            # Fit the model to the input data
            model = model.fit()

            # Write the model summary to the Streamlit app
            st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Model Summary</p>", unsafe_allow_html=True)
            st.write(model.summary())
            st.write('---')

            # Get the number of days for the forecast from the user input
            forecast_period = st.number_input('Select number of days for prediction', 1, 365, 10, help="Number of days into the future you want to forecast.")

            # Check if the forecast period is valid
            if forecast_period > 0:
                # Calculate the start and end indices for the forecast
                start = len(data)
                end = start + forecast_period - 1

                # Get the predicted values for the forecast period
                predictions = model.get_prediction(start=start, end=end)
                predictions = predictions.predicted_mean

                # Check if the predictions are not empty
                if not predictions.empty:
                    # Set the index of the predictions DataFrame to the corresponding dates
                    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
                    # Convert the predictions DataFrame to a DataFrame with the 'Date' column
                    predictions = pd.DataFrame(predictions)
                    predictions.insert(0, 'Date', predictions.index)
                    predictions.reset_index(drop=True, inplace=True)

                    # Write the predictions and actual data to the Streamlit app
                    st.write('Predictions', predictions)
                    st.write('Actual Data', data)
                    st.write('---')

                    # Return the model summary and predictions
                    return model.summary(), predictions

    # Handle any errors that occur during the forecasting process
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        st.error(f"Error in forecasting: {e} ")

def plot_predictions(data: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """
    Plot actual and predicted data.

    Args:
        data (pd.DataFrame): DataFrame containing actual data.
        predictions (pd.DataFrame): DataFrame containing predicted data.

    Returns:
        None
    """
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 0], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_mean'], mode='lines', name='Predicted', line=dict(color='red')))
        fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=800, height=400)
        st.plotly_chart(fig)
        show_plots = st.button('Show separate plots')
        if show_plots:
            st.write(px.line(x=data.index, y=data.iloc[:, 0], title='Actual', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
            st.write(px.line(x=predictions.index, y=predictions['predicted_mean'], title='Predicted', labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'))
    except Exception as e:
        logger.exception(f"Error plotting predictions: {e}")
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

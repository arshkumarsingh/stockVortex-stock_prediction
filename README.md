# StockVortex

StockVortex is a comprehensive web application built with Streamlit that provides a wide array of features for stock analysis, including fetching historical stock data, visualizing stock prices, performing technical analysis, risk analysis, forecasting, and more. This application is designed to help users make informed investment decisions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Author](#author)

## Features

- **Disclaimer Page**: Ensures users understand the educational nature of the application.
- **Fetch Historical Stock Data**: Fetch stock data from Yahoo Finance for user-specified time periods.
- **Display Stock Information**: Show detailed stock information including company details, market cap, and more.
- **Summary Statistics**: Provide descriptive statistics of the fetched data.
- **Data Visualization**: Visualize stock prices using line charts and candlestick charts.
- **Technical Analysis**: Add and visualize technical indicators like SMA, EMA, Bollinger Bands, RSI, and MACD.
- **Data Analysis**: Perform seasonal decomposition and stationarity tests on stock data.
- **Forecasting**: Use SARIMA models to forecast future stock prices.
- **Portfolio Analysis**: Analyze the performance of a user-defined portfolio.
- **Risk Analysis**: Calculate and display risk metrics such as Beta, Alpha, Sortino Ratio, and VaR.
- **Dividend Analysis**: Show and visualize dividend data for selected stocks.
- **Economic Indicators**: Display key economic indicators like interest rate, inflation rate, and unemployment rate.
- **News Sentiment Analysis**: Placeholder for sentiment analysis of recent news related to the stock.
- **Event Study Analysis**: Placeholder for analyzing the impact of significant events on stock prices.
- **Custom Alerts**: Allow users to set price alerts for selected stocks.
- **Debug Mode**: Enable users to download application logs for troubleshooting.
- **About the Author**: Display author information with links to LinkedIn and GitHub profiles.

## Installation

To run StockVortex locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/StockVortex.git
   cd StockVortex
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Disclaimer**: Upon starting the application, users will be presented with a disclaimer. Accept the disclaimer to proceed to the main page.

2. **Main Page**: Select the desired time period and stock ticker from the sidebar. Optionally, set a custom date range.

3. **Fetch Data**: Click the "Fetch Data" button to retrieve and display the stock data.

4. **Analyze Data**: Use the provided options to perform various analyses and visualizations on the fetched stock data.

5. **Explore Additional Features**: Navigate through various sections such as technical indicators, portfolio analysis, risk analysis, dividend analysis, and more to gain deeper insights.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

- **Shelly Bhalla**
  - [LinkedIn](https://www.linkedin.com/in/shelly-bhalla-58a7271b6)
  - [GitHub](https://github.com/Shellybhalla13)

Feel free to connect on LinkedIn and explore other projects on GitHub!

---

By using StockVortex, you agree to the disclaimer and acknowledge that the information provided is for educational purposes only and should not be considered financial advice. Always consult with a licensed financial advisor before making any investment decisions.

---

**Note**: Some features like news sentiment analysis and event study analysis are placeholders and need to be implemented as per the project's requirements.

For any questions or issues, please raise an issue in the GitHub repository or contact the author directly.

Enjoy exploring and analyzing stock data with StockVortex!
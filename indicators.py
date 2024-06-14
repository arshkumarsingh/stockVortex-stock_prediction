# indicators.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def add_technical_indicators(data):
    try:
        st.write("<p style='color:HotPink; font-size: 40px; font-family: Courier New;font-weight: bold;'>Technical Indicators</p>", unsafe_allow_html=True)
        
        # Simple Moving Average
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Simple Moving Average (SMA)</p>", unsafe_allow_html=True)
        data['SMA'] = data['Close'].rolling(window=20).mean()
        
        # Exponential Moving Average
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Exponential Moving Average (EMA)</p>", unsafe_allow_html=True)
        data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Bollinger Bands</p>", unsafe_allow_html=True)
        data['Bollinger_Upper'] = data['SMA'] + 2*data['Close'].rolling(window=20).std()
        data['Bollinger_Lower'] = data['SMA'] - 2*data['Close'].rolling(window=20).std()
        
        # RSI
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Relative Strength Index (RSI)</p>", unsafe_allow_html=True)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        
        # MACD
        st.write("<p style='color:lightPink; font-size: 25px; font-family: Courier New;font-weight: normal;'>Moving Average Convergence Divergence (MACD)</p>", unsafe_allow_html=True)
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
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")

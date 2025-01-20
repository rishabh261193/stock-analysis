import base64
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image file '{image_path}' not found. Please check the path.")
        return None
# Set the page configuration with a custom favicon
image_path = "RVSID.png"  # Replace with the correct path to your image
image_base64 = encode_image(image_path)

if image_base64:  # Proceed only if image encoding was successful
    st.set_page_config(
        page_title="Stock Market Analysis",
        page_icon=f"data:image/png;base64,{image_base64}",  # Use encoded image for the favicon
        layout="wide"
    )

# Add custom CSS for sticky header
st.markdown(
    """
    <style>
    .sticky-header {
        position: fixed;
        top: 0;  /* Ensure header starts at the top */
        left: 0;
        right: 0;
        background-color: lightgreen;
        z-index: 1000;
        display: flex;
        flex-direction: column;  /* Stack logo and header vertically */
        align-items: center;  /* Center the logo and title horizontally */
        justify-content: center;  /* Center them vertically */
        text-align: center;
        padding: 10px 0;  /* Padding to provide space for header content */
    }
    .sticky-header img {
        width: 50px;
        margin-bottom: 10px;
    }
    .sticky-header h1 {
        font-size: 40px;
        color: darkblue;
        margin: 0;
    }
    .stApp > div {
        padding-top: 120px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

# Create a sticky header with your logo and title
st.markdown(
    f"""
    <div class="sticky-header">
        <img src="data:image/png;base64,{image_base64}" alt="Logo">
        <h1>Stock Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)
else:
    st.error("Unable to load header icon. Please check the image path.")



# Helper Functions
def fetch_data(ticker, start_date, end_date):
    """Fetch stock data and financial info from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Adj Close"] = data["Close"]
    
    # Fetch the stock info to get P/E ratio
    stock_info = yf.Ticker(ticker).info

    # Return the stock data and P/E ratio
    return data, stock_info

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering(data):
    """Perform feature engineering on stock data."""
    data["RSI"] = calculate_rsi(data["Adj Close"], 14)
    return data.dropna()

def detect_breakout(data):
    """Detect breakout patterns in stock data."""
    recent_high = data['High'][-50:].max().item()
    last_close = data['Close'].iloc[-1].item()
    price_breakout = "Yes" if last_close > recent_high else "No"

    avg_volume = data['Volume'][-50:].mean().item()
    last_volume = data['Volume'].iloc[-1].item()
    volume_spike = "Yes" if last_volume > 1.5 * avg_volume else "No"

    rsi = data["RSI"].iloc[-1].item()
    rsi_breakout = "Yes" if rsi < 30 else "No"

    return {
        "Price Breakout": price_breakout,
        "Volume Spike": volume_spike,
        "RSI Breakout": rsi_breakout
    }
    
def classify_stock(breakout_data):
        indicators = list(breakout_data.values()).count("Yes")
        if indicators >= 3:
            return "Strong Chance"
        elif indicators == 2:
            return "Low Chance"
        elif indicators == 1:
            return "Hold"
        elif indicators == 0:
            return "Potential Sell"
        else:
            return "High Potential Sell"


def calculate_financial_ratios(stock_info, data):
    try:
        eps = stock_info.get("trailingEps", None)
        market_price = data["Close"].iloc[-1].item()
        book_value = stock_info.get("bookValue", None)
        roe = stock_info.get("returnOnEquity", None)
        debt_to_equity = stock_info.get("debtToEquity", None)
        interest_expense = stock_info.get("interestExpense", None)
        ebit = stock_info.get("ebitda", None)
        pe_ratio = stock_info.get("trailingPE", None)  # Ensure this key is fetched

        # Calculate ratios
        pb_ratio = market_price / book_value if book_value else None
        peg_ratio = stock_info.get("pegRatio", None)
        quick_ratio = stock_info.get("quickRatio", None)
        enterprise_value = stock_info.get("enterpriseValue", None)
        ev_to_ebit = (enterprise_value / ebit) if ebit else None
        operating_margin = stock_info.get("operatingMargins", None)

        # Interest Coverage Ratio
        icr = (ebit / interest_expense) if ebit and interest_expense else None

        return {
            "EPS": eps,
            "P/E Ratio": pe_ratio,
            "P/B Ratio": pb_ratio,
            "PEG Ratio": peg_ratio,
            "ROE": roe,
            "Debt/Equity Ratio": debt_to_equity,
            "Interest Coverage Ratio": icr,
            "EV/EBIT": ev_to_ebit,
            "Operating Margin": operating_margin,
            "Quick Ratio": quick_ratio,
        }
    except KeyError as e:
        logging.error(f"Key error in financial ratios calculation: {e}")
        return {"Error": "Data unavailable"}

def decide_buy_sell(financial_ratios):
    """Provide a buy/sell decision based on financial ratios."""
    try:
        buy_signals = 0
        sell_signals = 0

        # Example thresholds
        if "P/E Ratio" in financial_ratios and financial_ratios["P/E Ratio"] and financial_ratios["P/E Ratio"] < 15:
            buy_signals += 1
        elif "P/E Ratio" in financial_ratios and financial_ratios["P/E Ratio"] and financial_ratios["P/E Ratio"] > 25:
            sell_signals += 1
        if "PEG Ratio" in financial_ratios and financial_ratios["PEG Ratio"] and financial_ratios["PEG Ratio"] < 1:
            buy_signals += 1
        if "ROE" in financial_ratios and financial_ratios["ROE"] and financial_ratios["ROE"] > 15:
            buy_signals += 1
        if "Debt/Equity Ratio" in financial_ratios and financial_ratios["Debt/Equity Ratio"] and financial_ratios["Debt/Equity Ratio"] > 50:
            sell_signals += 1
        if "Quick Ratio" in financial_ratios and financial_ratios["Quick Ratio"] and financial_ratios["Quick Ratio"] < 1:
            sell_signals += 1

        # Final decision
        if buy_signals >= 3:
            return "Strong Buy"
        elif sell_signals >= 3:
            return "Strong Sell"
        elif buy_signals > sell_signals:
            return "Buy"
        elif sell_signals > buy_signals:
            return "Sell"
        else:
            return "Hold"
    except KeyError as e:
        return f"Missing key in financial_ratios: {e}"
      
    print(financial_ratios)

def predict_closing_price(data):
    """Predict the next day's closing price using linear regression."""
    recent_data = data[-30:]
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data["Adj Close"].values
    model = LinearRegression()
    model.fit(X, y)
    next_day = len(recent_data)
    return model.predict([[next_day]])[0].item()
    
def evaluate_market(data):
    ma20 = data["Adj Close"].rolling(window=20).mean().iloc[-1].item()
    ma50 = data["Adj Close"].rolling(window=50).mean().iloc[-1].item()
    return "Bullish" if ma20 > ma50 else "Bearish"
        
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom "Top Bar" Layout using Columns
top_bar = st.container()
with top_bar:
    with open("RVSID.png", "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode()
    
    # Centered welcome text
    st.markdown(
        """
        <p style="text-align: center; font-size: 18px;">
            Welcome to the Stock Analysis! Analyze stock trends and make informed decisions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .stDateInput, .stWrite {
            margin-top: -40px;  /* Adjust this value as needed */
            margin-bottom: -5px;  /* Adjust this value as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # First Row: Start Date and End Date
    row1_col1, row1_col2 = st.columns([1, 1])  # Create two columns
    with row1_col1:
        st.write("### Start Date")
        start_date = st.date_input("", value=date.today() - timedelta(days=150), label_visibility="hidden")  # Hidden label for accessibility
    with row1_col2:
        st.write("### End Date")
        end_date = st.date_input("", value=date.today(), label_visibility="hidden")  # Hidden label for accessibility
        
    
    st.write("### Analysis Mode")
    stocks = ["ADANIENSOL", "ADANIENT", "ADANIPOWER", "ANANTRAJ", "APOLLO",
                    "ARE&M", "ASIANPAINT", "ASTRAMICRO", "AXISBANK", "BAJAJFINSV",
                    "BARBEQUE", "BBOX", "BDL", "BEL", "BHARATFORG", 
                    "BHARTIARTL", "DABUR", "DRREDDY", "EMAMILTD", "GMRAIRPORT", 
                    "GMRP&UI", "GODREJCP", "HAVELLS", "HDFCBANK", "HFCL",
                    "HINDALCO", "HSCL", "ICICIBANK", "IDEA", "INDHOTEL",
                    "INFIBEAM", "IOC", "IRB", "IRCTC", "ITC",
                    "JIOFIN", "JSWSTEEL", "LORDSCHLO", "LTF", "MARICO",  
                    "MEDPLUS", "MGL", "NHPC", "NTPC", "ONGC",
                    "ORIENTHOT", "ORIENTTECH", "PARAS", "PGEL", "POWERGRID",  
                    "RELIANCE", "ROSSELLIND", "SALSTEEL", "SBIN", "SHRIRAMFIN",
                    "SUZLON", "SUNPHARMA", "TATACHEM", "TATACOMM", "TATACONSUM",
                    "TATAINVEST", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATATECH",
                    "TCIFINANCE", "TRIDENT", "TTML", "UBL", "UNIONBANK",  
                    "WIPRO", "YESBANK", "ZEEL", "ZOMATO"]  # Replace with your stock list

     # Centered Dropdown for Analysis Mode
    analysis_mode = st.selectbox("", ["All Stocks", "Choose Stocks"], index=0)

    # Display the selected mode
    if analysis_mode == "Choose Stocks":
        selected_stocks = st.multiselect("Select Stocks", stocks)
    else:
        selected_stocks = stocks

    # Custom CSS to center the dropdown
    st.markdown(
        """
        <style>
        .stSelectbox, .stRadio {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add `.NS` suffix for NSE
stocks_with_ns = [f"{stock}.NS" for stock in selected_stocks]



# Button
if st.button("Analyze Stocks"):
    
    # Create a container to display the results
    st.header("Stock Analysis Results")
    
    # Organize the results into rows of three columns
    columns_per_row = 3
    all_stocks = []
    
    for stock in stocks_with_ns:
        try:
            # Fetch and process data
            data, stock_info = fetch_data(stock, start_date, end_date)
            data = feature_engineering(data)
            breakout_data = detect_breakout(data)
            trend = evaluate_market(data)
            classification = classify_stock(breakout_data)
            financial_ratios = calculate_financial_ratios(stock_info, data)
            decision = decide_buy_sell(financial_ratios)
            predicted_price = predict_closing_price(data)
            last_close = data["Close"].iloc[-1].item()
            formatted_ratios = {key: f"{value:.2f}" if isinstance(value, (int, float)) else value
            for key, value in financial_ratios.items()}
            
            # Determine the background color based on the decision
            color_map = {
                "Strong Buy": "darkgreen",
                "Strong Sell": "darkred",
                "Buy": "green",
                "Sell": "lightcoral",
                "Hold": "orange"
            }
            box_color = color_map.get(decision, "gray")
            
            # Add stock info to all_stocks for display
            all_stocks.append((stock, last_close, predicted_price, breakout_data, formatted_ratios, decision, trend, box_color))
        
        except ValueError as ve:
            st.warning(f"{stock}: {ve}")
        except Exception as e:
            st.error(f"Error analyzing {stock}: {e}")
            
    # Display the results in rows of 3 columns
    for i in range(0, len(all_stocks), columns_per_row):
        row = st.columns(columns_per_row)  # Create 3 columns for each row

        # Loop through the current row's stock data and display it
        for j, (stock, last_close, predicted_price, breakout_data, formatted_ratios, decision, trend, box_color) in enumerate(all_stocks[i:i+columns_per_row]):
            with row[j]:
                # Assuming breakout_data is a dictionary
                breakout_list = "".join([f"<li style='font-size: 16px; line-height: 1;'><strong>{key}:</strong> {value}</li>" for key, value in breakout_data.items()])

                # Create a bullet list for Financial Ratios with reduced line spacing
                financial_ratios_list = "".join([f"<li style='font-size: 16px; line-height: 1;'><strong>{key}:</strong> {value}</li>" for key, value in formatted_ratios.items()])

                # Display the stock data with reduced line spacing
                st.markdown(f"""
                    <div style="border: 2px solid {box_color}; padding: 15px; background-color: {box_color}; color: white; border-radius: 10px; margin-bottom: 20px; font-size: 18px; line-height: 1.2;">
                        <h3 style="font-size: 24px; text-align: center; line-height: 1.2;">{stock}</h3>
                        <p style="font-size: 18px; line-height: 0.5;"><strong>Last Close:</strong> ₹{last_close:.2f}</p>
                        <p style="font-size: 18px; line-height: 0.5;"><strong>Predicted Price:</strong> ₹{predicted_price:.2f}</p>
                        <p style="font-size: 18px; line-height: 0.1;"><strong>Breakout Indicators:</strong><br><ul>{breakout_list}</ul></p>
                        <p style="font-size: 18px; line-height: 0.1;"><strong>Financial Ratios:</strong><br><ul>{financial_ratios_list}</ul></p>
                        <p style="font-size: 18px; line-height: 0.5;"><strong>Decision:</strong> {decision}</p>
                        <p style="font-size: 18px; line-height: 0.5;"><strong>Market Trend:</strong> {trend}</p>
                    </div>
                """, unsafe_allow_html=True)

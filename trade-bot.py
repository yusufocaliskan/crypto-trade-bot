import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import ta
import time
import joblib
import logging
import numpy as np

# Logger 
logging.basicConfig(filename='bot.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# APIS Envs.
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
client = Client(API_KEY, API_SECRET)

# Pairs
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
INTERVAL = '1h'

# Percentages for Stop-loss and take-profit 
STOP_LOSS_PERCENTAGE = 0.01  # %1 stop-loss
TAKE_PROFIT_PERCENTAGE = 0.02  # %2 take-profit

# the Risk percentage of the balance 
RISK_PER_TRADE = 0.01

model = joblib.load('model.pkl')


def get_account_balance(asset='USDT'):
    try:
        balance = client.get_asset_balance(asset=asset)
        return float(balance['free'])
    except BinanceAPIException as e:
        logging.error(f" ERror : {e}")
        return None

def get_realtime_data(symbol, interval):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=200)
        data = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                             'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                             'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
        data['Close'] = data['Close'].astype(float)
        data['Open'] = data['Open'].astype(float)
        data['High'] = data['High'].astype(float)
        data['Low'] = data['Low'].astype(float)
        data['Volume'] = data['Volume'].astype(float)
        data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
        data.set_index('Open Time', inplace=True)
        return data
    except BinanceAPIException as e:
        logging.error(f"Error - Binance: {e}")
        return None

def preprocess_data(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['Bollinger_Upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
    data['Bollinger_Lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    data.dropna(inplace=True)
    return data

# Make decision with the trained model.
def make_decision(data):
    latest_data = data.iloc[-1]
    features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    X = latest_data[features].values.reshape(1, -1)
    prediction = model.predict(X)
    return prediction[0]

def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss_price):
    risk_amount = account_balance * risk_percentage
    stop_loss_amount = abs(entry_price - stop_loss_price)
    position_size = risk_amount / stop_loss_amount
    return position_size

def place_order(side, quantity, symbol):
    try:
        logging.info(f" ORder for : {symbol} - {side}.  Amount: {quantity}")
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity)
        logging.info(f"Success: {order}")
        return order
    except BinanceAPIException as e:
        logging.error(f"Oorder Error--> BinanceAPIException: {e}")
        return None
    except BinanceOrderException as e:
        logging.error(f"BinanceOrderException-> : {e}")
        return None
    except Exception as e:
        logging.error(f"UnKnown Error: {e}")
        return None

def run_bot():
    positions = {}  
    while True:
        account_balance = get_account_balance()
        if account_balance is None:
            logging.error("balance loading...")
            time.sleep(60)
            continue

        for symbol in SYMBOLS:
            data = get_realtime_data(symbol, INTERVAL)
            if data is None:
                logging.error(f"Connot get data for {symbol}, still waiting...")
                continue
            data = preprocess_data(data)
            signal = make_decision(data)
            current_price = data['Close'].iloc[-1]

            if symbol not in positions:
                positions[symbol] = {'position': 0, 'entry_price': 0, 'stop_loss': 0, 'take_profit': 0}

            position = positions[symbol]['position']

            if position == 0 and signal == 1:

                # The signal of BUY 
                # Calculate Stop-loss and position amount 
                stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE)
                quantity = calculate_position_size(account_balance, RISK_PER_TRADE, current_price, stop_loss_price)
                quantity = round(quantity, 6)  
                order = place_order('BUY', quantity, symbol)
                if order:
                    positions[symbol]['position'] = 1
                    positions[symbol]['entry_price'] = current_price
                    positions[symbol]['stop_loss'] = stop_loss_price
                    positions[symbol]['take_profit'] = current_price * (1 + TAKE_PROFIT_PERCENTAGE)
                    logging.info(f"Position : {symbol}. Price: {current_price}, Stop-Loss: {stop_loss_price}, Take-Profit: {positions[symbol]['take_profit']}")
            elif position == 1:
                # Possion is open 
                if current_price <= positions[symbol]['stop_loss']:
                    logging.info(f"Reach Level : for {symbol} Stop-Loss")
                    order = place_order('SELL', quantity, symbol)
                    if order:
                        positions[symbol]['position'] = 0
                        logging.info(f"{symbol} - Position Closed, Close Price: {current_price}")
                elif current_price >= positions[symbol]['take_profit']:
                    logging.info(f"Reach take-profit {symbol} level.")
                    order = place_order('SELL', quantity, symbol)
                    if order:
                        positions[symbol]['position'] = 0
                        logging.info(f"{symbol} Positon has been closed. Exit price: {current_price}")
                elif signal == 0:
                    # Signal of the SELL 
                    logging.info(f"{symbol} Signal for SELL.")
                    order = place_order('SELL', quantity, symbol)
                    if order:
                        positions[symbol]['position'] = 0
                        logging.info(f"{symbol} - Position has been closed, exit price: {current_price}")
            else:
                logging.info(f"{symbol} Wating...")

        time.sleep(3600)

if __name__ == "__main__":
    run_bot()

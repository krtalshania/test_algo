#!pip install fyers-apiv2
# !pip install pandas_ta

# from IPython.display import clear_output
import requests
import pytz
from fyers_api import fyersModel, accessToken
from fyers_api.Websocket import ws
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta, datetime
import time

tz_NY  = pytz.timezone('Asia/Kolkata')

SLEEP_TIME = 60 

client_id="XU00399"
secret_key="R1JSCXLJ9G"
app_id = "CK7JSOQXD1-100"
redirect_uri="https://www.google.com"
response_type="code"
grant_type = "authorization_code"
access_token = '';

def t(msg):
  bot_token = '5124822628:AAExc_63O4Euolz7CRJyGSlOwQc6kP8-jh8'
  bot_chatID = '499083936'
  link = 'https://api.telegram.org/bot'+bot_token+'/sendMessage?chat_id='+bot_chatID+'&parse_mode=MarkdownV2&text=' + msg
  return requests.get(link).json()

def processAuthentication():
  session=accessToken.SessionModel(
      client_id=app_id,
      secret_key=secret_key,
      redirect_uri=redirect_uri, 
      response_type=response_type, 
      grant_type=grant_type)

  response = session.generate_authcode() 

  print(response)
  auth_code = input("Enter auth code: ")

  session.set_token(auth_code)
  return session.generate_token()

def login():
  print('Logging in')
  response = processAuthentication()
  if(response["s"]=="ok"):
    print(response)
    return response["access_token"]
  else:
    print("error occured, please try again")
    login()

def getTime():
  return datetime.now(tz_NY).strftime("%Y-%m-%d %I:%M:%S %p")



def get_data(range_from, range_to, symbol):
  global fyers, timeframe
  data = {"symbol":symbol,"resolution":timeframe,"date_format":"1","range_from":range_from,"range_to":range_to,"cont_flag":"1"}
  ss = fyers.history(data)['candles']
  # print(ss)
  df = pd.DataFrame(data=ss,columns=['date', 'open','high','low','close','volume'])
  df['date'] =[ datetime.fromtimestamp(float(x),tz_NY).strftime('%Y-%m-%d %I:%M:%S %p') for x in df['date']]
  return df
# get_data('2022-10-25', '2022-10-25', "NSE:NIFTYBANK-INDEX")

def getHistoricalData(range_from, range_to, symbol = "NSE:NIFTYBANK-INDEX"):
  global atr_period, atr_multiplier
  df = get_data(range_from, range_to, symbol)
  if not df.empty:
    df = refactorDataWithSupertrend(df)
  return df

def getQuote(symbol):
  global fyers
  return fyers.quotes({"symbols":symbol})

def getLTP(symbol):
  global fyers
  q = fyers.quotes({"symbols":symbol})['d'][0]['v']['lp']
  # print(q)
  return q

def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
      yield start_date + timedelta(n)

atr_period = 7
atr_multiplier = 2.5
timeframe = 3
currentTrend = ''
initial = True
orderPlaced = False
nPointsCaptured = 0
nPreviousOrderPrice = 0;
bLiveMode = True
start_date = date(2022, 10, 4)
end_date = date(2022, 10, 5)
bCloseDay = False

if bLiveMode:
  start_date = datetime.today().date()
  end_date = datetime.today().date()

resultdf = pd.DataFrame(index=[0])

d = datetime.now()
y = d.strftime("%y")
m = d.strftime("%b").upper()
currentFuture = f"NSE:BANKNIFTY{y}NOVFUT"

if bLiveMode:
  startLive()
else:
  startHistorical()

import time
def startLive():
  global resultdf, start_date, end_date, timeframe, bCloseDay, nPreviousOrderPrice, currentTrend
  print(f'Refreshing every {timeframe} minutes')
  while not bCloseDay:
    data = getHistoricalData(start_date, end_date).tail(7)
    # print(data)
    # worksheet.update([data.columns.values.tolist()] + data.values.tolist())
    futLTP = getLTP(currentFuture)
    currentTrendString = 'sell' if currentTrend=='down' else 'buy'
    if orderPlaced:
      points = futLTP - nPreviousOrderPrice if currentTrend=='up' else nPreviousOrderPrice - futLTP
      print(f'Current position: {currentTrendString} at {nPreviousOrderPrice} and FUTURE LTP : {futLTP}, points = {points}')
    processLiveSupertrend(start_date, data, futLTP)
    time.sleep(timeframe*60)
    
  # resultdf = resultdf.iloc[1:]
  # print(resultdf)

# startLive()

def processHistoricalSupertrend(day, df):
  global nPointsCaptured, resultdf
  nPointsCaptured = 0
  initial = True
  bIsWaitPrinted = False
  for index, row in df.iterrows():
    trend = row['STX']
    if(trend=='nan'):
      if(not bIsWaitPrinted):
        print('Waiting for clear direction')
        bIsWaitPrinted = True
      continue
    dt = row['date']
    ltp = row['fut_ltp']
    close = row['close']
    if initial:
      currentTrend = trend
      print(f'{dt} INIT Current trend: {trend}, INDEX: {close}, FUT: {ltp}')
      initial = False
      placeOrder(dt,ltp, trend)
    else:
      if currentTrend != trend:
        currentTrend = trend
        print(f'{dt} Trend changed to : {trend}, INDEX: {close}, FUT: {ltp}')
        placeOrder(dt,ltp, trend)
    
    candleTime = datetime.strptime(dt,'%Y-%m-%d %I:%M:%S %p').time()    
    closingTime = datetime.strptime("15:18:00",'%H:%M:%S').time()
    # print(candleTime, ", " , closingTime, candleTime == closingTime)
    if(candleTime == closingTime):
      # print('closing day')
      closeTheDay(trend,row, ltp)
      break
  # print(day, 'Total points captured : ', nPointsCaptured)
  resultdf = resultdf.append({ 'day': day, 'points': nPointsCaptured },ignore_index=True)

def startHistorical():
  global resultdf, start_date, end_date, timeframe
  for single_date in daterange(start_date, end_date):
    if(single_date.weekday() in [5,6]):
      continue
    range_from = single_date.strftime("%Y-%m-%d")
    range_to = single_date.strftime("%Y-%m-%d")
    histdata = getHistoricalData(range_from, range_to)
    # histdata.rename(columns={'STX':'trend'}, inplace=True)  
    if(not histdata.empty):
      futData = getHistoricalData(range_from, range_to, currentFuture)
      futData = futData.drop(['high','low','close','STX', 'TR','ATR_7', 'ST','date'], axis=1)
      futData.rename(columns={'open':'fut_ltp'}, inplace=True)
      histdata = histdata.join(futData)
      # print(futData)
      # for i, row in histdata.iterrows():
      #   print(row)

      processHistoricalSupertrend(range_from, histdata)
  resultdf = resultdf.iloc[1:]
  print(resultdf)

def closeTheDay(trend,row, futLTP):
  global nPointsCaptured, resultdf, nPointsCaptured, bClose
  trend = row['STX'] if type(row['STX']) == str else row['STX'].values[0]
  indexPrice = row['close'] if type(row['close']) == float else row['close'].values[0]
  ltp = futLTP
  points = 0
  msg = 'Day end, Closing '
  if trend=='up':
    points = ltp - nPreviousOrderPrice
    msg += 'buy'
  elif trend=='down':
    points = nPreviousOrderPrice - ltp
    msg += 'buy'
  
  msg += f' trade here at FUT:{ltp}, INDEX:{indexPrice} with points = {points}'
  print(msg)
  nPointsCaptured = nPointsCaptured + points
  bCloseDay = True

def processLiveSupertrend(day, df, futLTP):
  global nPointsCaptured, resultdf, initial, currentTrend
  nPointsCaptured = 0
  bIsWaitPrinted = False
  row = df.tail(1)
  trend = row['STX'].values[0]
  if(trend=='nan'):
    if(not bIsWaitPrinted):
      print('Waiting for clear direction')
      bIsWaitPrinted = True
      return
  dt = row['date'].values[0]
  ltp = futLTP
  if initial:
    currentTrend = trend
    print(f'{dt} INIT Current trend: {trend}')
    initial = False
    placeOrder(dt,ltp, trend)
  else:
    if currentTrend != trend:
      currentTrend = trend
      print(f'{dt} Trend changed to : {trend} CMP: {str(ltp)}')
      placeOrder(dt,ltp, trend)
  
  candleTime = datetime.strptime(dt,'%Y-%m-%d %I:%M:%S %p').time()    
  closingTime = datetime.strptime("15:18:00",'%H:%M:%S').time()
  # print(candleTime, ", " , closingTime, candleTime == closingTime)
  if(candleTime == closingTime):
    # print('closing day')
    closeTheDay(trend,row, futLTP)
    resultdf = resultdf.append({ 'day': day, 'points': nPointsCaptured },ignore_index=True)

def refactorDataWithSupertrend(df):
  df = df.drop(['volume'], axis=1)  
  # supertrend = ta.supertrend(df['high'], df['low'],df['close'],7,2.5)['SUPERT_7_2.5']
  df = Supertrend_kite(df)
  # supertrend = Supertrend_medium(df)
  # df = df.join(supertrend)
  # df = df.iloc[7:]
  # df.rename(columns={'SUPERT_7_2.5':'trend'}, inplace=True)  
  return df
  
  # for i,row in df.iterrows():
  #   if(pd.isna(row['trend']) or row['trend']==0.00):
  #     # df.at[i,'signal'] = 'nan'
  #     df.at[i,'trend'] = 0.00
  #   # elif(row['close'] > row['trend']):
  #   #   df.at[i,'signal'] = 'up'
  #   # else:
  #   #   df.at[i,'signal'] = 'down'
  # return df

def placeOrder(dt,ltp, trend, closeTheDay = False):
  global orderPlaced, nPreviousOrderPrice, nPointsCaptured
  # ltp = round(ltp)

  if trend=='up':
    if(orderPlaced):
      points = nPreviousOrderPrice - ltp
      print(dt, " Closing previous sell trade and Placing buy order at ", ltp, ' with points = ', points)
      nPointsCaptured = nPointsCaptured + points
      nPreviousOrderPrice = ltp
    else: 
      print(dt, " Placing buy order at ", ltp)
      nPreviousOrderPrice = ltp
  elif trend=='down':
    if(orderPlaced):
      points = ltp - nPreviousOrderPrice
      print(dt, " Closing previous buy trade and Placing sell order at ", ltp, ' with points = ', points)
      nPointsCaptured = nPointsCaptured + points
      nPreviousOrderPrice = ltp
    else:
      print(dt, " Placing sell order at ", ltp)
      nPreviousOrderPrice = ltp
  orderPlaced = True
  print('points till now', nPointsCaptured)

def getLiveData():
  print('Refreshing every ', str(timeframe), ' minute(s)')
  while True:
  # currentTime = getTime()
  # clear_output()
    df = get_data()
    processLiveSupertrend(df)
  # print(currentTime, ": up" if supertrend else " : down")
  df = df.join(supertrend)
  # df2 = df.iloc[[0, -1]]
  print(df)

  time.sleep(timeframe*60)

def EMA(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()

    df[target].fillna(0, inplace=True)
    return df

def ATR(df, period, ohlc=['open', 'high', 'low', 'close']):
    """
    Function to compute Average True Range (ATR)
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'TR', atr, period, alpha=True)

    return df

def Supertrend_kite(df, period = 7, multiplier=2.5, ohlc=['open', 'high', 'low', 'close']):
    """
    Function to compute SuperTrend
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR), ATR (ATR_$period)
            SuperTrend (ST_$period_$multiplier)
            SuperTrend Direction (STX_$period_$multiplier)
    """

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST' #+ str(period) + '_' + str(multiplier)
    stx = 'STX' #  + str(period) + '_' + str(multiplier)

    """
    SuperTrend Algorithm :
        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR
        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """

    # Compute basic upper and lower bands
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
        df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
        df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
            i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
                                             df['final_lb'].iat[i] else 0.00

        # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)
    return df

def main():
  global access_token, fyers
  if(access_token):
    print('we already logged in')
  else:
    access_token = login()
    fyers = fyersModel.FyersModel(token=access_token, is_async=False, log_path="/", client_id=app_id)

if __name__ == "__main__":
    main()
    
# from google.colab import auth
# auth.authenticate_user()

# import gspread
# from google.auth import default
# creds, _ = default()

# gc = gspread.authorize(creds)

# worksheet = gc.open('MyTest').sheet1
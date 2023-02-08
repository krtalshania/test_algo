import requests
import pytz
from fyers_api import fyersModel, accessToken
from fyers_api.Websocket import ws
import pandas as pd
import numpy as np
import math
import time, datetime
from urllib import parse
import json
import os
import calendar

tz_NY  = pytz.timezone('Asia/Kolkata')

secret_key="58UDT6B4XZ"
app_id = "GCN4EESH58-100"
redirect_uri="http://127.0.0.1"
response_type="code"
grant_type = "authorization_code"
access_token = '';

atr_period = 10
atr_multiplier = 3
timeframe = 1
currentTrend = ''
initial = True
orderPlaced = False
nPointsCaptured = 0
nPreviousOrderPrice = 0;
bLiveMode = False
start_date = datetime.date(2023, 2, 8)
end_date = datetime.date(2023, 2, 9)
bCloseDay = False
bDetailedLog = True

square_off_time = datetime.time(15,18)
trade_start_time = datetime.time(9,18)

if bLiveMode:
  start_date = datetime.date.today()
  end_date = datetime.date.today()

resultdf = pd.DataFrame(index=[0])

currentFuture = "MCX:NATURALGAS23FEBFUT"

pd.set_option("display.max_rows", None)


def t(msg):
  bot_token = '5124822628:AAExc_63O4Euolz7CRJyGSlOwQc6kP8-jh8'
  bot_chatID = '499083936'
  link = 'https://api.telegram.org/bot'+bot_token+'/sendMessage?chat_id='+bot_chatID+'&parse_mode=MarkdownV2&text=' + msg
  return requests.get(link).json()

def login():
  print('Logging in')

  session=accessToken.SessionModel(client_id=app_id,secret_key=secret_key,redirect_uri=redirect_uri, response_type=response_type, grant_type=grant_type)

  if(os.path.exists('sample.txt') and os.path.getsize('sample.txt')>0):
    f = open("sample.txt", 'r')
    txt = json.loads(f.read())
    auth_code = txt['auth_code']
    session.set_token(auth_code)
    response = txt['token']
    f.close()
    return response;
  else:
    f = open("sample.txt", "w")
    response = session.generate_authcode()
    print(response)
    
    auth_code = parse.parse_qs(parse.urlsplit(input("Enter auth code: ")).query)['auth_code'][0]
    session.set_token(auth_code)
    response = session.generate_token()
    
    f.write(json.dumps({'auth_code': auth_code, 'token': response['access_token']}))
    f.close()

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
  # print('get data for', range_from, range_to, symbol)
  data = {"symbol":symbol,"resolution":timeframe,"date_format":"1","range_from":range_from,"range_to":range_to,"cont_flag":"1"}
  # print(fyers.history(data))
  ss = fyers.history(data)['candles']
  # print(ss)
  df = pd.DataFrame(data=ss,columns=['date', 'open','high','low','close','volume'])
  df['date'] =[ datetime.datetime.fromtimestamp(float(x),tz_NY).strftime('%Y-%m-%d %I:%M:%S %p') for x in df['date']]
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
      yield start_date + datetime.timedelta(n)

def getHistoricalData(range_from, range_to, symbol):
  # d = (range_from - datetime.timedelta(days=1))#.strftime('%Y-%m-%d')
  # print('122',d)
  d = range_from.strftime('%Y-%m-%d')
  last_data = get_data(d, d, symbol).tail(10)
  if(last_data.empty):
    prev_day = (d - datetime.timedelta(days=1))#.strftime('%Y-%m-%d') #datetime.datetime.strptime(d,"%Y-%m-%d")
    last_data = getHistoricalData(prev_day, range_to, symbol)
      
  df = pd.concat([last_data,get_data(range_from, range_to, symbol)],ignore_index=True)
  df.reset_index(drop=True, inplace=True)
  if not df.empty:
    return Supertrend_kite(df.drop(['volume'], axis=1))

def processHistoricalSupertrend(day, df):
  global nPointsCaptured, resultdf, square_off_time
  nPointsCaptured = 0
  initial = True
  bIsWaitPrinted = False
  for index, row in df.iterrows():
    trend = row['STX']

    if(trend=='nan'):
      if(not bIsWaitPrinted):
        if(bDetailedLog):
          print('Waiting for clear direction')
        bIsWaitPrinted = True
      continue
    dt = row['date']
    ltp = row['fut_ltp']
    if initial:
      currentTrend = trend
      if(bDetailedLog):
        print(f'{dt} INIT Current trend: {trend}, FUT: {ltp}')
      initial = False
      placeOrder(dt,ltp, trend)
    else:
      if currentTrend != trend:
        currentTrend = trend
        if(bDetailedLog):
          print(f'{dt} Trend changed to : {trend}, FUT: {ltp}')
        placeOrder(dt,ltp, trend)
    
    candleTime = datetime.datetime.strptime(dt,'%Y-%m-%d %I:%M:%S %p').time()    

    if(candleTime >= square_off_time):
      # print('closing day')
      closeTheDay(trend,row, ltp)
      break
  print(day, 'Total points captured : ', round(nPointsCaptured,2))
  resultdf = pd.concat([resultdf,pd.DataFrame({ 'day': day, 'points': nPointsCaptured }, index=[0])], ignore_index=True)

def startHistorical():
  for single_date in daterange(start_date, end_date):
    if(single_date.weekday() in [5,6]):
      continue    
    
    # print('getting fut data for', single_date, currentFuture)
    futData = getHistoricalData(single_date, single_date, currentFuture)
    # print(futData)
    futData.rename(columns={'open':'fut_ltp'}, inplace=True)    
    
    processHistoricalSupertrend(single_date, futData)
  # resultdf = resultdf.iloc[1:]
  # display(resultdf)

def startLive():
  global resultdf, start_date, end_date, timeframe, bCloseDay, nPreviousOrderPrice, currentTrend, tz_NY, index_symbol, currentFuture, worksheet, atr_period
  while True:
    now = datetime.datetime.now(tz_NY).time()
    if now < trade_start_time:
      print("Waiting for 09:18 AM, current time: ", now.strftime("%H:%M:%S %p"))
      time.sleep(60)
    else: 
      break

  print(f'Refreshing every {timeframe} minutes')
  while not bCloseDay:
    data = None
    data = getHistoricalData(start_date, end_date, fut_symbol if bDirectFut else index_symbol).tail(atr_period)
    print(data)
    # display(data)
    # worksheet.update([data.columns.values.tolist()] + data.values.tolist())
    futLTP = getLTP(currentFuture)
    currentTrendString = 'sell' if currentTrend=='down' else 'buy'
    if orderPlaced:
      points = futLTP - nPreviousOrderPrice if currentTrend=='up' else nPreviousOrderPrice - futLTP
      currentTime = datetime.datetime.now(tz_NY).strftime("%I:%M:%S %p")
      print(f'{currentTime} Current position: {currentTrendString} at {nPreviousOrderPrice} and FUTURE LTP : {futLTP}, points = {points}')
    processLiveSupertrend(start_date, data, futLTP)
    time.sleep(timeframe*60)
    
  # resultdf = resultdf.iloc[1:]
  # print(resultdf)

def processLiveSupertrend(day, df, futLTP):
  global nPointsCaptured, resultdf, initial, currentTrend, square_off_time
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
        # print(type(dt))
    candleTime = datetime.datetime.strptime(dt,'%Y-%m-%d %H:%M:%S %p').time()
    # print('candleTime',candleTime, square_off_time, candleTime >= square_off_time)
    # return
    if(candleTime >= square_off_time):
        # print('closing day')
        closeTheDay(trend,row, futLTP)
        resultdf = pd.concat([resultdf, pd.DataFrame({ 'day': day, 'points': nPointsCaptured })],ignore_index=True)

def closeTheDay(trend,row, futLTP):
  global nPointsCaptured, resultdf, nPointsCaptured, bClose
  trend = row['STX'] if type(row['STX']) == str else row['STX'].values[0]
  indexPrice = row['close'] if type(row['close']) == float else row['close'].values[0]
  ltp = futLTP
  points = 0
  msg = 'Day end, Closing '
  if trend=='up':
    points = round(ltp - nPreviousOrderPrice, 2)
    msg += 'buy'
  elif trend=='down':
    points = round(nPreviousOrderPrice - ltp, 2)
    msg += 'buy'
  
  msg += f' trade here at FUT:{ltp}, INDEX:{indexPrice} with points = {points}'
  if(bDetailedLog):
    print(msg)
  nPointsCaptured = round(nPointsCaptured + points, 2)
  bCloseDay = True

def placeOrder(dt,ltp, trend, closeTheDay = False):
  global orderPlaced, nPreviousOrderPrice, nPointsCaptured, currentFuture, fyers, bLiveMode

  if trend=='up':
    if(orderPlaced):
      points = round(nPreviousOrderPrice - ltp, 2)
      if(bDetailedLog):
        print(dt, " Closing previous sell trade and Placing buy order at ", ltp, ' with points = ', points)
      data = {
          "symbol":currentFuture,
          "qty":25,
          "type":2, #  1 = limit, 2 = market
          "side":1, #  1 = buy, -1 = sell
          "productType":"INTRADAY",
          "limitPrice":0,
          "stopPrice":0,
          "disclosedQty":0,
          "validity":"DAY",
          "offlineOrder":"False",
          "stopLoss":0
      }
      if bLiveMode:
        fyers.exit_positions({})
        res = fyers.place_order(data)
        print(res)
      nPointsCaptured = nPointsCaptured + points
      nPreviousOrderPrice = ltp
    else:
      if(bDetailedLog):
        print(dt, " Placing buy order at ", ltp)
      data = {
          "symbol":currentFuture,
          "qty":25,
          "type":2, #  1 = limit, 2 = market
          "side":1, #  1 = buy, -1 = sell
          "productType":"INTRADAY",
          "limitPrice":0,
          "stopPrice":0,
          "disclosedQty":0,
          "validity":"DAY",
          "offlineOrder":"False",
          "stopLoss":0
      }
      if bLiveMode:
        res = fyers.place_order(data)
        print(res)
      nPreviousOrderPrice = ltp
  elif trend=='down':
    if(orderPlaced):
      points = round(ltp - nPreviousOrderPrice, 2)
      if(bDetailedLog):
        print(dt, " Closing previous buy trade and Placing sell order at ", ltp, ' with points = ', points)
      res = fyers.exit_positions({})
      data = {
          "symbol":currentFuture,
          "qty":25,
          "type":2, #  1 = limit, 2 = market
          "side":-1, #  1 = buy, -1 = sell
          "productType":"INTRADAY",
          "limitPrice":0,
          "stopPrice":0,
          "disclosedQty":0,
          "validity":"DAY",
          "offlineOrder":"False",
          "stopLoss":0
      }
      if bLiveMode:
        res = fyers.place_order(data)
        print(res)
      nPointsCaptured = nPointsCaptured + points
      nPreviousOrderPrice = ltp
    else:
      if(bDetailedLog):
        print(dt, " Placing sell order at ", ltp)
      data = {
          "symbol":currentFuture,
          "qty":25,
          "type":2, #  1 = limit, 2 = market
          "side":-1, #  1 = buy, -1 = sell
          "productType":"INTRADAY",
          "limitPrice":0,
          "stopPrice":0,
          "disclosedQty":0,
          "validity":"DAY",
          "offlineOrder":"False",
          "stopLoss":0
      }
      if bLiveMode:
        res = fyers.place_order(data)
        print(res)
      nPreviousOrderPrice = ltp
  orderPlaced = True
  if(bDetailedLog):
    print('points till now', round(nPointsCaptured,2))

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

def Supertrend_kite(df, period = 10, multiplier=3, ohlc=['open', 'high', 'low', 'close']):
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
    global access_token, fyers, bLiveMode, logpath
    access_token = login()
    fyers = fyersModel.FyersModel(token=access_token, is_async=False, log_path="logs", client_id=app_id)
    if bLiveMode:
      startLive()
    else:
      startHistorical()
        
if __name__ == "__main__":
    main()

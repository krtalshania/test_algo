import pytz
from fyers_api import fyersModel, accessToken
from fyers_api.Websocket import ws
import pandas as pd
import time, datetime
from urllib import parse
import json
import os
import my_modules

tz_NY  = pytz.timezone('Asia/Kolkata')

secret_key="58UDT6B4XZ"
app_id = "GCN4EESH58-100"
redirect_uri="http://127.0.0.1"
response_type="code"
grant_type = "authorization_code"
fyers = None
atr_period = 7
atr_multiplier = 2.5
timeframe = 1
currentTrend = ''
initial = True
orderPlaced = False
nPointsCaptured = 0
nPreviousOrderPrice = 0;
bLiveMode = True
bPlaceOrderInLiveMode = False
start_date = datetime.date(2023, 3, 20)
end_date = datetime.date(2023, 3, 21)
bCloseDay = False
bDetailedLog = True
index_symbol = "NSE:NIFTYBANK-INDEX"

square_off_time = datetime.time(15,18)
trade_start_time = datetime.time(9,18)

if bLiveMode:
  start_date = datetime.date.today()
  end_date = datetime.date.today()

resultdf = pd.DataFrame(index=[0])

currentFuture = f"NSE:BANKNIFTY23MARFUT"
pd.set_option("display.max_rows", None)

def login():
  global fyers, app_id, secret_key, redirect_uri, response_type, grant_type
  print('Logging in')

  session=accessToken.SessionModel(client_id=app_id,secret_key=secret_key,redirect_uri=redirect_uri, response_type=response_type, grant_type=grant_type)

  if(os.path.exists('sample.txt') and os.path.getsize('sample.txt')>0):
    f = open("sample.txt", 'r')
    txt = json.loads(f.read())
    auth_code = txt['auth_code']
    session.set_token(auth_code)
    access_token = txt['access_token']
    fyers = fyersModel.FyersModel(token=access_token, is_async=False, log_path="logs", client_id=app_id)
    f.close()
  else:
    f = open("sample.txt", "w")
    response = session.generate_authcode()
    print(response)
    
    auth_code = parse.parse_qs(parse.urlsplit(input("Enter auth code: ")).query)['auth_code'][0]
    session.set_token(auth_code)
    response = session.generate_token()
    
    if(response["s"]=="ok"):
      # print(response)
      access_token = response['access_token']
      f.write(json.dumps({'auth_code': auth_code, 'access_token': access_token}))
      f.close()
      fyers = fyersModel.FyersModel(token=access_token, is_async=False, log_path="logs", client_id=app_id)

def getHistoricalData(range_from, range_to, symbol):
  global fyers, timeframe, atr_period
  d = range_from.strftime('%Y-%m-%d')
  last_data = my_modules.get_data(fyers, timeframe, d, d, symbol).tail(atr_period)
  if(last_data.empty):
    prev_day = (d - datetime.timedelta(days=1))#.strftime('%Y-%m-%d') #datetime.datetime.strptime(d,"%Y-%m-%d")
    last_data = getHistoricalData(prev_day, range_to, symbol)
      
  df = pd.concat([last_data,my_modules.get_data(fyers, timeframe, range_from, range_to, symbol)],ignore_index=True)
  df.reset_index(drop=True, inplace=True)
  if not df.empty:
    return my_modules.Supertrend_kite(df.drop(['volume'], axis=1), atr_period, atr_multiplier)

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
    close = row['close']
    if initial:
      currentTrend = trend
      if(bDetailedLog):
        print(f'{dt} INIT Current trend: {trend}, INDEX: {close}, FUT: {ltp}')
      initial = False
      placeOrder(dt,ltp, trend)
    else:
      if currentTrend != trend:
        currentTrend = trend
        if(bDetailedLog):
          print(f'{dt} Trend changed to : {trend}, INDEX: {close}, FUT: {ltp}')
        placeOrder(dt,ltp, trend)
    
    candleTime = datetime.datetime.strptime(dt,'%Y-%m-%d %I:%M:%S %p').time()    

    if(candleTime >= square_off_time):
      # print('closing day')
      closeTheDay(trend,row, ltp)
      break
  print(day, 'Total points captured : ', nPointsCaptured)
  resultdf = pd.concat([resultdf,pd.DataFrame({ 'day': day, 'points': nPointsCaptured }, index=[0])], ignore_index=True)

def startHistorical():
  global resultdf, start_date, end_date, timeframe, index_symbol, atr_period
  for single_date in my_modules.daterange(start_date, end_date):
    if(single_date.weekday() in [5,6]):
      continue
    histdata = getHistoricalData(single_date, single_date, index_symbol)
    # display(histdata)
    # return
    if(not histdata.empty):
      print('getting fut data for', single_date, currentFuture)
      futData = getHistoricalData(single_date, single_date, currentFuture)
      futData = futData.drop(['high','low','close','STX', 'TR','ATR_'+str(atr_period), 'ST','date'], axis=1)
      futData.rename(columns={'open':'fut_ltp'}, inplace=True)
      histdata = histdata.join(futData)
      # worksheet.update([histdata.columns.values.tolist()] + histdata.values.tolist())
      # display(histdata)
      # return
      processHistoricalSupertrend(single_date, histdata)
  # resultdf = resultdf.iloc[1:]
  # display(resultdf)

def startLive():
  global fyers,resultdf, start_date, end_date, timeframe, bCloseDay, nPreviousOrderPrice, currentTrend, tz_NY, index_symbol, currentFuture, worksheet, atr_period
  while True:
    now = datetime.datetime.now(tz_NY).time()
    if now < trade_start_time:
      print("Waiting for 09:18 AM, current time: ", now.strftime("%H:%M:%S %p"))
      time.sleep(60)
    else: 
      break

  print(f'Refreshing every {timeframe} minutes')
  while not bCloseDay:
    data = getHistoricalData(start_date, end_date, index_symbol).tail(atr_period)
    # display(data)
    # worksheet.update([data.columns.values.tolist()] + data.values.tolist())
    futLTP = my_modules.getLTP(fyers,currentFuture)
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
    points = round(ltp - nPreviousOrderPrice)
    msg += 'buy'
  elif trend=='down':
    points = round(nPreviousOrderPrice - ltp)
    msg += 'buy'
  
  msg += f' trade here at FUT:{ltp}, INDEX:{indexPrice} with points = {points}'
  if(bDetailedLog):
    print(msg)
  nPointsCaptured = round(nPointsCaptured + points)
  bCloseDay = True

def placeOrder(dt,ltp, trend, closeTheDay = False):
  global orderPlaced, nPreviousOrderPrice, nPointsCaptured, currentFuture, fyers, bLiveMode
  # ltp = round(ltp)

  if trend=='up':
    if(orderPlaced):
      points = round(nPreviousOrderPrice - ltp)
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
      if bLiveMode and bPlaceOrderInLiveMode:
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
      if bLiveMode and bPlaceOrderInLiveMode:
        res = fyers.place_order(data)
        print(res)
      nPreviousOrderPrice = ltp
  elif trend=='down':
    if(orderPlaced):
      points = round(ltp - nPreviousOrderPrice)
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
      if bLiveMode and bPlaceOrderInLiveMode:
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
      if bLiveMode and bPlaceOrderInLiveMode:
        res = fyers.place_order(data)
        print(res)
      nPreviousOrderPrice = ltp
  orderPlaced = True
  if(bDetailedLog):
    print('points till now', nPointsCaptured)


def main():
  global bLiveMode
  login()
  if bLiveMode:
    startLive()
  else:
    startHistorical()
        
if __name__ == "__main__":
  main()

import datetime
import pandas as pd
import pytz
import numpy as np
import requests
tz_NY  = pytz.timezone('Asia/Kolkata')


def t(msg):
  bot_token = '5124822628:AAExc_63O4Euolz7CRJyGSlOwQc6kP8-jh8'
  bot_chatID = '499083936'
  link = 'https://api.telegram.org/bot'+bot_token+'/sendMessage?chat_id='+bot_chatID+'&parse_mode=MarkdownV2&text=' + msg
  return requests.get(link).json()

def getTime():
  return datetime.now(tz_NY).strftime("%Y-%m-%d %I:%M:%S %p")

def get_data(fyers, timeframe, range_from, range_to, symbol):
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

def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
      yield start_date + datetime.timedelta(n)

def getLTP(fyers, symbol):
  q = fyers.quotes({"symbols":symbol})['d'][0]['v']['lp']
  return q

def EMA(df, base, target, period, alpha=False):
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

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST' #+ str(period) + '_' + str(multiplier)
    stx = 'STX' #  + str(period) + '_' + str(multiplier)

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

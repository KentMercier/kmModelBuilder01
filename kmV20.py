# kmV20 is modified from TPQOA found at https://github.com/yhilpisch/tpqoa/blob/master/tpqoa/tpqoa.py
# and from https://colab.research.google.com/drive/1yt4aUMfV1vK0oSBSeODOr87FMuxVRJ8e?usp=sharing of which Kent has a modified colab in drive.

# todo: at some point study the thread, signal and threading code.

import _thread
# import configparser
import json
import signal
import threading
from time import sleep
import requests

import numpy as np
import pandas as pd
import v20
from v20.transaction import StopLossDetails, ClientExtensions
from v20.transaction import TrailingStopLossDetails, TakeProfitDetails
from datetime import datetime
from datetime import timedelta
import time
import asyncio
import json
from pathlib import Path

# variables is taking the place of accountConfigs.cfg
import variables as var
import kmTradingBot as kmTB

MAX_REQUEST_COUNT = float(5000)


class Job(threading.Thread):
    def __init__(self, job_callable, args=None):
        threading.Thread.__init__(self)
        self.callable = job_callable
        self.args = args

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()
        self.job = None
        self.exception = None

    def run(self):
        print('Thread #%s started' % self.ident)
        try:
            self.job = self.callable
            while not self.shutdown_flag.is_set():
                print("Starting job loop...")
                if self.args is None:
                    self.job()
                else:
                    self.job(self.args)
        except Exception as e:
            import sys
            import traceback
            print(traceback.format_exc())
            self.exception = e
            _thread.interrupt_main()


class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """

    def __init__(self, message=None):
        self.message = message

    def __repr__(self):
        return repr(self.message)


def service_shutdown(signum, frame):
    print('exiting ...')
    raise ServiceExit


# what is the object being passed into kmV20?
class KMv20:
    ''' tpqoa is a Python wrapper class for the Oanda v20 API. changing this to customized version of V20 in kmV20'''

    #     def __init__(self, conf_file):
    def __init__(self):

        # default leverage
        # todo not sure yet why we need to multiply times 100 and again by 2 to get 2/3 of balance placed on margin; or if we need to do this.
        self.leverage = 30 *100 * 2

        self.access_token = var.access_token
        self.account_id = var.account_id
        self.account_type = var.account_type

        # for requests:

        self.headers = {'Authorization': 'Bearer ' + self.access_token}
        self.apiAddress = ""
        self.hostname = ""
        self.stream_hostname = ""
        # self.apiAddress is for use with requests.  self.hostname is for use with context.

        if self.account_type == 'live':
            self.apiAddress = 'https://api-fxtrade.oanda.com'
            self.hostname = 'api-fxtrade.oanda.com'
            self.stream_hostname = 'stream-fxtrade.oanda.com'
        else:
            self.apiAddress = 'https://api-fxpractice.oanda.com'
            self.hostname = 'api-fxpractice.oanda.com'
            self.stream_hostname = 'stream-fxpractice.oanda.com'

        self.ctx = v20.Context(
            hostname=self.hostname,
            port=443,
            token=self.access_token,
            poll_timeout=10
        )
        self.ctx_stream = v20.Context(
            hostname=self.stream_hostname,
            port=443,
            token=self.access_token,
        )
        # The self.suffix is for the time suffix.
        self.suffix = '.000000000Z'

        self.stop_stream = False

        self.data = pd.DataFrame()
        self.workingData = pd.DataFrame()

        self.strategyFeatures = []
        # not sure if this is normalization and scaling???????????
        # self.mu = 0.0
        self.mu, self.std = self.getStats()
        #print('self.mu', self.mu)
        #print('self.std', self.std)
                #f = open('C:\\Users\Administrator\Documents\MostRecent\persistedMuStd.json')
        filepath = Path(__file__).resolve().parent
        minFile = 'min.csv'
        mincsv=Path.joinpath(filepath, minFile)
        
        self.minim = pd.read_csv(mincsv)
        #print('self.minim', self.minim)
        #self.minim = self.minim.transpose()
        #print('self.minim', self.minim)
        #self.minim.columns = self.minim.iloc[0]
        #self.minim = self.minim.transpose()
        #self.minim = self.minim.iloc[1: , :]
        #print('self.minim dropped Unnamed row: ')
        #print(self.minim)
        #print('self.minim with new column headers: ')
        #print(self.minim)
        #for columb in self.minim.columns:
        #    print('columb: ', columb)
        
        maxFile = 'max.csv'
        maxcsv=Path.joinpath(filepath, maxFile)
        self.maxim = pd.read_csv(maxcsv)
        # self.maxim = pd.read_csv(r'C:\Users\Administrator\Documents\MostRecent\max.csv')
        #print('self.maxim', self.maxim)
        #self.maxim = self.maxim.transpose()
        #print('self.maxim', self.maxim)


        stdFile = 'std.csv'
        stdcsv=Path.joinpath(filepath, stdFile)
        self.stand = pd.read_csv(stdcsv)
        #print('self.stand', self.stand)
        #self.stand = self.stand.transpose()
        #print('self.stand', self.stand)

        self.units = 0.0
        self.lags = 5
        
        
        
        
        # self.dfFromCSV = pd.read_csv('addFeatures_USD_JPY_2021-01-01_to_2021-11-04_M1_A.csv')

    def getStats(self):
        '''
        path note 1:  on aws need the double \\ before the start of the file names to account for the _ in the name.
        path note 2:  on aws need to use absolute addresses.
        e.g, f = open('C:\\Users\Administrator\Documents\MostRecent#\persistedMuStd.json')

        '''
      
        #f = open('C:\\Users\Administrator\Documents\MostRecent\persistedMuStd.json')
        filepath = Path(__file__).resolve().parent
        statsFile = 'persistedMuStd.json'
        p=Path.joinpath(filepath, statsFile)
        
        f = open(p)

        z = json.load(f)
        mu = z['mu']
        std = z['std']
        f.close()
        
        #self.min = pd.read_csv(r'C:\Users\Administrator\Documents\MostRecent\min.csv')
        #self.max = pd.read_csv(r'C:\Users\Administrator\Documents\MostRecent\max.csv')
        #self.stand = pd.read_csv(r'C:\Users\Administrator\Documents\MostRecent\std.csv')
        
        
        
        return mu, std

        
    def get_instruments(self):
        ''' Retrieves and returns all instruments for the given account. '''
        resp = self.ctx.account.instruments(self.account_id)
        instruments = resp.get('instruments')
        instruments = [ins.dict() for ins in instruments]
        instruments = [(ins['displayName'], ins['name'])
                       for ins in instruments]
        return sorted(instruments)

    def get_prices(self, instrument):
        ''' Returns the current BID/ASK prices for instrument. '''
        r = self.ctx.pricing.get(self.account_id, instruments=instrument)
        r = json.loads(r.raw_body)
        bid = float(r['prices'][0]['closeoutBid'])
        ask = float(r['prices'][0]['closeoutAsk'])
        return r['time'], bid, ask

    def transform_datetime(self, dati):
        ''' Transforms Python datetime object to string. '''
        if isinstance(dati, str):
            dati = pd.Timestamp(dati).to_pydatetime()
        # return dati.isoformat('T') + self.suffix
        return dati.isoformat('T')

    def retrieve_data(self, instrument, start, end, granularity, price):
        raw = self.ctx.instrument.candles(
            instrument=instrument,
            fromTime=start, toTime=end,
            granularity=granularity, price=price)
        raw = raw.get('candles')
        raw = [cs.dict() for cs in raw]
        if price == 'A':
            for cs in raw:
                cs.update(cs['ask'])
                del cs['ask']
        elif price == 'B':
            for cs in raw:
                cs.update(cs['bid'])
                del cs['bid']
        elif price == 'M':
            for cs in raw:
                cs.update(cs['mid'])
                del cs['mid']
        else:
            raise ValueError("price must be either 'B', 'A' or 'M'.")
        if len(raw) == 0:
            return pd.DataFrame()  # return empty DataFrame if no data
        data = pd.DataFrame(raw)
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data.index = pd.DatetimeIndex(data.index)
        for col in list('ohlc'):
            data[col] = data[col].astype(float)
        return data

    def get_history(self, instrument, start, end,
                    granularity, price, localize=True):
        ''' Retrieves historical data for instrument.
        Parameters
        ==========
        instrument: string
            valid instrument name
        start, end: datetime, str
            Python datetime or string objects for start and end
        granularity: string
            a string like 'S5', 'M1' or 'D'
        price: string
            one of 'A' (ask), 'B' (bid) or 'M' (middle)
        Returns
        =======
        data: pd.DataFrame
            pandas DataFrame object with data
        '''
        if granularity.startswith('S') or granularity.startswith('M') \
                or granularity.startswith('H'):
            multiplier = float("".join(filter(str.isdigit, granularity)))
            if granularity.startswith('S'):
                # freq = '1h'
                freq = f"{int(MAX_REQUEST_COUNT * multiplier / float(3600))}H"
            else:
                # freq = 'D'
                freq = f"{int(MAX_REQUEST_COUNT * multiplier / float(1440))}D"
            data = pd.DataFrame()
            dr = pd.date_range(start, end, freq=freq)

            for t in range(len(dr)):
                batch_start = self.transform_datetime(dr[t])
                if t != len(dr) - 1:
                    batch_end = self.transform_datetime(dr[t + 1])
                else:
                    batch_end = self.transform_datetime(end)

                batch = self.retrieve_data(instrument, batch_start, batch_end,
                                           granularity, price)
                data = data.append(batch)
        else:
            start = self.transform_datetime(start)
            end = self.transform_datetime(end)
            data = self.retrieve_data(instrument, start, end,
                                      granularity, price)
        if localize:
            data.index = data.index.tz_localize(None)

        return data[['o', 'h', 'l', 'c', 'volume', 'complete']]

    def create_order(self, instrument='USD_JPY', units=0, price=None, sl_distance=None,
                     tsl_distance=None, tp_price=None, comment=None,
                     touch=False, suppress=False, ret=False):
        ''' Places order with Oanda.
        Parameters
        ==========
        instrument: string
            valid instrument name
        units: int
            number of units of instrument to be bought
            (positive int, eg 'units=50')
            or to be sold (negative int, eg 'units=-100')
        price: float
            limit order price, touch order price
        sl_distance: float
            stop loss distance price, mandatory eg in Germany
        tsl_distance: float
            trailing stop loss distance
        tp_price: float
            take profit price to be used for the trade
        comment: str
            string
        touch: boolean
            market_if_touched order (requires price to be set)
        suppress: boolean
            whether to suppress print out
        ret: boolean
            whether to return the order object
        '''
        client_ext = ClientExtensions(
            comment=comment) if comment is not None else None
        sl_details = (StopLossDetails(distance=sl_distance,
                                      clientExtensions=client_ext)
                      if sl_distance is not None else None)
        tsl_details = (TrailingStopLossDetails(distance=tsl_distance,
                                               clientExtensions=client_ext)
                       if tsl_distance is not None else None)
        tp_details = (TakeProfitDetails(
            price=tp_price, clientExtensions=client_ext)
                      if tp_price is not None else None)
        if price is None:
            request = self.ctx.order.market(
                self.account_id,
                instrument=instrument,
                units=units,
                stopLossOnFill=sl_details,
                trailingStopLossOnFill=tsl_details,
                takeProfitOnFill=tp_details,
            )
        elif touch:
            request = self.ctx.order.market_if_touched(
                self.account_id,
                instrument=instrument,
                price=price,
                units=units,
                stopLossOnFill=sl_details,
                trailingStopLossOnFill=tsl_details,
                takeProfitOnFill=tp_details
            )
        else:
            request = self.ctx.order.limit(
                self.account_id,
                instrument=instrument,
                price=price,
                units=units,
                stopLossOnFill=sl_details,
                trailingStopLossOnFill=tsl_details,
                takeProfitOnFill=tp_details
            )

        # First checking if the order is rejected
        if 'orderRejectTransaction' in request.body:
            order = request.get('orderRejectTransaction')
        elif 'orderFillTransaction' in request.body:
            order = request.get('orderFillTransaction')
        elif 'orderCreateTransaction' in request.body:
            order = request.get('orderCreateTransaction')
        else:
            # This case does not happen.  But keeping this for completeness.
            order = None

        if not suppress and order is not None:
            print('\n\n', order.dict(), '\n')
        if ret is True:
            return order.dict() if order is not None else None

    def stream_data(self, instrument, stop=None, ret=False, callback=None):
        ''' Starts a real-time data stream.
        Parameters
        ==========
        instrument: string
            valid instrument name
        '''
        self.stream_instrument = instrument
        self.ticks = 0
        response = self.ctx_stream.pricing.stream(
            self.account_id, snapshot=True,
            instruments=instrument)
        msgs = []
        for msg_type, msg in response.parts():
            msgs.append(msg)
            # print(msg_type, msg)
            if msg_type == 'pricing.ClientPrice':
                self.ticks += 1
                self.time = msg.time
                if callback is not None:
                    callback(msg.instrument, msg.time,
                             float(msg.bids[0].dict()['price']),
                             float(msg.asks[0].dict()['price']))
                else:
                    self.on_success(msg.time,
                                    float(msg.bids[0].dict()['price']),
                                    float(msg.asks[0].dict()['price']))
                if stop is not None:
                    if self.ticks >= stop:
                        if ret:
                            return msgs
                        break
            if self.stop_stream:
                if ret:
                    return msgs
                break

    def _stream_data_failsafe_thread(self, args):
        try:
            print("Starting price streaming")
            self.stream_data(args[0], callback=args[1])
        except Exception as e:
            import sys
            import traceback
            print(traceback.format_exc())
            sleep(3)
            return

    def stream_data_failsafe(self, instrument, callback=None):
        ''' in this implementation of trading I am not streaming data.  Instead, we are pullingone historical minute at a time.'''
        ''' Method called when new data is retrieved. '''
        signal.signal(signal.SIGTERM, service_shutdown)
        signal.signal(signal.SIGINT, service_shutdown)
        signal.signal(signal.SIGSEGV, service_shutdown)
        try:
            price_stream_thread = Job(self._stream_data_failsafe_thread,
                                      [instrument, callback])
            price_stream_thread.start()
            return price_stream_thread
        except ServiceExit as e:
            print('Handling exception')
            import sys
            import traceback
            print(traceback)
            price_stream_thread.shutdown_flag.set()
            price_stream_thread.join()

    def on_success(self, time, bid, ask):
        ''' in this implementation of trading I am not streaming data.  Instead, we are pullingone historical minute at a time.'''
        ''' Method called when new data is retrieved. '''
        print(time, bid, ask)

    def get_account_summary(self, detailed=False):
        ''' Returns summary data for Oanda account.'''
        if detailed is True:
            response = self.ctx.account.get(self.account_id)
        else:
            response = self.ctx.account.summary(self.account_id)
        raw = response.get('account')
        return raw.dict()

    def get_transaction(self, tid=0):
        ''' Retrieves and returns transaction data. '''
        response = self.ctx.transaction.get(self.account_id, tid)
        transaction = response.get('transaction')
        return transaction.dict()

    def get_transactions(self, tid=0):
        ''' Retrieves and returns transactions data. '''
        response = self.ctx.transaction.since(self.account_id, id=tid)
        transactions = response.get('transactions')
        transactions = [t.dict() for t in transactions]
        return transactions

    def print_transactions(self, tid=0):
        ''' Prints basic transactions data. '''
        transactions = self.get_transactions(tid)
        for trans in transactions:
            try:
                templ = '%4s | %s | %7s | %8s | %8s'
                print(templ % (trans['id'],
                               trans['time'][:-8],
                               trans['instrument'],
                               trans['units'],
                               trans['pl']))
            except Exception:
                pass

    def get_positions(self):
        ''' Retrieves and returns positions data. '''
        response = self.ctx.position.list_open(self.account_id).body
        positions = [p.dict() for p in response.get('positions')]
        return positions

    # KM added functions

    def closeOpenPosition(self):
        print('*** CLOSING OUT OPEN POSITION***')
        print('kmTB.kmTrade.position')
        # get positions
        lastPosition = self.get_positions()
        # if exist, then transaction opposite
        print('lastPosition', lastPosition)
        # if kmTB.kmTrade.position == 0
        #     try:
        #         order = self.create_order(kmTB.symbol,
        #                                   units=-kmTB.position * kmTB.units,
        #                                   suppress=True, ret=True)
        #         kmTB.report_trade(kmTB.time, 'NEUTRAL', order)
        #         if kmTB.verbose:
        #             print(order)
        #         print(75 * '=')
        #     except:
        #         pass

    def get_latest_minute_history(self, instrument='USD_JPY', granularity='M1', price='A'):
        '''
        This will be used to retreive the minute by minute return from Oanda (which includes the volume).  The return will be submected to feature engineering, then submitted as the state to retreive the policy/action to select the order to submit.
        
        The minute candlestick returned is not the current "now" minute, but the last minute.  This way the "last minute" is a completed candlestick with final volume, etc.
        
        This may result in a loss of accuracy, I may need to do research into what is being lost, and how to a have the decision made as close in time to the completion of the candle as possible.
        
        query count:  	The number of candlesticks to return in the response. Count should not be specified if both the start and end parameters are provided, as the time range combined with the granularity will determine the number of candlesticks to return. [default=500, maximum=5000].
        Shouldn't the query count be set to 1?  I don't remember why I set it to 2.
        '''

        query = {"count": 2, 'granularity': 'M1', "price": 'A'}

        CANDLES_PATH = f"/v3/instruments/{instrument}/candles"
        #         print(self.apiAddress+CANDLES_PATH)

        response = requests.get(self.apiAddress + CANDLES_PATH, headers=self.headers, params=query)
        # print('response:  ', response)

        if response.ok:
            x = response.json()
            y = x['candles'][0]

            if len(y) == 0:
                return pd.DataFrame()  # return empty DataFrame if no data

            print('now: ', datetime.now())

            z = {
                'time': y['time'],
                'o': y['ask']['o'],
                'h': y['ask']['h'],
                'l': y['ask']['l'],
                'c': y['ask']['c'],
                'volume': y['volume']
            }
            self.data = pd.DataFrame([z])

            self.data['time'] = self.data['time'].astype('datetime64[ns]')

            self.data = self.data.set_index('time')
            for col in list('ohlc'):
                self.data[col] = self.data[col].astype(float)
            self.data = self.feStrategy1(self.data)
            self.workingData.append(self.data)
            return self.data
        else:
            return 'ERROR:  Data not retrieved'

    def get_latest_50_minute_history(self, instrument='USD_JPY', granularity='M1', price='A'):
        '''
        experimental / alternative to calling last minute row
        This will be used to retreive latest 50 rows of  minute by minute return from Oanda (which includes the volume).  The return will be submected to feature engineering, then submitted as the state to retreive the policy/action to select the order to submit.

        The minute candlestick returned is not the current "now" minute, but the last minute.  This way the "last minute" is a completed candlestick with final volume, etc.

        This may result in a loss of accuracy, I may need to do research into what is being lost, and how to a have the decision made as close in time to the completion of the candle as possible.

        query count:  	The number of candlesticks to return in the response. Count should not be specified if both the start and end parameters are provided, as the time range combined with the granularity will determine the number of candlesticks to return. [default=500, maximum=5000].
        Shouldn't the query count be set to 1?  I don't remember why I set it to 2.
        '''

        query = {"count": 50, 'granularity': 'M1', "price": 'A'}

        CANDLES_PATH = f"/v3/instruments/{instrument}/candles"
        # print(self.apiAddress + CANDLES_PATH)

        response = requests.get(self.apiAddress + CANDLES_PATH, headers=self.headers, params=query)
        # print('response:  ', response)

        if response.ok:
            x = response.json()
            # y = x['candles'][0]
            y = x['candles']

            if len(y) == 0:
                return pd.DataFrame()  # return empty DataFrame if no data

            # print('now: ', datetime.now())
            # print("len(y): ", len(y))
            # print('type of y: ', type(y))
            self.data = pd.DataFrame(y)
            #print('y dataframe: ', self.data)
            # print('self.data.tail(5): ', self.data.tail(5))
            #for col in self.data.columns:
            #    print('self.data.columns', col)
            #print('self.data.ask: ', self.data.ask[0])

           
            self.data['time'] = self.data['time'].astype('datetime64[ns]')

            self.data = self.data.set_index('time')
            self.data.index = self.data.index.tz_localize('GMT')
            self.data.drop('complete', 1, inplace = True)


            self.data = pd.concat([self.data, self.data["ask"].apply(pd.Series)], axis=1)
            # print('self.data split ask: ', self.data.tail(3))
            #for col in self.data.columns:
            #    print('cols: ', col)
            
            # ??? I forgot why I am dropping ask
            self.data = self.data.drop('ask', 1)
            # print('self.data after drop ask: ', self.data)

            # for col in list('ohlc'):
            #     self.data = self.data.astype(float)
            # print('pre feStrategy: ', self.data.tail(2))
            
            
            self.data = self.feStrategy1()
            
            # print('self.data.tail(1): ', self.data.tail(1))
            # in this version of data retreival we do not use the workingData.append.
            # instead we feature engineer each batch on a minute per minute basis.
            # this may prove to be inefficient, however the feature engineering currently requires
            # no 'd' column to calculate the feature engineering.
            # self.workingData.append(self.data)
            return self.data.tail(1)
        else:
            return 'ERROR:  Data not retrieved'

    def pushToCSVfile(self):
        # self.data should be saved at the end of the trading session
        self.data.to_csv('addFeatures_USD_JPY_2021-01-01_to_2021-11-04_M1_A.csv', mode='a', header=False)

    def pushToRaw(self):
        pass

    def feStrategy1(self):
        '''                
This is based on ModelBuilderEnv_66.py  _prepare_data
        '''

        for col in list('ohlc'):
            self.data[col] = self.data[col].astype(float)
            
        self.data['volume'] = self.data['volume'].astype(float)
        # print(lastRowsDf)
        # ??? why are we dropping this ?
        #print('pre drop self.data: ', self.data)
        #self.data.drop(self.data.iloc[:, 5:], inplace=True, axis=1)
        #print('post drop self.data: ', self.data)
        # print('truncatedColumns: ', lastRowsDf)
        # frames = [self.data, data]
        # data = pd.concat(frames)
        # print('data: ', data)
        #print('0self.data')
        #print(self.data)
        # this will add feature engineered columns of data to self.data
        if not 'd' in self.data.columns:
            ''' Method to prepare additional time series data
                (such as features data).
            '''
            window = 5
            #print('self.data.head()', self.data.head())
            # print('self.data.tail()', self.data.tail())


            # self.data['r'] = np.log(self.data['c'] / self.data['c'].shift(1))
            
            # don't use diff(), this returns the absolute value of the difference
            # self.raw['r'] = self.raw['c'].diff()
            self.data['r'] = self.data['c'].shift(1) - self.data['c']
            self.data.dropna(inplace=True)



            #  This creates a dataset that has Gaussian NORMALIZATION of the data.
            #  however, it is self.data_ is not used here, all of these data methods may need to be
            #  brought into the oadndaenv.py file.
            
            # normalization will not work with strings, so remove the time and complete column
            #self.data.drop('time', axis = 1, inplace = True)
            # self.data.drop('complete', axis = 1, inplace = True)

            #self.data = self.data.drop(['complete'], axis=1)
            # print('self.data without complete: ', self.data.tail(5))


            # todo the creation of the .mean() needs to come from the entire dataset.
                # here it is just coming from the last 50, so the mu needs to be saved to a file
            # and retreived for use here.  Also, the same for the std().
            # this is a truncated dataset and the normalized data will not match anything in the
            # in the model.
            
            # the volume is skewed very much, so we will apply np.log to the volume column before normalizing
            #self.data['volume'] = np.log(self.data['volume'])
            #print('')
            #print('volume: ', self.data['volume'])
            self.data['volume'] = np.log(self.data['volume'])
            #print('logged volume', self.data['volume'])
            
            #for x in self.data.columns:
                #print('featureName in self.data: ', x)
            #for x in self.maxim.columns:
                #print('featureName in self.maxim: ', x)    
            #for x in self.minim.columns:
                #print('featureName in self.minim: ', x)  
            #print('self.data.head(2): ')
            #print(self.data.head(2))
            #for index, row in self.data.iterrows():
            #for i in range(len(self.data)):
            #for i in self.data.index:
                #print('i: ', i)
                #for featureName in self.data.columns:
                    # the 'r' column is the reward column and does not get normalized.
                    #if featureName != 'r':
                    
                        #print('featureName: ', featureName)
                        #print('self.data[featureName][i]: ', self.data[featureName][i])
                        #print('int(self.minim[featureName]): ', int(self.minim[featureName]))
                        #print('int(self.maxim[featureName]): ', int(self.maxim[featureName]))

                        #self.data[featureName] = (self.data[featureName] - self.minim[featureName])) / (
                        #            self.maxim[featureName] - self.minim[featureName])
                        #self.data[featureName][i] = (self.data[featureName][i] - self.minim[featureName]) / (
                        #            self.maxim[featureName] - self.minim[featureName])
                        
                    #else:
                    #    self.data['r'] = self.data['r']

            self.data['sma'] = self.data['c'].rolling(window).mean()
            self.data['min'] = self.data['c'].rolling(window).min()
            self.data['max'] = self.data['c'].rolling(window).max()
            self.data['mom'] = self.data['c'].rolling(window).mean()
            # do I really want to add a std column???
            self.data['std'] = self.data['r'].rolling(window).std()
            self.data.dropna(inplace=True)
            
            
            
            # This also works, but it removes the column headers which we might want later.
            # x = df1.values #returns a numpy array
            # min_max_scaler = preprocessing.MinMaxScaler()
            # x_scaled = min_max_scaler.fit_transform(x)
            # df = pd.DataFrame(x_scaled)
            
            # don't want to normalize d column, it will be used ass the "action", so create 'd'  after normalization occurs.  Note, we may want to add a hold action later
            # here 1 = buy, 0 = sell.
            
            # ??? do we want to lag the r and d columns?
            self.features = self._add_lags(self.data)
            
            
            # the direction of the diffence in log return
            self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
            # change direction from string to integer
            self.data['d'] = self.data['d'].astype(int)
            
        
        #print('1self.data: ')
        #print(self.data)
        # populate history through lags to be added to features to for state.




            # if self.mu is None:
            #     self.mu = self.data['c'].mean()
            #     self.std = self.data.std()
            #self.data = (self.data - self.mu) / self.std
            #self.data = self.data.tail(1)
            #print('self.data.head(): ', self.data.head())
            # self.mu = None
        #print('2self.data length: ', len(self.data))
        #print(self.data)
        
        return self.data


    def _add_lags(self, data):
        ''' This is a function that is used in _prepare_data
        '''
        cols = []
        # print(data.columns)
        for x in self.data.columns:
            cols.append(x)
        #print('cols: ', cols)
        features = ['o', 'h', 'l', 'c', 'volume', 'r', 'sma', 'min', 'max',
                    'mom', 'std']
        for f in features:
            for lag in range(1, self.lags + 1):
                col = f'{f}_lag_{lag}'
                self.data[col] = self.data[f].shift(lag)
                cols.append(col)
        self.data.dropna(inplace=True)
        #         print('after Adding lags: ', self.data.head(1))
        self.features = cols
        self.features.append('d')
        return cols       
        
        
        








    def feStrategy(self, data):
        '''
        # to add the features for the current state to go into the policy, we need to get the last row in the csv
        database, perform the calculations to create the complete new minute row.
            # print('self.dfFromCSV')
            # print(self.dfFromCSV)
            # getting the last 100 rows is arbitrarily above the number of lags.  We could likely select 100 tails rows
            to have more than enough data to calculate 5 lags.
        '''
        lastRowsDf = self.dfFromCSV.tail(100)
        # print(lastRowDf)
        lastRowsDf = pd.DataFrame(lastRowsDf)

        lastRowsDf = lastRowsDf.set_index('time')
        for col in list('ohlc'):
            lastRowsDf[col] = lastRowsDf[col].astype(float)
        # print(lastRowsDf)
        lastRowsDf.drop(lastRowsDf.iloc[:, 5:], inplace=True, axis=1)
        # print('truncatedColumns: ', lastRowsDf)
        frames = [lastRowsDf, data]
        data = pd.concat(frames)
        # print('data: ', data)

        # this will add feature engineered columns of data to self.data
        if not 'd' in data.columns:
            ''' Method to prepare additional time series data
                (such as features data).
            '''
            window = 20

            data['r'] = np.log(data['c'] / data['c'].shift(1))

            data.dropna(inplace=True)
            data['sma'] = data['c'].rolling(window).mean()
            data['min'] = data['c'].rolling(window).min()
            data['max'] = data['c'].rolling(window).max()
            data['mom'] = data['c'].rolling(window).mean()
            data['std'] = data['r'].rolling(window).std()
            data.dropna(inplace=True)
            data['d'] = np.where(data['r'] > 0, 1, 0)
            data['d'] = data['d'].astype(int)

            lags = 5
            cols = []
            for x in data.columns:
                cols.append(x)
            features = ['o', 'h', 'l', 'c', 'volume', 'r', 'sma', 'min', 'max',
                        'mom', 'std', 'd']
            for f in features:
                for lag in range(1, lags + 1):
                    col = f'{f}_lag_{lag}'
                    data[col] = data[f].shift(lag)
                    cols.append(col)
            data.dropna(inplace=True)

            #  This creates a dataset that has Gaussian NORMALIZATION of the data.
            #  however, it is self.data_ is not used here, all of these data methods may need to be
            #  brought into the oadndaenv.py file.
            if self.mu is None:
                self.mu = data['c'].mean()
                self.std = data.std()
            self.data_ = (self.data - self.mu) / self.std
            self.data = data.tail(1)
            return self.data

    def getUnitsToTrade(self):
        '''The plan here is to dynamically get the maximum units to buy or sell while keeping approximately 1/3 of cash to cushion for margin calls.
        If there is already a position, then the units are simply the units of the position already held.
        The units here will always be positive.  If closing out a long position or closing out a short position the negative or positive units are handled in other code.
        The positionValue is the 'positive' number of units of an instrument owned, not the price. if it is 0, then we can calculate the number of units to trade, otherwise, we need to be looking at a closed postion.
        return:  we return a float because that is how oanda needs units entered for an order ????????????????  or should we convert from float to string.
        if we own positions, then positionValue will not be equal to 0.0; and, we will want to return the number of units we own that so we can close or reverse the position'''
        actSummary = self.get_account_summary(self)
        
        #print('actSummary: ', actSummary)
        balance = float(actSummary['balance'])
        positionValue = actSummary['positionValue']
        balanceTwoThirds = balance* .666666
        print('balanceTwoThirds: ', balanceTwoThirds)
        
        cashToTrade = balanceTwoThirds * self.leverage
        print('cashToTrade: ', cashToTrade)
        bidAsk = self.get_prices('USD_JPY')
        print('askPrice: ', bidAsk[2])
        
        print('shouldBeUnits: ', cashToTrade/ bidAsk[2])
        
        # print('balance: ', balance)
        # print('positionValue: ', positionValue)
        if positionValue == '0.0':
            cashAppliedToTrade = int(balance * .666666 * self.leverage)
            bidAsk = self.get_prices('USD_JPY')
            askPrice = bidAsk[2]
            # print('bidAsk: ', bidAsk)
            # print('cashAppliedToTrade: ', cashAppliedToTrade)
            self.units = cashAppliedToTrade / askPrice
            # print('kmV20 units: ', int(units))
            return int(self.units), cashAppliedToTrade
        #else:
            #return 0, 0
        elif self.units != 0.0:
            cashAppliedToTrade =positionValue
            return self.units, cashAppliedToTrade
        else:
            return 0,0
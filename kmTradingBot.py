#
# Oanda Trading Bot
# and Deployment Code
# 
# (c) Dr. Yves J. Hilpisch
# Kent Mercier modified this file.
# Artificial Intelligence in Finance
#
# ***********************************************************
# This class is for production trading. NOT model development
# ***********************************************************
import json
import sys
import kmV20
import tensorflow
import numpy as np
import pandas as pd
import time
import csv
from pathlib import Path

# sys.path.append('../ch11/')

def prove_in_bot():
    # print('in KMTradingBot')
    pass


# class KMTradingBot:
class KMTradingBot:
    # prove_in_bot()

    def __init__(self, granularity='M1', units=0,
                 sl_distance=None, tsl_distance=None, tp_price=None,
                 verbose=True):
        super(KMTradingBot, self).__init__()

        self.position = 0
        # self.agent = agent
        # self.symbol = self.agent.learn_env.symbol
        self.symbol = 'USD_JPY'
        # self.env = agent.learn_env
        self.window = 20
        self.granularity = granularity
        # if granularity is None:
        #     self.granularity = agent.learn_env.granularity
        # else:
        #     self.granularity = granularity
        self.units = units
        self.cashAppliedToTrade = 0
        self.sl_distance = sl_distance
        self.tsl_distance = tsl_distance
        self.tp_price = tp_price
        self.trades = 0
        self.tick_data = pd.DataFrame()
        # self.min_length = (self.agent.learn_env.window +
        #                    self.agent.learn_env.lags)
        self.pl = list()
        self.verbose = verbose
        self.data = pd.DataFrame
        self.workingData = pd.DataFrame
        logHeader = ['time', 'trade', 'direction', 'units', 'cost', 'profitLoss', 'actualOrder']
        with open('tradesHistory3.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(logHeader)

    def get_data(self):
        '''
        This function is not used, as when trading we use get last 50 minute history as we don't need the entire dataset to perform feature engineering.
        '''
        self.raw = pd.read_csv('addFeatures_USD_JPY_2021-01-01_to_2021-11-04_M1_A.csv', parse_dates=['time'])

        self.raw = self.raw.set_index('time')
        # the HISTORICAL data is culled to window/lag size to permit feature engineering
        # now we will let self.data be the subset for either learning, validating or testing purposes.
        self.workingData = self.raw.tail(50)

    def _prepare_data(self):
        '''Method to prepare additional time series data
                (such as features data).
                Generally this function will not be used. Historical and minute by minute data will have been
        preprocessed through kmV20 functions.

        The data coming from HISTORICAL DATA "addFeatures...csv" will have a "d"  column, while
        the data coming from a new oanda live interval retrieval will not have a "d"  columns.
        so, if no "d" column then this is a live retrieval and we need to perform feature engineering on oanda data to create the state for the step.

        IF being fed live OANDA interval data, then we still need historical data to do feature engineering, e.g. ".shift(1)"
        ??? or, do we handle this in a different function e.g. kmV20.get_latest_minute_history, kmV20.feStrategy1, and kmV20.pushToCSVfile.
        if it is being handled by kmV20, then it will have the d column as it has been feature engineered through kmV20.feStrategy1
        '''

        # print('list(self.data.length)', len(list(self.data)))
        if not 'd' in self.data.columns:
            self.data['r'] = np.log(self.data['c'] / self.data['c'].shift(1))
            self.data.dropna(inplace=True)
            # Simple Moving Average of the window time period
            self.data['sma'] = self.data['c'].rolling(self.window).mean()
            # Rolling Minimum of the window time period
            self.data['min'] = self.data['c'].rolling(self.window).min()
            # rolling Maximum of the window time period
            self.data['max'] = self.data['c'].rolling(self.window).max()
            # Momentum as average of log returns
            self.data['mom'] = self.data['r'].rolling(self.window).mean()
            # std is the volatility of the rolling log return
            self.data['std'] = self.data['r'].rolling(self.window).std()
            self.data.dropna(inplace=True)

            # the volume is skewed very much, so we will apply np.log to the volume column before normalizing
            self.data['volume'] = np.log(self.data['volume'])

            # normalization will not work with strings, so remove the time and complete column
            self.data.drop('time', 'complete')
            # machine learning runs faster on normalized data.

            # must retreive the min and max values from the persisted min and max of the training set for consistency on the normalizing of the prediction data

            minim = pd.read_csv('minim.csv')
            maxim = pd.read_csv('maxim.csv')
            stand = pd.read_csv('stand.csv')

            for column in self.data.columns:
                # the 'r' column is the reward column and does not get normalized.
                if column != 'r':

                    self.data[column] = (self.data[column] - minim[column]) / (maxim[column] - minim[column])
                else:

                    self.data['r'] = self.data['r']
            # This also works, but it removes the column headers which we might want later.
            # x = df1.values #returns a numpy array
            # min_max_scaler = preprocessing.MinMaxScaler()
            # x_scaled = min_max_scaler.fit_transform(x)
            # df = pd.DataFrame(x_scaled)

        # don't want to normalize 'd' column, it will be used as the "action", so create 'd'  after normalization occurs.  Note, we may want to add a hold action later
        # here 1 = buy, 0 = sell.
        # the direction of the diffence in log return
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        # change direction from string to integer
        self.data['d'] = self.data['d'].astype(int)

#         ****************************************
#         CLASS IMBALANCE
#         The following applies here when retreiving from Oanda, and within the feature engineering component of modelbuildenv.py
#         The "d' needs to be mathematically fixed, otherwise the learning will almost always converge to 
#         which ever occurs most, e.g buy or sell.
#         This is common with binary features.  See AIFF.

#        ONLY USE THIS WHEN TRAINING THE MODEL
#         ****************************************
        
        
        
        
        #         if done is not None:

        #             # note below is not using dates for index, but instead int indices.
        #             self.data = self.data.iloc[ :self.end-self.start]
        #             self.data_ = self.data_.iloc[:self.end - self.start]
        # **********

        # populate history through lags to be added to features to for state.
        self.features = self._add_lags(self.data)

        # ********** not to sure of the purpose in the below code.  This looks like it is  normalizing the data.

        if self.mu is None:
            self.mu = self.data.mean()
            self.std = self.data.std()
        self.data_ = (self.data - self.mu) / self.std  # Figure this out todo

        self.data_['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data_['d'] = self.data_['d'].astype(int)

    # This is only added in feature engineering which is likely handled in kmV20 and then fed into environment.
    def _add_lags(self, data):
        ''' This is a function that is used in _prepare_data
        '''
        cols = []
        #print(data.columns)
        for x in data.columns:
            cols.append(x)
        #print('cols: ', cols)
        features = ['o', 'h', 'l', 'c', 'volume', 'r', 'sma', 'min', 'max',
                    'mom', 'std', 'd']
        for f in features:
            for lag in range(1, self.lags + 1):
                col = f'{f}_lag_{lag}'
                self.data[col] = self.data[f].shift(lag)
                cols.append(col)
        self.data.dropna(inplace=True)
        #         print('after Adding lags: ', self.data.head(1))
        return cols

    def report_trade(self, time, side, order):
        ''' Reports trades and order details.
        '''
        self.trades += 1
        if 'pl' in order:
            print('there is an order pl: ', order['pl'])
            pl = float(order['pl'])

            data = [time, self.trades, side, self.units, self.cashAppliedToTrade, pl, order]
            with open('tradesHistory3.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow(data)



        else:
            print('there is not an order pl')
            pl = float(0.0)
        self.pl.append(pl)
        cpl = sum(self.pl)
        formattedCash = "${:,.2f}".format(self.cashAppliedToTrade)

        print(f'{time} | Trade {self.trades} GOING {side}, {self.units} for {formattedCash}')
        print(f'{time} | PROFIT/LOSS={pl:.2f} | CUMULATIVE={cpl:.2f}')
        if self.verbose:
            print('actual order: ', order)

    def on_50_minute_success(self):
        ''' Contains the main trading logic.
        '''
        # print('on success')
        kmV201 = kmV20.KMv20()
        self.units, self.cashAppliedToTrade = kmV20.KMv20.getUnitsToTrade(kmV201)
        # print('units: ', self.units)

        # df = pd.DataFrame({'ask': ask, 'bid': bid, 'mid': (bid + ask) / 2},
        #                   index=[pd.Timestamp(time)])
        # print('data head before resampling', df.head())
        # self.tick_data = self.tick_data.append(df)
        # print('self.tick_data', self.tick_data.head())
        # self._resample_data()
        # print('resampled data head', self.data.head())

        # if len(self.data) > self.min_length:
        # self.min_length += 1
        # self._prepare_data()
        # *******************************************************************************************
        
        #print(50*'^')
        #print('self.data.shape: ', self.data.shape)
        #print('self.data: ', self.data)
        state = self.data.to_numpy()
        # print columns
        #print(50*'^')
        #print('state.shape: ', state.shape)
        #print('state: ', state)
        # in KM version we do not import the agent, instead, just the model.
        # prediction = np.argmax(self.agent.model.predict(state)[0, 0])
        # 1) load model

        #'''
        #path note 1:  on aws need the double \\ before the start of the file names to account for the _ in the name.
        #path note 2:  on aws need to use absolute addresses.
        #e.g., model = tensorflow.keras.models.load_model('C:\\Users\Administrator\Documents\MostRecent\\ai_trader_9.h5')
        #'''




        filepath = Path(__file__).resolve().parent
        modelFile = 'ai_trader_30.h5'
        modelPath=Path.joinpath(filepath, modelFile)
        model = tensorflow.keras.models.load_model(modelPath)
        # print('model loaded')
        #print('state: ', state)
        prediction = np.argmax(model.predict(state))
        print('prediction: ', prediction)
        if prediction == 0:
            action = 'sell'
            # print(f'model.predict(state): {model.predict(state)}; Prediction: {action}')
            print(f'Prediction: {action}')

        else:
            action = 'buy'
            # print(f'model.predict(state): {model.predict(state)}; Prediction: {action}')
            print(f' Prediction: {action}')


        # *******************************************************************************************
        position = 1 if prediction == 1 else -1
        # print('position: ', position)
        # print('self.position: ', self.position)
        if self.position in [0, -1] and position == 1:
            order = kmV201.create_order(self.symbol,
                                        units=(1 - self.position) * self.units,
                                        sl_distance=self.sl_distance,
                                        tsl_distance=self.tsl_distance,
                                        tp_price=self.tp_price,
                                        suppress=True, ret=True)
            self.report_trade(self.data.index.values, 'LONG', order)
            self.position = 1
        elif self.position in [0, 1] and position == -1:
            order = kmV201.create_order(self.symbol,
                                        units=-(1 + self.position) * self.units,
                                        sl_distance=self.sl_distance,
                                        tsl_distance=self.tsl_distance,
                                        tp_price=self.tp_price,
                                        suppress=True, ret=True)
            # x = self.data[:-1].index.values
            # print('**********: ', self.data)
            x = self.data.index.values
            time = x[0]
            self.report_trade(time, 'SHORT', order)
            # fulfilledYesNo =
            # print('fulfilled: ', fulfilledYesNo)
            self.position = -1
            return

    def trading(self):
        ''' seach presentation to the model will require
        this approach will require a call to oanda to retreive the last 50 minutes of minute data
        that 50 rows of data is then subjected to feature engineering
        then the last row is submitted to the model for a prediction

        self.data. consists of a feature engineering database

        after prediction empty self.data to prepare for a new 50 rows of data
        '''
        kmV201 = kmV20.KMv20()
        # kmTrade = KMTradingBot()

        from time import time, sleep
        minuteCounter = 0
        startWhileTime = time()
        while minuteCounter <= 1000:
            sleep(60 - time() % 60)
            minuteCounter = minuteCounter + 1
            print('')
            print('Process Session Minute: ', minuteCounter)
            self.data = kmV201.get_latest_50_minute_history(instrument='USD_JPY', granularity='M1', price='A')
            #print('kmTradingBot self.data: ', self.data)
            if len(self.data.index) > 0:
                # print('kmTradingBot.py self.data.tail(1): ', self.data.tail(1))
                print(
                    f'some of state: volume: {self.data["volume"].values}; close: {self.data["c"].values}; r: {self.data["r"].values}; sma: {self.data["sma"].values};'
                    f' mom: {self.data["mom"].values}; d: {self.data["d"].values}')
                kmTrade.on_50_minute_success()
                self.data = self.data.empty
        print('elapsed time: ', time() - startWhileTime)
        print("CLOSING OUT TRADING SESSION")
        print('Process Session Minute: ', minuteCounter + 1)

        # send to closeOpenPosition which will determine if there is a position, then close out.
        # kmV201.closeOpenPosition()
        lastPosition = kmV201.get_positions()
        if lastPosition == []:
            print('There are no positions to close.')
        else:
            order = kmV201.create_order('USD_JPY',
                                        units=-kmTrade.position * kmTrade.units,
                                        suppress=True, ret=True)
            kmTrade.report_trade(kmV20.time, 'NEUTRAL', order)
            print('closing order: ', order)
        print('Proof from Oanda: get_positions: ', kmV201.get_positions())

        # if kmV20.verbose:
        #     print(order)
        # print(71 * '=')


if __name__ == '__main__':
    # 2) load agent
    # do we need the agent? or can we just run model.predict...

    # agent = pickle.load(open('trading.bot', 'rb'))
    # agent = tradingbot.TradingBot(learn_env=learn_env, valid_env = valid_env)
    # print('agent loaded')

    # 3) trading tools
    # ? are they in kmV20 or elsewhere.
    # using kmV20 for now.
    # *************************************************************************
    print(70 * '=')
    print('Starting Trading Session')
    kmTrade = KMTradingBot()
    
    # timer to start and stop function
    
    kmTrade.trading()
    
    
    
    print('Completed Trading Session')
    print(71 * '=')

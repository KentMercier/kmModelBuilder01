#
# Finance EnvironmentF
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#


# note in reinforcement learning episodes is the same as epochs in machine learning.  It is the number of times you iterate through the entire dataset.

import math
# import tpqoa
import kmV20
import random
import numpy as np
import pandas as pd
# import pickle


# for tf-agents *****************************
import base64
import imageio
import IPython

import numpy as np
import pyvirtualdisplay
import math
import numpy as np

import tensorflow as tf

from pathlib import Path


# from tf_agents.agents.ddpg import actor_network
# from tf_agents.agents.ddpg import critic_network
# from tf_agents.agents.ddpg import ddpg_agent

# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.drivers import dynamic_step_driver
# from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import tf_metrics
# from tf_agents.networks import q_network
# from tf_agents.policies import random_tf_policy
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.trajectories import trajectory
# from tf_agents.trajectories import policy_step
# from tf_agents.utils import common

# import gym
# from gym import Env

# from gym import spaces
# from gym.utils import seeding
# from gym.envs.registration import register


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # env.seed(seed)
    # env.action_space.seed(100)


class observation_space:
    def __init__(self, n):
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n

    def seed(self, seed):
        pass

    def sample(self):
        # return one of the action options, e.g. if sample space = 2, then either 0 or 1
        return random.randint(0, self.n - 1)


# class ModelBuilderEnv(gym.Env):
class ModelBuilderEnv:
    '''
    min_accuracy default is used only in learning runs.  When performing validation or testing, the min_accuracy is set to zero.
    '''

    def __init__(
            self,
            symbol,
            start,
            end,
            granularity,
            price,
            features,
            window,
            lags,
            rows,
            leverage=1,
            min_accuracy=.10,
            min_performance = .85,
            mu=None,
            std=None
    ):

        self.symbol = symbol
        self.start = start
        self.end = end
        self.granularity = granularity
        self.price = price
        # self.api = tpqoa.tpqoa('accountConfigs.cfg')
        self.api = kmV20

        # the features define the state
        self.features = features
        self.n_features = len(features)

        # the window is the number of rows being used to create calculate momentum, sma, etc. to prepare feature engineering.  HOWEVER:  here we are using the lags in the preprocessing/feature engineering section as the state.
        # The window calculations are performed for each lag, which is placed within each row.
        self.window = window

        # lags are the number of candlesticks analyzed.
        # need to be careful, because the lags are used for feature engineering, e.g. each row will contain historical data from the number of lags.
        # thus, we do not want to also be pulling multiple rows based on number of lags.
        self.lags = lags
        self.rows = rows
        self.leverage = leverage

        self.min_accuracy = min_accuracy
        # for some reason I have not yet discovered, the self.min_accuracy gets set to a
        # one element tuple, e.g. (.85,)
        # here we check, and if it is a tuple, then we change it to the actual value.
        if type(self.min_accuracy) == tuple:
            self.min_accuracy = self.min_accuracy[0]

        # the minimum gross performance of the RL Model needed to end the training.
        self.min_performance = min_performance
        self.mu = mu
        self.std = std

        # this observation space will be the number of rows to analyze.  here are letting this be a paramater/argument.  However, the feature engineering we are using will incorporate historic prices based on the lags.  Therefore, the observation space should be one row, which in itself contains the historic data of x number of lags.
        self.observation_space = observation_space(self.rows)

        # ??? should the action space be three to permit holding?
        self.action_space = action_space(2)
        self.raw = pd.DataFrame()

        # this may need to be cleaned up.  It takes the number of rows (most likely 1 for the reasons discussed above)
        # self.bar cannot be <= self.lags because causes division by zero or by a negative number in the correct value within the step function.
        self.bar = self.lags -1

        # here we are getting data to prepare the model, not implement the model ???
        self._get_data()

        # we are most likely using the kmV20 data preparation funcitons.
        # self._prepare_data()

        # self.done = True

    def _get_data(self):
        try:
            # *************************************
            # OPTION 1:  Get data from Pickle file on Google Cloud

            #             from google.cloud import storage
            #             from io import BytesIO

            #             file_name ='oandaRaw_USD_JPY_2021-01-01_to_2021-09-21_M1_A.pkl'
            #             bucket_name = "oanda_usd_jpy_1m_a"

            #             client = storage.Client()
            #             print("Client created using default project: {}".format(client.project))
            #             bucket = client.get_bucket(bucket_name)
            # #             print("Bucket name: {}".format(bucket.name))
            # #             print("Bucket location: {}".format(bucket.location))
            # #             print("Bucket storage class: {}".format(bucket.storage_class))
            #             print('bucket created')
            #             blob = bucket.get_blob(file_name)
            #             print('blob created')
            #             content = blob.download_as_string()
            #             print('content created')
            #             print('content 50: ', content[:50])
            # #             self.raw = pd.read_pickle(BytesIO(content))
            #             self.raw = pd.read_pickle(BytesIO(content))
            
            
            # *************************************
            # OPTION 2:  Get Data from local pickle file
            # NOTE THAT This will not work on google cloud platform created pickle files
            # self.raw = pd.read_csv('addFeatures_USD_JPY_2021-01-01_to_2021-09-13_M1_A.pkl')

            #             self.raw = pd.read_pickle('addFeatures_USD_JPY_2021-08-01_to_2021-10-05_M1_A.pkl')

            # Get data from local csv file (not working, trouble converting index to datetime

            # dataFile = 'addedFeaturesToNormalized_'+ticker+'_'+start+'_'+'to_'+end+'_'+granularity+'_'+price+'.pkl'
            # dataFile = 'addedFeaturesToNormalized_USD_JPY_2021-01-01_to_2021-12-31_M1_A' + '.pkl'

            # self.raw = self.raw.set_index('time')
            # self.raw = pd.read_pickle(dataFile)
            # print('HISTORICAL DATA WAS COLLECTED')

        #             # Set the index as datetime
        # #             rawData.index = pd.to_datetime(rawData.index)
        #             rawData.DatetimeIndex()
        #             print(rawData.head(1))

        # except Exception as e:
        #     print('HISTORICAL DATA NOT COLLECTED')
        #     print(e)
            
        #             self.raw = self.api.get_history(self.symbol, self.start,
        #                                        self.end, self.granularity,
        #                                        self.price)
        #             self.raw.to_csv(self.fn)
        #             print('secondary data was collected')

        # these start and end dates will change depending upon whether the runs are learning, validation or test.
        # print('self.start', self.start)
        # print('type self.start: ', type(self.start))
        # print('self.end', self.end)
        # print('type self.end: ', type(self.end))

        #         print('type of index: ', type(rawData.loc[0]))
        #         print('index id: ', rawData.loc[0])
        #         print('type(rawData.index): ', type(rawData.index))

        # the HISTORICAL data will be trimmed to permit the culling of data into learning, validation and test subsets
        
        # *************************************
        # OPTION 3 get data from csv file
        
        pathToCSV = Path('..').joinpath('OandaHistorical').joinpath('2021M1A_Raw').joinpath(csvFile)
        self.raw = pd.read_csv(pathToCSV)
            

        self.raw = self.raw[self.start:self.end]
        # now we will let self.data be the subset for either learning, validating or testing purposes.
        self.data = self.raw

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

        # if data coming from the raw csv file, (which it probably should be) then handle the time issue

        
        
        
        # print('list(self.data.length)', len(list(self.data)))
        if not 'd' in self.data.columns:
            self.data['r'] = np.log(self.data['c'] / self.data['c'].shift(1))
            self.data.dropna(inplace=True)
            # Simple Moving Average of the window time period
            self.data['sma'] = self.data['c'].rolling(self.window).mean()

            # may choose not to add this in the dataset.
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

            # t persist the min and max values for use on the production/prediction data

            minim = self.data.min()
            maxim = self.data.max()
            stand = self.data.std()
            minim.to_csv('min.csv')
            maxim.to_csv('max.csv')
            stand.to_csv('std.csv')


            # machine learning runs faster on normalized data.

            for column in self.data.columns:
                # the 'r' column is the reward column and does not get normalized.
                if column != 'r':
                    self.data[column] = (self.data[column] - self.data[column].min()) / (
                                self.data[column].max() - self.data[column].min())
                else:
                    self.data['r'] = self.data['r']
            # This also works, but it removes the column headers which we might want later.
            # x = df1.values #returns a numpy array
            # min_max_scaler = preprocessing.MinMaxScaler()
            # x_scaled = min_max_scaler.fit_transform(x)
            # df = pd.DataFrame(x_scaled)

        # don't want to normalize d column, it will be used ass the "action", so 'd'  after normalization occurs.  Note, we may want to add a hold action later
        # here 1 = buy, 0 = sell.
        # the direction of the diffence in log return
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        # change direction from string to integer
        self.data['d'] = self.data['d'].astype(int)

        #         if done is not None:

        #             # note below is not using dates for index, but instead int indices.
        #             self.data = self.data.iloc[ :self.end-self.start]
        #             self.data_ = self.data_.iloc[:self.end - self.start]
        # **********

        # populate history through lags to be added to features to for state.
        self.features = self._add_lags(self.data)
    
    def class_weights(self):
        '''class imbalance'''
        df = self.data
        c0, c1 = np.bincount(df['d'])
        # print('c0: ', c0)
        # print('c1: ', c1)
        w0 = (1 / c0) * (len(df))/ 2
        w1 = (1 / c1) * (len(df))/ 2
        print('w0 ', w0)
        print('w1 ', w1)
        # df.loc[df['d'] == 1, 'd'] *= w1
        # df.loc[df['d'] == 0, 'd'] += w0
        return{0: w0, 1: w1}

    # This is only added in feature engineering which is likely handled in kmV20 and then fed into environment.
    def _add_lags(self, data):
        ''' This is a function that is used in _prepare_data
        '''
        cols = []
        print(data.columns)
        for x in data.columns:
            cols.append(x)
        print('cols: ', cols)
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

    #   FOR THE NEXT TWO METHODS NEED TO CHANGE INDEX OR CALL TO INDEX.

    def _get_state(self):
        ''' Private method that returns the state of the environment.
        The original code takes several rows of data as the state.
        todo
        ??? will the self.bar work with this, or does it need to be converted to 
        datetime index references.
        todo
        ??? since the feature engineering adds lag data, isn't it sufficient to load one row?
        ??? so, why do we need to load the number of rows = to the number of lags?
        '''
        # here we are obtaining the entire dataset less those from the beginning???
        # return self.data_[self.features].iloc[self.bar -
        #                             self.lags:self.bar].values
        # print('self.bar: ', self.bar)

        # here we are returning one row, and this MAY NOT BE CORRECT, see immediately above.
        # return self.data_[self.features].iloc[self.bar].values
        # print('data no values: ', self.data_[self.features].iloc[self.bar])
        # print('data values: ', self.data_[self.features].iloc[self.bar].values)
        # print(f'self.bar: {self.bar} / len(self.data_[self.features].iloc[self.bar]): {len(self.data_[self.features].iloc[self.bar])}')
        # return self.data[self.features].iloc[self.bar].values

        # or should we be restarting the bars after every episode
        return self.data[self.features].iloc[0].values

    # def get_state(self, bar):
    #     ''' Method that selects the data defining the state of the environment.
    #     ??? will the self.bar work with this, or does it need to be converted to 
    #     datetime index references.
    #     ??? since the feature engineering adds lag data, isn't it sufficient to load one row?
    #     ??? so, why do we need to load the number of rows = to the number of lags?
    #     '''
    #     # return self.data_[self.features].iloc[bar - self.lags:bar].values
    #     return self.data_[self.features].iloc[self.bar].values

    def reset(self):
        # print('called reset')
        ''' Method to reset the environment.
        resets the rewards and accuracy to neutral
        then creates and returns a new state for submission for training of model (? or selection of policy?
        '''
        self.totalCorrect = 0
        self.accuracy = 0
        self.performance = 1
        self.totalReward = 0
        self.bar = self.lags + 1
        # self.bar = self.lags
        # self.bar = 0
        # print('self.bar: ', self.bar)

        # this returns a state to get the next episode started
        state = self._get_state()
        # print('state: ', state)
        # print('modelBuilderEnv.py state.shape: ', state.shape)
        # print('state.values: ', state.values)
        # the get_state() function returns values only. so here we don't need to return state.values
        return state

    def step(self, action):
        ''' checks whether the right action has been taken, defines the reward accordingly, and checks for success or failure.
        e.g. success can be when an agent trades successfully for 1,000 steps.  A failure is defined as an accuracy ration of less than
        e.g. 50%... the total rewards divided by total number of steps.  HOWEVER, this is only checked after a certain number of steps to avoid the high initial variance of this metric.
        '''
        # first we check whether the agent has selected the correct action
        # if d = 1, then correct, if d = 0 then not correct.
        # so what is the purpose of correct = action == 
        # ??? why not just correct == self.data...

        # we need to determine if the action take was the correct action.
        correct = (action == self.data['d'].iloc[self.bar])
        isCorrect = 1 if correct else 0
        # accumulate the total "wins" 1 if true, 0 if false
        self.totalCorrect += isCorrect

        # the return based reward FOR THE STEP.
        # set up reward system.
        # reward 1 is very simplely win or lose
        # reward_2 is the cash reward or loss

        # isn't the following line redundant since correct is already 0 or 1
        # reward_1 = 1 if correct else 0 

        # the leverged return FOR THE STEP.
        # the tradeReturn is the return from ['r'] times the leverage amount.  (? This represents cash
        # return) and is positive or negative.

        tradeReturn = self.data['r'].iloc[self.bar] * self.leverage

        # ??? why do we need reward_2. isn't the positive and negative reflected in the ['r'] value?  No, we need to account for buying long and selling short.  This sets the return to a positive addition of money if the trade was correct.  
        # reward_2 = abs(tradeReturn) if correct else -abs(tradeReturn)
        reward = abs(tradeReturn) if isCorrect else -abs(tradeReturn)
        
        self.totalReward += reward
        
        # the actual reward is the two rewards times the leverage.
        # ??? why are we multiplying by the leverage.  Doesn't the tradeReturnalready do this?
        # reward = reward_1 + reward_2 * self.leverage

        # The self.bar - self.lags determines accuracy of all trades completed.
        # ??? why are we doing this accuracy setting after self.bar is incremented and not before.
        # perhaps due to zero based index?
        # so i am going to try just using the self.bar as the counter to divide into the total reward, get a percentage of accuracy.

        self.accuracy = self.totalCorrect / (self.bar)


        # we do not need to keep track of the "osn" observation space number because our self.data has already been trimmed of the leading
        # rows
        # self.accuracy = self.totalCorrect / (self.bar - self.osn)
        # KM's approach is to place lags in one row as features and iterate one row at a time.
        # The self.data has had rows with empty data deleted so we do not need to subtract a certain number of bars.
        # (however, we man need to check and see how this "loop" considers the very first row.

        # increment the bar aka the row to move the environment forward.
        self.bar += 1

        # ??? why is self.performance multiple by the exponent of reward_2
        # this is supposed to be the gross performance after the step.
        # why multiplied by the exponent at reward???
        # it looks like it is that so the number will be between 0 and 1 for comparison to the self.min_performance

        self.performance *= math.exp(reward)
        

        # determine whether the episodes should be terminated.  This model is build on time series.
        # so if we don't place a method to stop if the model building gets off to a bad start, then the learning process
        # will need to run through all of the
        # the 2nd elif checks to see if accuracy has fallen below minimum accuracy allowed, if so, the episode ends as a failure.
        # the reason there is a second conditional that is checked is to make sure that the accuracy is not tested against the minimal
        # accuracy until after a certain threshold of attempts.  The scalar (e.g. 15) is simply a random number of iterations added to the
        # lags to make sure there are a few iterations before terminating the iteration through the rows.
        # the 3rd elif checks to see
        # the 4th elif check to see if the performace has met a minimup threshold prior to permitting the model to continue ot be developed.

        # final else allows all remaining cases to move forward

        # done = False

        # print('self.min_performance: ', self.min_performance)
        # print('self.performance: ', self.performance)
        # print('self.accuracy: ', self.accuracy)
        # print('self.min_accuracy: ', self.min_accuracy)
        # use this for debugging only, then switch to self.data
        # if self.bar >= 2:

        # if self.bar % 2000 == 0:
            # print(f'self.bar: {self.bar}; self.performance: {self.performance}')

            # print('reward: ', reward)
            # print(f'self.bar: {self.bar} self.totalReward: {self.totalReward}')

        if self.bar >= (len(self.data) - 1):
            done = True
            print('self.bar >= len(self.data); self.bar: ', self.bar)
        # elif correct == 1:
        #     done = False
        # elif (self.accuracy < self.min_accuracy and
        #       self.bar > self.lags + 15):
        elif ((self.accuracy < self.min_accuracy) and
              (self.bar > 50)):
            # print('Failed:  self.accuracy < self.min_accuracy; self.bar: ', self.bar)
            done = True
        # elif (self.performance < self.min_performance and
        #       self.bar > self.lags + 15):
        #     done = True
        elif (self.performance < self.min_performance and
              self.bar > 50):
            # print('Failed: self.performance < self.min_performance:  ', self.bar)
            done = True
        else:
            # print('accurcy and performance > minimum thresholds')
            done = False

            # now we are getting a new state for next policy selection.\
        state = self._get_state()

        # ??? not to sure why we need the info
        info = {}
        return state, reward, done, info

#
# Financial Q-Learning Agent
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
import os
import random
import logging
import time
from timeit import default_timer as timer
from datetime import datetime
from datetime import timedelta
import numpy as np
import math
from pylab import plt, mpl
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

os.environ['PYTHONHASHSEED'] = '0'
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'



import os
from twilio.rest import Client



def set_seeds(seed=100):
    ''' Function to set seeds for all
        random number generators.
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TradingBot:
    ''' TradingBot contains the tf.keras rl model.  a calling function will set up the learning env, then a second calling function will start the learning with this bot.
    '''

    def __init__(self, learn_env, valid_env=None, hidden_units=32, learning_rate=0.001,
                 val=True, dropout=False):
        self.learn_env = learn_env
        self.dataLength = len(self.learn_env.raw)
        self.valid_env = valid_env
        self.val = val
        
        self.account_sid = "ACaaec9a369c3fcfd58d59e414b2641557"
        self.auth_token = "5cf5fdd6e3050d791e2896b69880028a"
        self.client = Client(self.account_sid, self.auth_token)

        self.classWeights = self.learn_env.class_weights()
        print('self.classWeights: ', self.classWeights)
        
        # self.epsilon = the threshold for exploration vs exploitation. A random number 0-1 is generated and if less than the epsilon, then exploration.
        #         The epsilon parameter is used to determine whether we should use a random action or to use the model for the action. We start by setting it to 1.0 so that it takes random actions in the beginning when the model is not trained.
        # Over time we want to decrease the random actions and instead we can mostly use the trained model, so we set epsilon_final to 0.01
        # Then set the speed of the decreasing epsilon in the epsilon_decay parameter

        self.epsilon = 1.0
        self.epison_final = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        # We need to set an gamma parameter to 0.95, which helps to maximize the current reward over the long-term reward.  This should generally be a value between 0.9 and 1.
        self.gamma = 0.95

        self.batch_size = 128
        # self.batch_size = 390  # 390 minutes in a trading day

        self.max_treward = 0
        self.averages = list()
        self.trewards = []
        # self.trewards = list()

        self.averages = list()
        self.performances = list()
        self.aperformances = []
        self.vperformances = list()
        self.arrayBalanceOfTrades = [0]
        # deque is a python list that pops out the oldest entry.
        # We set the experience replay memory to deque with 2000 elements inside it
        self.memory = deque(maxlen=2000)

        # initialize the model
        # hidden_units is an integer
        # learning rate is a number between 0 and 1
        # dropout is true or false
        self.model = self._build_model(hidden_units,
                                       learning_rate, dropout)

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="tb_callback_dir",
            histogram_freq=1)

        self.replayCounter = 0
        print('1- self.replayCounter: ', self.replayCounter)

        self.physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs:", len(self.physical_devices))

    def _build_model(self, hidden_units, lr, dropout):
        ''' Method to create the DNN model.
        
        hu = the hidden units or hidden layers.
        lr = learning rate
        dropout = the layer dropout rate.
        '''
        model = tf.keras.models.Sequential()
        # ??? why are we adding the number of lags to the input_shape of the Sequential() model?
        # ??? Shouldn't it be just the features, as the features already contain the lag data from preprocessing?
        hidden_units_2 = hidden_units * 2
        hidden_units_4 = hidden_units * 4
        # perhaps insert next 5 lines of commented out code after the original model.add()
        # if dropout:
        #     model.add(Dropout(0.3, seed=100))
        # model.add(Dense(hu, activation='relu'))
        # if dropout:
        #     model.add(Dropout(0.3, seed=100))

        # if use the below model, then need to add a tensorflow flatten function
        # model.add(layers.Dense(units = hu, input_dim=(
        #     self.learn_env.lags, self.learn_env.n_features),
        #     activation='relu'))

        # the states are a singular row of feature engineered data, which includes calculated features for each of the previous rows, the number of which are determined by the lag input.  e.g. 5 lags are = to 5 previous days, 5 previous minutes, 5 previous tics, etc. all compiled into one row.
        # the alternative, which many tutorials uses is to simply use a number of lags on the tics being live streamed into the application.  The problem with this approach is that the streaming ticks do NOT contain volumne information as generally, only ask and bid prices are streamed.  (This may be able to be cured by another approach which is to look at a streamed order book which may contain 'size' on some platforms)
        # note the state is nothing more than a vector of numbers.  when using the lag approach, the array of lags and features must be flattened into a single vector. With the feature engineering approach I am currently using, the vector is in essence created by each row containined feature enginerred lag data within the row.
        # here we will use a fully connected, or "Dense" network.  The units and activation function are all "guesses" and required much further analysis for comparisson of options.  Here we will simply use what the tutorials are suggesting.
        # the first layer is the input layer and the input_dim will be set to the state size, which here is the number of features or the number of column headings in the row submitted as the state.  We will be modifying our actions based upon our rewards which is a continuous number, and not a class.

        model.add(layers.Dense(units=hidden_units, input_dim=(
            self.learn_env.n_features),
                               activation='relu'))
        model.add(Dropout(0.3, seed=100))
        model.add(layers.Dense(units=hidden_units_2, activation='relu'))
        model.add(Dropout(0.3, seed=100))
        model.add(layers.Dense(units=hidden_units_4, activation='relu'))

        # the neurons of the output layer should contain the same number as the number of classes or actions available for the model to choose  from. e.g. buy, sell, hold.
        # Because we are going to use the mean squqred error for our loss, we will change the activation to linear.
        model.add(layers.Dense(units=self.learn_env.action_space.n, activation='linear'))

        # model.compile(
        #     loss='mse',
        #     optimizer=RMSprop(lr=lr),
        #     metrics = ["accuracy"]
        # )           
        # Finally, we need to compile the model. Since this is a regression task we can't use accuracy as our loss, so we use mse. We then use the Adam optimizer and set the learning rate to 0.001 and return the model:
        # ??? should the learning rate become a scheduled decaying learning rate see https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
        # becauase we are not using classification, but instead regression, we cannot use accuracy as the metric.  So, we just leave this parameter as empty.
        # model.compile(
        #     loss='mse',
        #     optimizer=tf.keras.optimizers.RMSprop(learning_rate = lr),
        #     metrics = ["accuracy"]
        # )

        #  the default lr for the .Adam optimizer as 0.001.  This will require some experimentation and study to find the best parameter.
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.RMSprop(lr=lr)
        )
        return model

    # in some videos the following is called the "trade()" function, see, e.g. http://localhost:8888/lab/tree/km-development/km_rlOanda_deployment-02/tradingbot.py

    def act(self, state):
        ''' 
        SUMMARY:  This is the actuall function that returns whether to buy, sell, or hold.
        The function is called within the looping through the data rows in the learn function.
        Method for taking action based on
            a) exploration
            b) exploitation
        Now that we've defined the neural network we need to build a function to trade that takes the state as input and returns an action to perform in that state. To do this we're going to create a function called trade or act that takes in one argument: state.  For each state, we need to determine if we should use a randomly generated action (exploration) or the action from the neural network prediction (exploitation).  To do this, we use the random library, and if it is less than our epsilon we return a random action with random.randrange() and pass in self.action_space.  If the number is greater than epsilon (the rate at which we inject exploration) we use our model to choose the action. To do this, we define actions equal to self.model.predict and pass in the state as the argument.  We then return a single number with np.argmax to return only the action with the highest probability.

To summarize:
The function takes as input the state and generates a random number
If the number is less than or equal to epsilon it will generate a random action (this will always be the case in the beginning)
If it is greater than epsilon it will use the model to perform a prediction on the input state based upon the model and return the action that has the highest probability
        '''
        # print('state: ', state)
        if random.random() <= self.epsilon:
            # print('self.learn_env.action_space.sample(): ', self.learn_env.action_space.sample())

            return self.learn_env.action_space.sample()

        # if false, then use the model.predict based on the state to return an action. ??? does the self.model.predict return more than one action, since the return is getting the max of the action?
        # todo why not just state ???  was this left over from when the algorithm passed in lags and features?
        # print('type of state: ', type(state))
        # state = np.array(state)

        # state = state.tolist()
        # print('state:', state)
        # print('type of state: ', type(state))
        # print('state.shape: ', state.shape)

        # state = np.array(state)
        # state = state.reshape(72)

        # print('state.shape: ', state.shape)

        # print('shape of state: ', state.shape)
        # if the state.shape is not equal to 72, (don't yet understand why this is happening)
        # then the return is not acted upon by caller of this function  learn()
        if state.shape != 72:
            # print('&&&&& row # that is not = 72: ', self.learn_env.bar)
            # print('state.shape: ', state.shape)
            # print('length of state: ', len(state))
            # print('self.learn_env.n_features: ', self.learn_env.n_features)
            return 'state.shape != 72'
        # action = self.model.predict(np.expand_dims(state, 0))
        # print('state[0]: ', state[0])
        # action = self.model.predict(state)

        # print('********** action: ', action)
        # print('np.argmax(actions: ', np.argmax(action))
        # np.argmax(action) returns an array of two numbers totalling 1,
        # with each number consisting of the index number of the highest value.
        # This index number becomes the action to be taken, 0 for sell, 1 for buy.
        # the return is the index number for the maximum value.
        return np.argmax(self.model.predict(state))

    def replay(self):
        '''  Training the Model
        Method to train the DNN model based on batches of memorized experiences.
        
Now that we've implemented the trade function let's build a custom training function. The size of the batch is something that may need to be experimeted with.

This function will take a batch of saved data and train the model on that, below is a step-by-step process to do this:

We define this function batch_trade and it will take batch_size as an argument
We select data from the experience replay memory by first setting batch to an empty list [hilpisch uses an actual random sample and this is what I do here]
We then iterate through the memory with a for loop
Since we're dealing with time series data we need to sample from the end of the memory instead of randomly sampling from it
Now that we have a batch of data we need to iterate through each batch—state, reward, next_state, and done—and train the model with this
If the agent is not in a terminal state we calculate the discounted total reward as the current rewardreplay
Next we define the target variable which is also predicted by the model
Next we fit the model with self.model.fit()
At the end of this function we want to decrease the epsilon parameter so that we slowly stop performing random actions
        '''

        # 1) select data from the experience in self.memory.

        # ??? do we  want to use random samples since this is time series and we need to take them in order???
        # batch = []
        # print('***** in replay')

        # print('self.batch_size: ', self.batch_size)
        # print('self.memory', self.memory)
        # print('len(self.memory): ', len(self.memory))
        batch = random.sample(self.memory, self.batch_size)
        # print('batch', batch)
        # print('length of batch: ', len(batch))

        # create the batch; ... not to sure how this loop works yet.
        # ______________________________
        # print('memory range: ', range((len(self.memory)-self.batch_size), len(self.memory)))
        # for i in range(len(self.memory)-self.batch_size +1, len(self.memory)):
        #     # print('self.memory[i]: ' [i], self.memory[i])
        #     batch.append(self.memory[i])
        # __________________________

        # look through the batch.
        for state, action, reward, next_state, done in batch:
            # we need to make sure the variables coming in are in the correct order, so let's make sure that we set reward as the reward ???
            # reward = reward

            # print(f'action: {action}; reward: {reward}; done: {done}')
            # make sure not in the terminal state
            # then calculate the discounted total reward.
            # print('next_state: ', next_state)
            # print(f'state: {state}; action: {action}; reward: {reward}; next_state: {next_state}; done {done}')
            # print('     done: ', don)
            if not done:
                # print('     IN IF STATEMENT')
                # print('     len(state): ', len(state))
                # print('     len(next_state): ', len(next_state))
                # print('typeof(next_state): ', type(next_state))
                #                 print('     state.shape ', state.shape)
                #                 print('     next_state.shape.state: ', next_state.shape)

                # try:
                #     print('    self.model.predict(next_state): ', self.model.predict(next_state))
                # except :
                #     print('    error')
                # print('np.amax(self.model.predict(next_state)): ', np.amax(self.model.predict(next_state)))
                reward += self.gamma * np.amax(self.model.predict(next_state))
                # print('replay reward: ', reward)

                # if len(state) == (72,):
                state = np.reshape(state, [1,
                                           self.valid_env.n_features])

                target = self.model.predict(state)
                # print('replay target: ', target)
                target[0][action] = reward
                # print('target: ', target)
                # print('replaytarget  2 : ', target)

                # verbose = 2 prints out one line per epoch
                # Epoch 1/3
                # 1/1 - 1s - loss: 8.0583e-04 - accuracy: 1.0000 - 1s/epoch - 1s/sample
                # verbose = 1 prints out a progress bar
                # verbose = 0 prints nothing.
                # setting epochs to one means we will train very frequently, on each sample from the batch

                # note in reinforcement learning episodes is the same as epochs in machine learning.
                # self.model.fit(state, target, epochs=1,
                #                class_weight = self.classWeights,
                #                verbose=False,
                #                callbacks=[self.tensorboard_callback])
                self.model.fit(state, target, epochs=1,
                   class_weight = self.classWeights,
                   verbose=False,
                )
        # now reduce the exploration function that so as we proceed we can shift from frequently exploring to infrequently exploring with actions and frequently applying the model prediction for action.
        if self.epsilon > self.epison_final:
            self.epsilon *= self.epsilon_decay

    # we need to Remember the experience in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        ''' need to remember the experience in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing action on state
            next_state (tensor): next state
            '''
        # ? should item be an array?
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    def learn(self, episodes):

        ''' Method to train the DQL agent.
        '''
        episodesPlus = episodes + 1
        self.trewards.clear()

        for episode in range(1, episodesPlus):
            # if e % 100 == 0:
            #     print(f"_______________ Learning Episode: {e}/{episodes}")
            # debugging
            print('_____________________________________________')
            print(f"START EPISODE {episode} of {episodes}")
            start = datetime.now()
            startFormatted = datetime.strftime(start, "%A, %B %d, %Y %I:%M:%S")
            print('episode start time: ', datetime.strftime(start, "%A, %B %d, %Y %I:%M:%S"))
#             print('start time: ', start)
#             smsMessage = f"Start {episode}/{episodes} episodes at {startFormatted}"
#             self.client.messages.create(
#                 to = "13377813583",
#                 from_ = "18647783608",
#                 body = smsMessage
#             )
            balanceOfTrades = 0
            total_reward = 0
            # print("self.dataLength: ", self.dataLength)

            # reset the starting state to start learning again
            # This reset call returns the state based on the bar that the env maintains.  the reset component resets the metrices to to measure the new episode performance.

            state = self.learn_env.reset()

            # print('state: ', state)

            # working with state and rows, 
            # create the measurements.
            # total_profit = 

            # state = np.reshape(state, [1, self.learn_env.lags,
            #                            self.learn_env.n_features])
            # state = np.reshape(state, [1, self.learn_env.n_features])
            # print('len(state): ', len(state))

            # get the length of the dataset for the loop in learn.

            # *** change when debugging.
            rows = self.dataLength - self.learn_env.lags

            # for debugging
            # rows = 390

            # here we loop though all of the rows in the dataset to BUILD A MODEL.
            # for row in range(self.dataLength):
            # for debugging, we simply list a few rows.
            # must be rows -1 that so we don't get a zero indexed out of bounds error.
            for row in range(rows):
                if self.learn_env.bar > (len(self.learn_env.raw) -self.learn_env.lags):
                    # print(f'len(self.learn_env.data) - self.learn_env.lags: {(len(self.learn_env.data) - self.learn_env.lags)}')
                    # print(f'self.learn_env.data["d"].iloc[self.learn_env.bar]: {self.learn_env.data["d"].iloc[self.learn_env.bar]}')
                    # print(f'action: {action}')
                    break
                # print(f'row: {row}')
                # if row %130 == 0:
                # print(f'Episode: {e}; 130th row: {row}')

                # state = state.tolist()
                # print('type of state: ', type(state))
                # print('state: ', state)
                # print('len(state)', len(state))
                action = self.act(state)
                if action == 0 or action == 1:
                    next_state, reward, done, info = self.learn_env.step(action)
                    # next_state = np.reshape(next_state,
                    #                         [1, self.learn_env.lags,
                    #                          self.learn_env.n_features])
                    # print('done: ', done)
                    if action == 0:
                        actionType = 'sell'
                    else:
                        actionType = 'buy'
                    # print(f'actionType: {actionType}; reward: {reward}; done: {done}')

                    # print(f'learn_env.bar: {self.learn_env.bar} row: {row}; action: {actionType} reward: {reward}')
                    # next_state = np.reshape(next_state,
                    #         [1, self.learn_env.n_features])

                    next_state = np.reshape(next_state,
                                            [1, self.learn_env.n_features])
                    # print(f'row: {row} len(next_state) {len(next_state[0])}')

                    # print('type(state): ', type(state))
                    # print('state: ', state)
                    # print('type(next_state: ', type(next_state))
                    # print('next_state: ', next_state)
                    
                    self.remember(state, action, reward, next_state, done)
                    # self.memory.append([state, action, reward,
                    #                     next_state, done])

                    # print('len(next_state): ', len(next_state))

                    state = next_state

                    # print('reward: ', reward)

                    self.trewards.append(reward)

                    # print('1-self.trewards: ', self.trewards)

                    # if row == (rows -1):
                if done == True:
                    # print(' ')
                    # print(f'row: {row}; done = {done}')

                    # print('2-self.trewards: ', self.trewards)
                    balanceOfTrades = sum(self.trewards)
                    # print('balanceOfTrades: ', balanceOfTrades)
                    self.arrayBalanceOfTrades.append(balanceOfTrades)
                    np.savetxt("arrayBalanceOfTrades.csv", 
                               self.arrayBalanceOfTrades,
                               delimiter =", ", 
                               fmt ='% s')

                    # print(self.arrayBalanceOfTrades)
                    average = sum(self.trewards) / (self.learn_env.bar + 1)
                    # print('average: ', average)
                    # print(f'self.learn_env.bar : {self.learn_env.bar}; row = {row}')
                    self.averages.append(average)
                    perf = self.learn_env.performance
                    self.performances.append(perf)
                    self.aperformances.append(
                        sum(self.performances[-25:]) / 25)
                    # self.max_treward = max(self.max_treward, treward)

                    # print('self.max_treward: ', self.max_treward)
                    # templ = 'episode: {:2d}/{} | '
                    # templ += 'perf: {:5.3f} | av: {:5.1f}'
                    # print(templ.format(e, episodes, perf,
                    #                    av, self.max_treward), end='\r')
                    # print(' ')

            else:
                pass
                # print('action not 0 or 1:', action)
        # move following line to a test when running larger episodes
        # if e %130 == 0:
            
            
            print(f"END OF EPISODE {episode} of {episodes}")
            end = datetime.now()
            endFormatted = datetime.strftime(start, "%A, %B %d, %Y %I:%M:%S")
            elapsed = end-start
            print('episode end time: ', datetime.strftime(end, "%A, %B %d, %Y %I:%M:%S")) 
            elapsedFormatted = str(elapsed)
            bal = 0.0
            if len(self.arrayBalanceOfTrades) >1:
                bal = self.arrayBalanceOfTrades[-1]
            
            smsMessage2 = f"End {episode}/{episodes} episodes at {endFormatted} for a total of {elapsedFormatted} with the following balance of trades {bal} "
            print('episode time duration: ', elapsed)
            print(smsMessage2)
            print('_____________________________________________')
            epmod = episode % 10
            print('epmod: ', epmod)
            
            print('episodes: ', episodes)
            print(episode == 1)
            print(episode % 10 == 0)
            print(episode == episodes)
            if (episode == 1) or (episode > 9 and (episode % 10 == 0)) or (episode == episodes):
                smsMessage2 = f"End {episode}/{episodes} episodes at {endFormatted} for a total of {elapsedFormatted} with the following balance of trades {bal} "
                self.model.save("ai_trader_{}.h5".format(episode))
                self.client.messages.create(
                    to = "13377813583",
                    from_ = "18647783608",
                    body = smsMessage2
                )
                

            
            # if self.valid_env:
            #     self.validate(e, episodes)
            # print ('self.memory: ', self.memory)
            # print(f'len(self.memory): {len(self.memory)} len(self.batch_size): {self.batch_size}')
            if len(self.memory) > self.batch_size: 
                # print('go to self.replay')
                self.replay()

    def validate(self, e, episodes):
        print('in validation')
        ''' Method to validate the performance of the
            DQL agent.
        '''
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.n_features])

        #         for _ in range(10000):

        for _ in range(self.dataLength):
            action = np.argmax(self.model.predict(state)[0, 0])
            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(next_state, [1,
                                            self.valid_env.n_features])
            if done:
                print('_ = : ', _)
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)

                #                 if e % int(episodes / 6) == 0:
                if e % math.ceil(episodes / 6) == 0:
                    templ = 71 * '='
                    templ += '\nepisode: {:2d}/{} | VALIDATION | '
                    templ += 'treward: {:4d} | perf: {:5.3f} | eps: {:.2f}\n'
                    templ += 71 * '='
                    print(templ.format(e, episodes, treward,
                                       perf, self.epsilon))
                break

    # def plot_treward(agent):
    def plot_totalReward(self):
        ''' Function to plot the total reward
            per training eposiode.
        '''
        plt.figure(figsize=(10, 6))
        x = range(1, len(self.averages) + 1)
        y = np.polyval(np.polyfit(x, self.averages, deg=3), x)
        plt.plot(x, self.averages, label='moving average')
        plt.plot(x, y, 'r--', label='regression')
        plt.xlabel('episodes')
        plt.ylabel('total reward')
        plt.legend()

    def v_plot_performance(self):
        ''' Function to plot the financial gross
            performance per training episode.
        '''
        plt.figure(figsize=(10, 6))

        x = range([1, len(self.vperformances) + 1])
        #     x = np.array([1, len(agent.performances) + 1])
        print('x: ', x)
        print('type(x): ', type(x))
        y = np.polyval(np.polyfit(x, self.vperformances, deg=3), x)
        print('y: ', y)
        plt.plot(x, self.vperformances[:], label='training')
        plt.plot(x, y, 'r--', label='regression (train)')
        if agent.val:
            y_ = np.polyval(np.polyfit(x, self.vperformances, deg=3), x)
            plt.plot(x, agent.vperformances[:], label='validation')
            plt.plot(x, y_, 'r-.', label='regression (valid)')
        plt.xlabel('episodes')
        plt.ylabel('gross performance')
        plt.legend()

    def a_plot_performance(self):
        ''' Function to plot the financial gross
            performance per training episode.
        '''
        plt.figure(figsize=(10, 6))
        
        x = range(1, len(self.aperformances) + 1)
        
        #     x = np.array([1, len(agent.performances) + 1])

        y = np.polyval(np.polyfit(x, self.aperformances, deg=3), x)
        plt.plot(x, self.aperformances[:], label='training')
        plt.plot(x, y, 'r--', label='regression (train)')
        # if self.val:
        #     y_ = np.polyval(np.polyfit(x, self.aperformances, deg=3), x)
        #     plt.plot(x, self.aperformances[:], label='validation')
        #     plt.plot(x, y_, 'r-.', label='regression (valid)')
        plt.xlabel('episodes')
        plt.ylabel('gross performance')
        plt.legend()
        
        
        
        
        
        
        
        
        # These three functions are not used and are already accounted for in ___  they are
        # left here for demonstrative purposes.
        '''
    Stock Market Data Preprocessing
    Now that we've built our AI_Trader class we now need to create a few helper functions that will be used in the learning process.

    In particular, we need to define the following 3 functions:
        '''

    def sigmoid(x):
        '''  1. sigmoid - sigmoid is an activation function, generally used at the end of a network for binary classification as it scales a number to a range from 0 to 1. This will be used to normalize stock price data.
        '''
        pass
        # return 1(1+mat.exp(-x))

    def stock_price_format(n):
        '''
    2. stocks_price_format - this is a formatting function to print out the prices of the stocks we bought or sold.
        '''
        if n < 0:
            return "- # {0:2f}".format(abs(n))
        else:
            return "$ {o:2f}".format(abs(n))

#     def dataset_loader(stock_name):
#         '''
#     3. dataset_loader - this function connects with a data source and pulls the stock data from it, in this case we're loading data from Yahoo Finance:
#         '''
#         dataset = data_reader.DataReader(stock_name, data_source = "yahoo:)
#         start_date = str(dataset.index[0]).split()[o]
#         end_date = str(dataset.index[1]).split()[0]
#         close = dataset['Close']
#         return close

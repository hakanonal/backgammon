import numpy as np
import random
from os import path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Reshape
from keras.optimizers import Adam
from keras.models import load_model
from keras.activations import softmax
import wandb

class agent:

    def __init__(self,color, discount=0.95,exploration_rate=0.9,decay_factor=0.9999):
        self.color = color # value must be -1 or +1 (-1 for black and +1 is for white)
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.decay_factor = decay_factor

        wandb.config.update({'model_name':'Dense,208-104-52 with relu, adam, mean_squared_error'})
        if(path.exists(self._getModelFilename())):
            self.model = load_model(self._getModelFilename())
        else:
            self.model = Sequential()
            self.model.add(Dense(208,activation="relu", input_shape=(28,)))
            self.model.add(Dense(104,activation="relu"))
            self.model.add(Dense(52))
            self.model.add(Reshape((2,26)))
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


    def _getModelFilename(self):
        return "agent_model%d.h5" % self.color
    
    def _getBARi(self):
        if(self.color == 1):
            return 25
        return 0
    def _getLOpponentBARi(self):
        if(self.color == 1):
            return 0
        return 25


    def _getHomeRange(self):
        if(self.color == 1): 
            return range(1,7)
        return range(20,26)

    def _getTotalCheckersAtHome(self,state):
        t1 = 0
        for i in self._getHomeRange():
            if((state[i] * self.color) > 0):
                t1 += (state[i] * self.color)
        return t1
    
    def play(self,old_state,action):
        state = old_state.copy()
        di1,di2 = action
        d1 = state[26] * self.color
        d2 = state[27] * self.color
        bari = self._getBARi()
        
        #These are invalid plays and return negative points.
        #can not play other players checker or empty location
        if(state[di1] * self.color <= 0 or state[di2] * self.color <= 0):
            return state,-20        
        #can not play outside from 26 locations (including bars).
        if(di1 not in range(26) or di2 not in range(26)):
            return state,-20
        #can not play outside to 24 locations. TODO: this is not valid when collecting checkers at the end of the game.
        if(di1-d1 not in range(1,25) or di2-d2 not in range(1,25)):
            return state,-20
        #if two dice location is same there should be more than 1 cheker of that location. TODO: if two dice is same then there should more than 3 checker at the same
        if(di1==di2 and state[di1] * self.color <= 1):
            return state,-20
        #can not play from other than BAR if there is a checker in BAR
        if(state[bari] != 0 and (di1 != bari and di2 != bari)):
            return state,-20
        #if BAR has checker and can not play to destination it is not invalid. do not punish
        if(state[bari] != 0 and (state[di1-d1] * self.color < -1 or state[di2-d2] * self.color < -1)):
            return state, 0
        #can not play to location where opponent's has more than 1 checkers.
        if(state[di1-d1] * self.color < -1 or state[di2-d2] * self.color < -1):
            return state,-20        

        #advance state
        reward = 20
        state[di1] -= self.color
        if(state[di1-d1] * self.color == -1):
            state[di1-d1] = 0
            state[self._getLOpponentBARi()] -= self.color
            reward += 5
        state[di1-d1] += self.color
        #TODO: check if checkers must be collected or not according to if checkers are at the home or not.
        state[di2] -= self.color
        if(state[di2-d2] * self.color == -1):
            state[di2-d2] = 0
            state[self._getLOpponentBARi()] -= self.color
            reward += 5
        state[di2-d2] += self.color
        

        #punish if target location has only one checker
        if(abs(state[di1-d1]) == 1):
            reward -= 3
        if(abs(state[di2-d2]) == 1):
            reward -= 3
        if(abs(state[di1]) == 1):
            reward -= 3
        if(abs(state[di2]) == 1):
            reward -= 3


        #declaring win when all chekers at home.TODO: check and give max point all checkers are collected. win game.
        t1 = self._getTotalCheckersAtHome(state)
        if(t1 == 15):
            reward = 50

        return state,reward

    def playAll(self,state):
        y = np.empty((26,26))        
        for i in range(26):
            for j in range(26):
                _,reward = self.play(state,(i,j))
                y[i][j] = reward
        return y


    def get_next_action(self, state):
        if random.random() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        return tuple(np.argmax(self.getQ(state),axis=1))
    def random_action(self):
        return (random.randint(0,25),random.randint(0,25))
    def best_action(self,state):
        imax = np.argmax(self.playAll(state))        
        return ( int((imax-(imax%26))/26) ,int(imax%26))

    def getQ(self,state):
        state_to_predict = np.expand_dims(state,0)
        action_prediction = self.model.predict(state_to_predict)
        return action_prediction[0]

    def train(self, old_state, new_state, action, reward):
        
        old_state_prediction = self.getQ(old_state)
        new_state_prediction = self.getQ(new_state)
        a1,a2 = action

        old_state_prediction[0][a1] = reward + self.discount * np.amax(new_state_prediction[0])
        old_state_prediction[1][a2] = reward + self.discount * np.amax(new_state_prediction[1])

        x = np.expand_dims(old_state,0)
        y = np.expand_dims(old_state_prediction,0)
        self.model.fit(x,y,verbose=0)

    def update(self, old_state, new_state, action, reward):        
        self.train(old_state, new_state, action, reward)
        self.exploration_rate *= self.decay_factor
        #self.saveModel()

    def saveModel(self):
        self.model.save(self._getModelFilename())
        wandb.save(self._getModelFilename())

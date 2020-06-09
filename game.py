import numpy as np
from IPython.display import clear_output
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import wandb
from agent import agent

class environment:

    def __init__(self):
        wandb.init(project="tavla2")
        self.config = wandb.config
        self.white_agent = agent(color=1,discount=self.config['discount'],exploration_rate=self.config['exploration_rate'],decay_factor=self.config['decay_factor'])
        self.black_agent = agent(color=-1,discount=self.config['discount'],exploration_rate=self.config['exploration_rate'],decay_factor=self.config['decay_factor'])
        self.white_max_reward = 0
        self.black_max_reward = 0
        self.white_tot_reward = 0
        self.black_tot_reward = 0
        self.total_penalty = 0
        self.total_valid = 0
        self.episode = self.config['episode']        
        self._initGame()


    def _initGame(self):
        self.white_reward = 0
        self.black_reward = 0
        self.state = [0,  -2, 0, 0, 0, 0,+5,   0,+3, 0, 0, 0,-5,   +5, 0, 0, 0, -3, 0,  -5, 0, 0, 0, 0,+2,  0,   4,6]  #initial sate
        #self.state = [0,   4, 0, 6, 0, 0, 0,   1, 1, 0, 0, 0, 0,    0, 0, 0, 1, -1, 1,   1, 0, 0, 0, 0,-14, 0,   2,3]  #test state
        self.turn = random.randrange(-1,2,2) #The first round is decded number (-1 or 1)
    
    def roll(self):
        self.state[26] = random.randint(1,6)
        self.state[27] = random.randint(1,6)


    def start(self):
        for game_no in range(1,self.episode+1):
            reward = 0
            self._initGame()
            self.roll()
            i = 1
            while reward != 50:
                #self.render()
                #self._plot()
                if(self.turn == 1):
                    if( np.amax(self.white_agent.playAll(self.state)) < 0 ):
                        break
                    action_to_play = self.white_agent.get_next_action(self.state)
                    new_state, reward = self.white_agent.play(self.state, action_to_play)
                    self.white_agent.update(old_state=self.state,new_state=new_state,action=action_to_play,reward=reward)
                    self.state = new_state
                    self.white_reward += reward
                    self.white_tot_reward += reward
                    self.white_max_reward = max(self.white_max_reward,self.white_reward)
                if(self.turn == -1):
                    if( np.amax(self.black_agent.playAll(self.state)) < 0 ):
                        break
                    action_to_play = self.black_agent.get_next_action(self.state)
                    new_state, reward = self.black_agent.play(self.state, action_to_play)
                    self.black_agent.update(old_state=self.state,new_state=new_state,action=action_to_play,reward=reward)
                    self.state = new_state
                    self.black_reward += reward
                    self.black_tot_reward += reward
                    self.black_max_reward = max(self.black_max_reward,self.black_reward)
                #print(action_to_play)
                #self.render()            
                #print(reward)
                #print(self.state)
                if(reward >= 0):
                    self.turn *= -1
                    self.roll()
                    i = 1
                    self.total_valid += 1
                else:
                    i += 1
                    self.total_penalty += 1
                    break

            metrics = {
                'full-game' : (game_no-self.total_penalty),
                'full-game-rate' : ((game_no-self.total_penalty)/game_no),
                'valid-total' : self.total_valid,
                'valid-avarage' : self.total_valid/game_no,
                'exploration-rate-white' : self.white_agent.exploration_rate,
                'exploration-rate-black' : self.black_agent.exploration_rate,
                'max-reward-white' : self.white_max_reward,
                'max-reward-black' : self.black_max_reward,
                'tot-reward-white' : self.white_tot_reward,
                'tot-reward-black' : self.black_tot_reward,
                'avg-reward-white' : self.white_tot_reward/game_no,
                'avg-reward-black' : self.black_tot_reward/game_no,
                'reward-white':self.white_reward,
                'reward-black':self.black_reward
            }
            if game_no % 1000 == 0:
                clear_output(wait=True)
                print("Game           : %d"%game_no)
                print("Full Game      : %d - %f "%(metrics['full-game'],metrics['full-game-rate']))
                print("Valid          : %d - %f "%(metrics['valid-total'],metrics['valid-avarage']))
                print("Exp Rate       : \x1b[6;30;47m%f\x1b[0m \x1b[0;37;40m%f\x1b[0m" % (metrics['exploration-rate-white'],metrics['exploration-rate-black']))
                print("Max Rewards    : \x1b[6;30;47m%f\x1b[0m \x1b[0;37;40m%f\x1b[0m" % (metrics['max-reward-white'],metrics['max-reward-black']))
                print("Tot Rewards    : \x1b[6;30;47m%f\x1b[0m \x1b[0;37;40m%f\x1b[0m" % (metrics['tot-reward-white'],metrics['tot-reward-black']))
                print("Avg Rewards    : \x1b[6;30;47m%f\x1b[0m \x1b[0;37;40m%f\x1b[0m" % (metrics['avg-reward-white'],metrics['avg-reward-black']))
            wandb.log(metrics,step=game_no)

        self.white_agent.saveModel()
        self.black_agent.saveModel()
        
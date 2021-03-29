import os
import pickle
import random

import numpy as np
from scipy.special import softmax

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.mod = 0
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.array([.25, .25, .25, .25, 0, 0])#np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        self.mod = 0
    else:
        self.logger.info("Loading model from saved state.")
        self.mod = 1
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """


    self.T = 0.5
    self.epsilon = 0.1
    self.score = game_state['self'][1]
    self.step = game_state['step']

    random_prob = self.epsilon
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 20% wait. 0% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])

    # if not self.train and bombs_left and random.random() < random_prob:
    if not self.train  and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])#[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    if self.mod:
        qs = self.model[:,:-1].dot(state_to_features(game_state).T)
        #qs = self.model.dot(state_to_features(game_state).T)/np.sum(self.model.dot(state_to_features(game_state).T))
        T = self.T

        prob=np.squeeze(softmax(qs/T)).T
        prob = prob.tolist()+[0]+[0]
        return np.random.choice(ACTIONS, p=prob)#ACTIONS[np.argmax(qs)]#np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])#p=qs) #

    else:
       return np.random.choice(ACTIONS, p=self.model)#np.random.choice(ACTIONS, p=qs)###


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    arena = game_state['field']
    box_indices = (arena == 1) #delete boxes
    arena[box_indices] = 0

    _, score, bombs_left, (x, y) = game_state['self']

    # Tile values: coins
    coins = game_state['coins']

    #print(target)
    for coin in coins:
        arena[coin] = 5
        xc = coin[0]
        yc = coin[1]

        arena[(xc-1,yc)] = arena[(xc-1,yc)] + (arena[(xc-1,yc)]+1)
        arena[(xc+1,yc)] = arena[(xc+1,yc)] + (arena[(xc+1,yc)]+1)
        arena[(xc,yc-1)] = arena[(xc,yc-1)] + (arena[(xc,yc-1)]+1)
        arena[(xc,yc+1)] = arena[(xc,yc+1)] + (arena[(xc,yc+1)]+1)


    #Relative arena definition

    s = ((3*17),(3*17))
    arena_ext = -np.ones(s) #create a matrix with "mantinels"
    arena_ext[16:33,16:33] = arena #insert the real arena into the middle
    neigh = 7
    agent_neigh = np.array(arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)])


    channels = []
    neigharea = (2*neigh+1)**2
    channels = [agent_neigh.reshape(neigharea,1)]
    stacked_channels = np.vstack(channels)
    return stacked_channels.T

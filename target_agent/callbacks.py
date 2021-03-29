import os
import pickle
import random

from collections import deque
from random import shuffle

import numpy as np
from scipy.special import softmax

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
actions_first_task = ['UP', 'RIGHT', 'DOWN', 'LEFT']

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
    self.step = 0
    self.exploded_boxes = 0
    if True: #self.train:
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 7)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0
    self.savedecision = 0
    self.form = 'SELF'
    self.mod = 0
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights =np.array([.2, .2, .2, .2, .15, .05])#.25, .25, .25, .25, 0, 0])#np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        self.mod = 0
    else:
        self.logger.info("Loading model from saved state.")
        self.mod = 1
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            if not self.train: print(self.model)
            #betas = self.model.betas

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """


    _, score, bombs_left, (x, y) = game_state['self']
    self.step = game_state['step']

    random_prob = 0.1
    if not self.train and bombs_left and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])#p=[., .1, .1, .1, .1, .5])#)

    self.logger.debug("Querying model for action.")


    #if there is model to be loaded, approximates Q values and does decision by softmax
    if self.mod:

        qs = self.model[:,:-1].dot(state_to_features(game_state).T)#/np.sum(np.squeeze(self.model[:,:-1].dot(state_to_features(game_state).T)))
        _, score, bombs_left, (x, y) = game_state['self']
        T = 0.4+0.15*(bombs_left)

        prob=np.squeeze(softmax(qs/T)).T
        prob = prob.tolist()

        choice = np.random.choice(ACTIONS, p=prob)
        accept =0

        return choice
        #if there is no model found, does random decision
    else:

     choice = np.random.choice(ACTIONS, p=self.model)
     return choice


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

    arena = game_state['field'].copy()
    box_indices = (arena == 1) #delete boxes
    arena[box_indices] = 0

    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]


    s = ((3*17),(3*17))
    arena_ext = -np.ones(s) #create a matrix with "mantinels"
    arena_ext[16:33,16:33] = arena #insert the real arena into the middle
    neigh = 1
    neigharea = (2*neigh+1)**2



    arena = game_state['field'].copy()

    arena_ext[16:33,16:33] = arena
    agent_neigh_bomb = np.zeros(17)

    #detect if bombs are threatening agent
    if bombs:

        for bomb in bomb_xys:
            arena[bomb] = -100
        arena_ext[16:33,16:33] = arena
        agent_neigh_bomb[0:9] = (arena[(x-1):(x+2),(y-1):(y+2)]==-100).reshape(9)
        for i in [2,3]:
            if arena_ext[(16+x+i,16+y)] == -100:
                agent_neigh_bomb[7+i] = 1
            if arena_ext[(16+x-i,16+y)] == -100:
                agent_neigh_bomb[i+9] = 1
            if arena_ext[(16+x,16+y+i)] == -100:
                agent_neigh_bomb[i+11] = 1
            if arena_ext[(16+x,16+y-i)] == -100:
                agent_neigh_bomb[i+13] = 1

    arena_ext = -np.ones(s)
    arena = game_state['field'].copy()
    arena_ext[16:33,16:33] = arena
    potential_neigh = 5
    potential_neigh_matrix = np.array(arena_ext[(16+x-potential_neigh):(17+x+potential_neigh),(16+y-potential_neigh):(17+y+potential_neigh)])


    epsilon = 1e-2
    lateral_crates_potential = 7
    center_mat = potential_neigh
    bomb_potential = -np.sum(np.sum(np.abs(arena_ext[(16+x-1):(17+x+1),(16+y-1):(17+y+1)])))

    scape_potential = 0
    scape_potential_temp = 0
    wall_value = -1
    crate_value = 1


    #compute bomb potential
    for i in range (1,4):
        if potential_neigh_matrix[center_mat+i,center_mat] == wall_value: break
        else:
            if (np.abs(potential_neigh_matrix[center_mat+i,center_mat] - crate_value)) < epsilon:
                bomb_potential += lateral_crates_potential
                scape_potential_temp = -1
            else:
                if ((potential_neigh_matrix[center_mat+i,center_mat+1]==0) or (potential_neigh_matrix[center_mat+i,center_mat-1]==0)) and scape_potential_temp > -1:
                    scape_potential = 1
                if (i == 3) and potential_neigh_matrix[center_mat+i+1,center_mat] != wall_value and (np.abs(potential_neigh_matrix[center_mat+i+1,center_mat] - crate_value)) > epsilon and scape_potential_temp > -1: scape_potential = 1

    scape_potential_temp = 0
    for i in range (1,4):
        if potential_neigh_matrix[center_mat-i,center_mat] == wall_value: break
        else:
            if (np.abs(potential_neigh_matrix[center_mat-i,center_mat] - crate_value)) < epsilon:
                bomb_potential += lateral_crates_potential
                scape_potential_temp = -1
            else:
                if ((potential_neigh_matrix[center_mat-i,center_mat+1]==0) or (potential_neigh_matrix[center_mat-i,center_mat-1]==0)) and scape_potential_temp > -1:
                    scape_potential = 1
                if (i == 3) and potential_neigh_matrix[center_mat-i-1,center_mat] != wall_value and (np.abs(potential_neigh_matrix[center_mat-i-1,center_mat] - crate_value)) > epsilon and scape_potential_temp > -1: scape_potential = 1

    scape_potential_temp = 0
    for i in range (1,4):
        if potential_neigh_matrix[center_mat,center_mat+i] == wall_value:

            break
        else:
            if (np.abs(potential_neigh_matrix[center_mat,center_mat+i] - crate_value)) < epsilon:
                bomb_potential += lateral_crates_potential

                scape_potential_temp = -1
            else:
                if ((potential_neigh_matrix[center_mat+1,center_mat+i]==0) or (potential_neigh_matrix[center_mat-1,center_mat+i]==0)) and scape_potential_temp > -1:
                    scape_potential = 1

                if (i == 3) and potential_neigh_matrix[center_mat,center_mat+i+1] != wall_value and (np.abs(potential_neigh_matrix[center_mat,center_mat+i+1] - crate_value)) > epsilon and scape_potential_temp > -1:
                    scape_potential = 1

    scape_potential_temp = 0
    for i in range (1,4):
        if potential_neigh_matrix[center_mat,center_mat-i] == wall_value: break
        else:
            if (np.abs(potential_neigh_matrix[center_mat,center_mat-i] - crate_value)) < epsilon:
                bomb_potential += lateral_crates_potential
                scape_potential_temp = -1
            else:
                if ((potential_neigh_matrix[center_mat+1,center_mat-i]==0) or (potential_neigh_matrix[center_mat-1,center_mat-i]==0)) and scape_potential_temp > -1:
                    scape_potential = 1
                if (i == 3) and potential_neigh_matrix[center_mat,center_mat-i-1] != wall_value and (np.abs(potential_neigh_matrix[center_mat,center_mat-i-1] - crate_value)) > epsilon and scape_potential_temp > -1: scape_potential = 1


    if bombs_left == 0 or scape_potential < 1: bomb_potential = -5

    arena = game_state['field'].copy()
    coins = game_state['coins']
    crate_indices = (arena == 1)
    arena[crate_indices] = -1

    #find nearest coin
    dist = []
    for coin in coins:

        xc = coin[0]
        yc = coin[1]
        dist.append(np.linalg.norm((np.array(coin)-np.array((x,y))), ord=1))
    if coins:
        nearestcoin = np.argmin(dist)
        targetcoin = np.array(coins[nearestcoin]-np.array((x,y)))
    else:
        targetcoin = np.zeros(2)

    lingering_indices = (game_state['explosion_map'] ==1)
    linger = np.sum(lingering_indices)
    arena[lingering_indices] = -2
    arena_ext[16:33,16:33] = arena


    #find the 4-neighborhood
    agent_neigh_close = np.array(arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)]).reshape(neigharea).tolist()
    agent_neigh_closed = agent_neigh_close.copy()
    for i in range(5):
        agent_neigh_close.remove(agent_neigh_closed[2*i])

    agent_neigh_close = np.array(agent_neigh_close)
    safe_neigh = 5
    arena = -game_state['field'].copy()
    crate_indices = (arena==-1)

    arena[crate_indices] = 1

    arena_ext = -arena_ext
    arena_ext[16:33,16:33] = arena

    #if threatened by bomb, find directions to safety
    safeplaces = []
    if np.sum(agent_neigh_bomb) > 0:
        flag = 0
        for bomb in bomb_xys:
            xb, yb = bomb
            safepoint_matrix = np.array(arena_ext[(16+xb-safe_neigh):(17+xb+safe_neigh),(16+yb-safe_neigh):(17+yb+safe_neigh)])
            center_mat = safe_neigh

            scape_potential_temp = 0
            for i in range (1,4):
                if flag ==1: break
                if safepoint_matrix[center_mat,center_mat-i] == 1:
                    break
                else:
                     if (np.abs(potential_neigh_matrix[center_mat,center_mat-i] - 1)) < epsilon:

                         scape_potential_temp = -1
                     else:
                         if (safepoint_matrix[center_mat+1,center_mat-i])<epsilon and scape_potential_temp > -1:

                             scape_potential = 1
                             safeplaces.append((xb+1,yb-i))

                             flag = 1
                         if (safepoint_matrix[center_mat-1,center_mat-i])<epsilon and scape_potential_temp > -1:
                             scape_potential = 1

                             safeplaces.append((xb-1,yb-i))
                             flag = 1
                         if (i == 3) and safepoint_matrix[center_mat,center_mat-i-1] != 1 and scape_potential_temp > -1:
                             scape_potential = 1

                             safeplaces.append((xb,yb-i-1))
                             flag = 1
            scape_potential_temp = 0
            for i in range (1,4):
                   if flag ==1: break
                   if safepoint_matrix[center_mat-i,center_mat] == 1:
                       break
                   else:
                        if (np.abs(potential_neigh_matrix[center_mat-i,center_mat] - 1)) < epsilon:

                            scape_potential_temp = -1
                        else:
                            if (safepoint_matrix[center_mat-i,center_mat+1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb-i,yb+1))
                                flag = 1
                            if (safepoint_matrix[center_mat-i,center_mat-1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb-i,yb+1))
                                flag = 1
                            if (i == 3) and safepoint_matrix[center_mat-i-1,center_mat] != 1 and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb,yb-i-1))
                                flag = 1
            scape_potential_temp = 0
            for i in range (1,4):
                   if flag ==1: break
                   if safepoint_matrix[center_mat,center_mat+i] == 1:
                       break
                   else:
                        if (np.abs(potential_neigh_matrix[center_mat,center_mat+i] - 1)) < epsilon:

                            scape_potential_temp = -1
                        else:
                            if (safepoint_matrix[center_mat+1,center_mat+i])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb+1,yb+i))
                                flag = 1
                            if (safepoint_matrix[center_mat-1,center_mat+i])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb-1,yb+i))
                                flag = 1
                            if (i == 3) and safepoint_matrix[center_mat,center_mat+i+1] != 1 and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb,yb+i+1))
                                flag = 1
            scape_potential_temp = 0
            for i in range (1,4):
                   if flag ==1: break
                   if safepoint_matrix[center_mat+i,center_mat] == 1:
                       break
                   else:
                        if (np.abs(potential_neigh_matrix[center_mat+i,center_mat] - 1)) < epsilon:

                            scape_potential_temp = -1
                        else:
                            if (safepoint_matrix[center_mat+i,center_mat+1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb+i,yb+1))
                                flag = 1
                            if (safepoint_matrix[center_mat+i,center_mat-1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb+i,yb-1))
                                flag = 1
                            if (i == 3) and safepoint_matrix[center_mat+i+1,center_mat] != 1 and scape_potential_temp > -1:
                                scape_potential = 1

                                safeplaces.append((xb+i+1,yb))
                                flag = 1

    dist = []
    for place in safeplaces:
        dist.append(np.linalg.norm((np.array(place)-np.array((x,y))), ord=1))
    if safeplaces:
        nearestplace = np.argmin(dist)

        safeway = np.array(safeplaces[nearestplace]-np.array((x,y)))
        safereached = 0
        if np.min(dist) < 0.5:
            safereached = 1
    else:
        safeway = np.zeros(2)
        safereached = 0
    if np.sum(lingering_indices) > 0.5:
        safereached = 1


    channels = []


    channels = [np.sign(targetcoin.reshape(2).T), np.sign(safeway), agent_neigh_close, bomb_potential]

    stacked_channels = np.hstack(channels)

    return stacked_channels.T

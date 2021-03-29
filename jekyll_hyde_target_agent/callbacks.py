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

    self.tau = game_state['new_parameters'][0]
    self.gamma = game_state['new_parameters'][1]
    self.T = game_state['new_parameters'][2]
    self.epsilon = game_state['new_parameters'][3]
    self.alpha_0 = game_state['new_parameters'][4]
    _, score, bombs_left, (x, y) = game_state['self']
    self.step = game_state['step']
    #print(x,y)
    #print('oooooooooooooooooooooooo')
    #print('oooooooooooooooooooooooo')
    if game_state['step']==1:
        if  x == 1:
            if y ==1:
                self.orient = 1
            else:
                self.orient = 2
        else:
            if y==1:
                self.orient = 3
            else:
                self.orient = 4

    #print(self.orient)
    # randint = np.random.randint(9)
    # if self.train and game_state['step']<8 and self.orient==1:
    #     return ['RIGHT', 'BOMB', 'LEFT', 'DOWN', 'WAIT', 'WAIT', 'BOMB', 'UP', 'RIGHT', 'WAIT'][game_state['step']-1]
    # if self.train and game_state['step']<8 and self.orient==2:
    #     return ['RIGHT', 'BOMB', 'LEFT', 'UP', 'WAIT', 'WAIT', 'BOMB', 'DOWN', 'RIGHT', 'WAIT'][game_state['step']-1]
    # if self.train and game_state['step']<8 and self.orient==3:
    #     return ['DOWN','BOMB', 'UP', 'LEFT', 'WAIT','WAIT','BOMB'][game_state['step']-1]
    # if self.train and game_state['step']<8 and self.orient==4:
    #     return ['LEFT', 'BOMB', 'RIGHT', 'UP', 'WAIT', 'WAIT', 'BOMB', 'DOWN', 'LEFT', 'WAIT'][game_state['step']-1]
    # # todo Exploration vs exploitation
    if np.mod(game_state['step'],10) ==1:
        self.form = np.random.choice(['RULE', 'SELF'],p=[.5, .5])

    if not bombs_left:
        if self.savedecision == 0:
            self.save = np.random.choice(['RULE', 'SELF'],p=[1, 0])
            self.savedecision = 1
    else:
        self.save = 'SELF'
        self.savedecision = 0
    #if self.train and self.savedecision ==1:
    #print(self.alpha_0)
    if self.alpha_0 > 0 and self.train and (self.form == 'RULE' or self.save == 'RULE'):
        #print('ruling')
        return act_ruled(self, game_state)


    #if not self.train and game_state['step']<20:
    #    return act_ruled(self, game_state)
    random_prob = 0.1
    if not self.train and bombs_left and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])#p=[., .1, .1, .1, .1, .5])#)

    self.logger.debug("Querying model for action.")



    if self.mod:
#        print(self.model)
        qs = self.model[:,:-1].dot(state_to_features(game_state).T)#/np.sum(np.squeeze(self.model[:,:-1].dot(state_to_features(game_state).T)))
        #print('Q values are', qs)
        #qs = self.model.dot(state_to_features(game_state).T)/np.sum(self.model.dot(state_to_features(game_state).T))
        _, score, bombs_left, (x, y) = game_state['self']
        T = 0.4+0.15*(bombs_left)
    #print(qs)
        prob=np.squeeze(softmax(qs/T)).T
        prob = prob.tolist()
    #print((prob))
    #while True:
    #      choice = np.random.choice(ACTIONS, p=prob)
    #      if choice =

        choice = np.random.choice(ACTIONS, p=prob)
        accept =0
#        print('Probabilities are', prob)
#        print('Original chosen', choice)
        # if not self.train:
        #     for i in range(100):
        #
        #         if accept == 1:
        #                 break
        #                 print(i)
        #         if choice in actions_first_task:
        #
        #                 cy=0
        #                 cx=0
        #                 if choice == 'UP': cy=-1
        #                 if choice == 'DOWN': cy=1
        #                 if choice == 'RIGHT': cx=1
        #                 if choice == 'LEFT': cx=-1
        #                 _, score, bombs_left, (x, y) = game_state['self']
        #                 bombs = game_state['bombs']
        #                 lingering_indices = (game_state['explosion_map'] ==1)
        #                 bomb_xys = [xy for (xy, t) in bombs]
        #                 obstacles = game_state['field'].copy()
        #                 for bomb in bombs:
        #                     obstacles[bomb] = 1
        #                 obstacles[lingering_indices] = 1
        #                 if obstacles[x+cx,y+cy] ==0:
        #                     accept = 1
        #                     print('Probabilities are', prob)
        #                     print('Action chosen', choice)
        #                     print(i)
        #                     #break
        #                 else:
        #                     print('invalid')
        #                     print(ACTIONS.index(choice))
        #                     qs[ACTIONS.index(choice)] = np.min(qs)-10
        #                     print(qs)
        #                     prob=np.squeeze(softmax(qs/T)).T
        #                     prob = prob.tolist()
        #                     choice = np.random.choice(ACTIONS, p=prob)
        #             #print((prob))
        #             #while True:
        #             #      choice = np.random.choice(ACTIONS, p=prob)
        #             #      if choice =
        #
        #
        #
        #         else: break
#        print('Probabilities are', prob)
#        print('Action chosen', choice)
        return choice#ACTIONS[np.argmax(qs)]#np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])#p=qs) #
    else:
     #  print(self.mod)
     choice = np.random.choice(ACTIONS, p=self.model)
#     print('Action chosen', choice)
     return choice#np.random.choice(ACTIONS, p=qs)###
    #

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
    #others = [xy for (n, s, b, xy) in game_state['others']]


    s = ((3*17),(3*17))
    arena_ext = -np.ones(s) #create a matrix with "mantinels"
    arena_ext[16:33,16:33] = arena #insert the real arena into the middle
    neigh = 1
    neigharea = (2*neigh+1)**2


    #agent_neigh_close = []
    #list_accepted = np.arrange(25).tolist()
    #list_accepted.remove(12)
    #for t in list_accepted:
    #    agent_neigh_close.append(agent_neigh[t])
    arena = game_state['field'].copy()

    arena_ext[16:33,16:33] = arena
    agent_neigh_box = (arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)]==-50).reshape(neigharea)*20
    agent_neigh_box += (arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)]==1).reshape(neigharea)
    agent_neigh_box = agent_neigh_box.tolist()
    agent_neigh_box.remove(agent_neigh_box[4])
    #for i in [-1,1]:
    #    if arena[(x+i,y)] == 1:
#            agent_neigh_box[1+i] = 1
#        if arena[(x,y+i)] == 1:
#            agent_neigh_box[2+i] = 1
    agent_neigh_bomb = np.zeros(17)
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
    #print('Bomb potential is', bomb_potential)
    scape_potential = 0
    scape_potential_temp = 0
    wall_value = -1
    crate_value = 1

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
            # print('wall below')
            break
        else:
            if (np.abs(potential_neigh_matrix[center_mat,center_mat+i] - crate_value)) < epsilon:
                bomb_potential += lateral_crates_potential
                # print('crate below for i = ', i)
                scape_potential_temp = -1
            else:
                if ((potential_neigh_matrix[center_mat+1,center_mat+i]==0) or (potential_neigh_matrix[center_mat-1,center_mat+i]==0)) and scape_potential_temp > -1:
                    scape_potential = 1
                    # print('free sides below for i = ', i)
                if (i == 3) and potential_neigh_matrix[center_mat,center_mat+i+1] != wall_value and (np.abs(potential_neigh_matrix[center_mat,center_mat+i+1] - crate_value)) > epsilon and scape_potential_temp > -1:
                    scape_potential = 1
                    # print('free last tile')

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

    # print(scape_potential)
    if bombs_left == 0 or scape_potential < 1: bomb_potential = -5
    #print('Escape potential is', scape_potential)
    #if bombs_left == 0 or scape_potential < 1: bomb_potential = -5

    #rint('Bomb potential is', bomb_potential)
        #agent_neigh_close = np.array(agent_neigh_close).reshape(4,1)
    #print(agent_neigh)
#    print('Bomb features are', agent_neigh_bomb)
    # For example, you could construct several channels of equal shape, ...
    arena = game_state['field'].copy()
    coins = game_state['coins']
    crate_indices = (arena == 1)
    arena[crate_indices] = -1
    dist = []
#    print('Position is', (x,y))
    for coin in coins:
        #arena[coin] = 5
        xc = coin[0]
        yc = coin[1]

        #arena[(xc-1,yc)] = 2*arena[(xc-1,yc)] + 1
        #arena[(xc+1,yc)] = 2*arena[(xc+1,yc)] + 1
        #arena[(xc,yc-1)] = 2*arena[(xc,yc-1)] + 1
        #arena[(xc,yc+1)] = 2*arena[(xc,yc+1)] + 1
        dist.append(np.linalg.norm((np.array(coin)-np.array((x,y))), ord=1))
    if coins:
        nearestcoin = np.argmin(dist)
#        print('Nearest coin is at', coins[nearestcoin])
        targetcoin = np.array(coins[nearestcoin]-np.array((x,y)))
    else:
        targetcoin = np.zeros(2)
#        print('There are no coins')
    lingering_indices = (game_state['explosion_map'] ==1)
    linger = np.sum(lingering_indices)
    arena[lingering_indices] = -2
    arena_ext[16:33,16:33] = arena
#     crate_indices = (arena == 1)
#     distbox = []
#     for crate in crate_indices:
#         distbox.append(np.linalg.norm((np.array(box)-np.array((x,y))), ord=1))
#     if :
#         nearestcoin = np.argmin(dist)
# #        print('Nearest coin is at', coins[nearestcoin])
#         targetcoin = np.array(coins[nearestcoin]-np.array((x,y)))
#     else:
#         targetcoin = np.zeros(2)


    agent_neigh_close = np.array(arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)]).reshape(neigharea).tolist()
    agent_neigh_closed = agent_neigh_close.copy()
    for i in range(5):
        agent_neigh_close.remove(agent_neigh_closed[2*i])
        #print(2*i)
    agent_neigh_close = np.array(agent_neigh_close)
    safe_neigh = 5
    arena = -game_state['field'].copy()
    crate_indices = (arena==-1)
    #print(crate_indices)
    arena[crate_indices] = 1
    #print(arena)
    arena_ext = -arena_ext
    arena_ext[16:33,16:33] = arena

    safeplaces = []
    if np.sum(agent_neigh_bomb) > 0:
        flag = 0
        for bomb in bomb_xys:
            xb, yb = bomb
            safepoint_matrix = np.array(arena_ext[(16+xb-safe_neigh):(17+xb+safe_neigh),(16+yb-safe_neigh):(17+yb+safe_neigh)])
            center_mat = safe_neigh
            #print(safepoint_matrix)
            #print(safepoint_matrix[center_mat, center_mat])
            scape_potential_temp = 0
            for i in range (1,4):
                if flag ==1: break
                if safepoint_matrix[center_mat,center_mat-i] == 1:
                    break
                else:
                     if (np.abs(potential_neigh_matrix[center_mat,center_mat-i] - 1)) < epsilon:
                         #bomb_potential += lateral_crates_potential
                         scape_potential_temp = -1
                     else:
                         if (safepoint_matrix[center_mat+1,center_mat-i])<epsilon and scape_potential_temp > -1:
                             #print('safe value', safepoint_matrix[center_mat+1,center_mat-i])
                             scape_potential = 1
                             safeplaces.append((xb+1,yb-i))
                             #print('a')
                             flag = 1
                         if (safepoint_matrix[center_mat-1,center_mat-i])<epsilon and scape_potential_temp > -1:
                             scape_potential = 1
                             #print('b')
                             safeplaces.append((xb-1,yb-i))
                             flag = 1
                         if (i == 3) and safepoint_matrix[center_mat,center_mat-i-1] != 1 and scape_potential_temp > -1:
                             scape_potential = 1
                             #print('c')
                             safeplaces.append((xb,yb-i-1))
                             flag = 1
            scape_potential_temp = 0
            for i in range (1,4):
                   if flag ==1: break
                   if safepoint_matrix[center_mat-i,center_mat] == 1:
                       break
                   else:
                        if (np.abs(potential_neigh_matrix[center_mat-i,center_mat] - 1)) < epsilon:
                            #bomb_potential += lateral_crates_potential
                            scape_potential_temp = -1
                        else:
                            if (safepoint_matrix[center_mat-i,center_mat+1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('d')
                                safeplaces.append((xb-i,yb+1))
                                flag = 1
                            if (safepoint_matrix[center_mat-i,center_mat-1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('e')
                                safeplaces.append((xb-i,yb+1))
                                flag = 1
                            if (i == 3) and safepoint_matrix[center_mat-i-1,center_mat] != 1 and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('f')
                                safeplaces.append((xb,yb-i-1))
                                flag = 1
            scape_potential_temp = 0
            for i in range (1,4):
                   if flag ==1: break
                   if safepoint_matrix[center_mat,center_mat+i] == 1:
                       break
                   else:
                        if (np.abs(potential_neigh_matrix[center_mat,center_mat+i] - 1)) < epsilon:
                            #bomb_potential += lateral_crates_potential
                            scape_potential_temp = -1
                        else:
                            if (safepoint_matrix[center_mat+1,center_mat+i])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('g')
                                safeplaces.append((xb+1,yb+i))
                                flag = 1
                            if (safepoint_matrix[center_mat-1,center_mat+i])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('h')
                                safeplaces.append((xb-1,yb+i))
                                flag = 1
                            if (i == 3) and safepoint_matrix[center_mat,center_mat+i+1] != 1 and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('i')
                                safeplaces.append((xb,yb+i+1))
                                flag = 1
            scape_potential_temp = 0
            for i in range (1,4):
                   if flag ==1: break
                   if safepoint_matrix[center_mat+i,center_mat] == 1:
                       break
                   else:
                        if (np.abs(potential_neigh_matrix[center_mat+i,center_mat] - 1)) < epsilon:
                            #bomb_potential += lateral_crates_potential
                            scape_potential_temp = -1
                        else:
                            if (safepoint_matrix[center_mat+i,center_mat+1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('j')
                                safeplaces.append((xb+i,yb+1))
                                flag = 1
                            if (safepoint_matrix[center_mat+i,center_mat-1])<epsilon and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('k')
                                safeplaces.append((xb+i,yb-1))
                                flag = 1
                            if (i == 3) and safepoint_matrix[center_mat+i+1,center_mat] != 1 and scape_potential_temp > -1:
                                scape_potential = 1
                                #print('l')
                                safeplaces.append((xb+i+1,yb))
                                flag = 1

    dist = []
    for place in safeplaces:
        dist.append(np.linalg.norm((np.array(place)-np.array((x,y))), ord=1))
    if safeplaces:
        nearestplace = np.argmin(dist)
#        print('Nearest coin is at', coins[nearestcoin])
        safeway = np.array(safeplaces[nearestplace]-np.array((x,y)))
        safereached = 0
        if np.min(dist) < 0.5:
            safereached = 1
    else:
        safeway = np.zeros(2)
        safereached = 0
    if np.sum(lingering_indices) > 0.5:
        safereached = 1

    #if arena[np.int(x+safeway[0]),y] != 0: safeway[0] = 0
    #if arena[x,np.int(y+safeway[1])] != 0: safeway[1] = 0
    #if np.sum(safeway) !=0: print('Way to safety is', safeway)
    channels = []
    #channels.append(x,y,arena.reshape(17*17,1), )

    #print(np.sign(safeway))
    #channels = [agent_neigh.reshape(neigharea,1),coins.reshape(2*numcoins,1),np.zeros((2*(9-numcoins),1))]

    channels = [np.sign(targetcoin.reshape(2).T), np.sign(safeway), agent_neigh_close, safereached, bomb_potential]
    #channels = [x,y,arena.reshape(17*17,1),coins.reshape(2*numcoins,1),np.zeros((2*(9-numcoins),1))]
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.hstack(channels)
#    print('Features sent are', (stacked_channels.T))
    # and return them as a vector
    #print(np.size(stacked_channels.T))
    #print('Bomb potential is', bomb_potential)
    return stacked_channels.T#stacked_channels.reshape(-1)

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def act_ruled(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    #random_prob = 0.02
    #if self.train and random.random() < random_prob:
    #    self.logger.debug("Choosing action purely at random.")
    #    # 80%: walk in any direction. 10% wait. 10% bomb.
    #    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .199, .001])#p=[., .1, .1, .1, .1, .5])#)
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a

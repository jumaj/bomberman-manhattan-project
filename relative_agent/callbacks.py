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
    self.step = 0
    self.mod = 0
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.array([.25, .25, .25, .25, 0, 0])
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
    # todo Exploration vs exploitation
    random_prob = 0.1 #epsilon
    self.step = game_state['step']
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    if self.mod:
        qs = self.model[:,:-1].dot(state_to_features(game_state).T)#/np.sum(self.model[:,:-1].dot(state_to_features(game_state).T))

        T = 0.5*10 #Temperature
        prob=np.squeeze(softmax(qs/T)).T
        prob = prob.tolist()
        action = np.random.choice(ACTIONS, p=prob)
        return action #ACTIONS[np.argmax(qs)]#np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])#p=qs) #
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

    arena = game_state['field'].copy()
    #Tile values
    coin_value = 5*10
    crate_value = -0.7*10
    bomb_value = -3*10
    wall_value = -1*10


    _, score, bombs_left, (x, y) = game_state['self']


    #Tile values:coins and surrounding


    coins = game_state['coins']
    box_indices = (arena == 1)

    wall_indices = (arena ==-1)
    arena[wall_indices] = wall_value


    for coin in coins:
        arena[coin] = coin_value
        xc = coin[0]
        yc = coin[1]

        arena[(xc-1,yc)] = arena[(xc-1,yc)] + (arena[(xc-1,yc)]+10) #Gives +1 to the potential of coins
        arena[(xc+1,yc)] = arena[(xc+1,yc)] + (arena[(xc+1,yc)]+10)
        arena[(xc,yc-1)] = arena[(xc,yc-1)] + (arena[(xc,yc-1)]+10)
        arena[(xc,yc+1)] = arena[(xc,yc+1)] + (arena[(xc,yc+1)]+10)

    #Tile values: boxes, bombs and explosions

    arena[box_indices] = crate_value
    # print(type(arena))

    lingering_indices = (game_state['explosion_map'] == 1)
    arena[lingering_indices] = bomb_value #I could not check this is correct yet. Do so when the agents stops killing himself

    bombs = game_state['bombs']
    for bomb in bombs:
        arena[bomb[0]] = bomb_value
        #print(bomb[0])
        xc = bomb[0][0]
        yc = bomb[0][1]
        # print(xc, yc)


    #New way of calculating tile values for bombs
    bombs = game_state['bombs']
    for bomb in bombs:
        arena[bomb[0]] = bomb_value
        #print(bomb[0])
        xc = bomb[0][0]
        yc = bomb[0][1]
        # print(xc, yc)

        #Potential for bombs in the tiles of the explosions
        val_dir = 0
        for i in range(1, 4):
            if arena[(xc-i,yc)] == wall_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i  #I think the for later changes also the present tile
                if val_dir == 0:
                     for j in range(i):
                         arena[(xc-j,yc)] = bomb_value - 5*j
                break
            elif arena[(xc-i,yc)] == crate_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i
                if val_dir == 0:
                    for j in range(i):
                        arena[(xc-j,yc)] = bomb_value - 5*j
                for j in range(3-i):
                    arena[(xc-i-j,yc)] = bomb_value - 5*(j+i)
                    #if arena[(xc+i+j,yc)] == wall_value: break #This should solve bugs when there is a wall after a crate
                break
            else:
                if (arena[(xc-i,yc+1)] + arena[(xc-i,yc-1)]) > 2*crate_value:
                    val_dir = 1
                arena[(xc-i,yc)] = bomb_value + (20+i)
                if i == 3 and (arena[(xc-i-1,yc)] == wall_value or arena[(xc-i-1,yc)] == crate_value) and val_dir == 0:
                    # print('entered if of last tile direction xc - i')
                    for j in range(i+1):
                         arena[(xc-j,yc)] = bomb_value - 5*j

        val_dir = 0
        for i in range(1, 4):
            if arena[(xc+i,yc)] == wall_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i  #I think the for later changes also the present tile
                if val_dir == 0:
                     for j in range(i):
                         arena[(xc+j,yc)] = bomb_value - 5*j
                break
            elif arena[(xc+i,yc)] == crate_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i
                if val_dir == 0:
                    for j in range(i):
                        arena[(xc+j,yc)] = bomb_value - 5*j
                for j in range(3-i): #perhaps 3 (?)
                    arena[(xc+i+j,yc)] = bomb_value - 5*(j+i)
                    #if arena[(xc+i+j,yc)] == wall_value: break
                break
            else:
                if (arena[(xc+i,yc+1)] + arena[(xc+i,yc-1)]) > 2*crate_value:
                    val_dir = 1
                arena[(xc+i,yc)] = bomb_value + (20+i)
                if i == 3 and (arena[(xc+i+1,yc)] == wall_value or arena[(xc+i+1,yc)] == crate_value) and val_dir == 0:
                    # print('entered if of last tile direction xc + i')
                    for j in range(i+1):
                         arena[(xc+j,yc)] = bomb_value - 5*j

        val_dir = 0
        for i in range(1, 4):
            if arena[(xc,yc-i)] == wall_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i  #I think the for later changes also the present tile
                if val_dir == 0:
                     for j in range(i):
                         arena[(xc,yc-j)] = bomb_value - 5*j
                break
            elif arena[(xc,yc-i)] == crate_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i
                if val_dir == 0:
                    for j in range(i):
                        arena[(xc,yc-j)] = bomb_value - 5*j
                for j in range(3-i):
                    arena[(xc,yc-j-i)] = bomb_value - 5*(j+i)
                    #if arena[(xc+i+j,yc)] == wall_value: break
                break
            else:
                if arena[(xc+1,yc-i)] + arena[(xc-1,yc-i)] > 2*crate_value:
                    val_dir = 1
                arena[(xc,yc-i)] = bomb_value + (20+i)
                if i == 3 and (arena[(xc,yc-i-1)] == wall_value or arena[(xc,yc-i-1)] == crate_value) and val_dir == 0:
                    # print('entered if of last tile direction yc - i')
                    for j in range(i+1):
                         arena[(xc,yc-j)] = bomb_value - 5*j


        val_dir = 0
        for i in range(1, 4):
            if arena[(xc,yc+i)] == wall_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i  #I think the for later changes also the present tile
                if val_dir == 0:
                     for j in range(i):
                         arena[(xc,yc+j)] = bomb_value - 5*j
                break
            elif arena[(xc,yc+i)] == crate_value:
                #arena[(xc-i,yc)] = bomb_value - 10*i
                if val_dir == 0:
                    for j in range(i):
                        arena[(xc,yc+j)] = bomb_value - 5*j
                for j in range(3-i):
                    arena[(xc,yc+j+i)] = bomb_value - 5*(j+i)
                    #if arena[(xc+i+j,yc)] == wall_value: break
                break
            else:
                if (arena[(xc+1,yc+i)] + arena[(xc-1,yc+i)]) > 2*crate_value:
                    val_dir = 1
                arena[(xc,yc+i)] = bomb_value + (20+i)
                if i == 3 and (arena[(xc,yc+i+1)] == wall_value or arena[(xc,yc+i+1)] == crate_value) and val_dir == 0:
                    # print('entered if of last tile direction yc + i')
                    for j in range(i+1):
                         arena[(xc,yc+j)] = bomb_value - 5*j


    #Relative arena definition (after all tiles have their values)
    s = ((3*17),(3*17))
    arena_ext = -10*np.ones(s) #create a matrix with "mantinels"
    arena_ext[16:33,16:33] = arena #insert the real arena into the middle
    neigh = 5
    agent_neigh = np.array(arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)])


    ##Bomb potential feature
    #+5 for lateral crates
    #-30 if not bombs left, no scape or nothing to win

    potential_neigh = 5
    potential_neigh_matrix = np.array(arena_ext[(16+x-potential_neigh):(17+x+potential_neigh),(16+y-potential_neigh):(17+y+potential_neigh)])
    potential_neigh_matrix[potential_neigh,potential_neigh] = 0
    #Matrix of the value with which each tile contributes to the potential


    epsilon = 1e-2
    lateral_crates_potential = 0.5*10
    center_mat = potential_neigh
    bomb_potential = 0
    scape_potential = 0
    scape_potential_temp = 0

    for i in range (1,4):
        if potential_neigh_matrix[center_mat+i,center_mat] == wall_value: break
        else:
            if (np.abs(potential_neigh_matrix[center_mat+i,center_mat] - crate_value)) < epsilon:
                bomb_potential += lateral_crates_potential
                scape_potential_temp = -1
            else:
                if (potential_neigh_matrix[center_mat+i,center_mat+1] + potential_neigh_matrix[center_mat+i,center_mat-1] > 2*crate_value) and scape_potential_temp > -1:
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
                if (potential_neigh_matrix[center_mat-i,center_mat+1] + potential_neigh_matrix[center_mat-i,center_mat-1] > 2*crate_value) and scape_potential_temp > -1:
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
                if (potential_neigh_matrix[center_mat+1,center_mat+i] + potential_neigh_matrix[center_mat-1,center_mat+i] > 2*crate_value) and scape_potential_temp > -1:
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
                if (potential_neigh_matrix[center_mat+1,center_mat-i] + potential_neigh_matrix[center_mat-1,center_mat-i] > 2*crate_value) and scape_potential_temp > -1:
                    scape_potential = 1
                if (i == 3) and potential_neigh_matrix[center_mat,center_mat-i-1] != wall_value and (np.abs(potential_neigh_matrix[center_mat,center_mat-i-1] - crate_value)) > epsilon and scape_potential_temp > -1: scape_potential = 1

    # print(scape_potential)
    if bombs_left == 0 or scape_potential < 1 or bomb_potential == 0 : bomb_potential = -3*10
    # print('bomb potential', bomb_potential)

    channels = []
    neigharea = (2*neigh+1)**2
    channels = [bomb_potential, agent_neigh.reshape(neigharea,1)]
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.vstack(channels)
    # and return them as a vector
    return stacked_channels.T#stacked_channels.reshape(-1)

import os
import pickle
import random

from collections import deque
from random import shuffle

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
    if self.train:
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0

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

            #betas = self.model.betas

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.step = game_state['step']
    _, score, bombs_left, (x, y) = game_state['self']
    #print(x,y)
    # if game_state['step']==1:
    #     if  x == 1:
    #         if y ==1:
    #             self.orient = 1
    #         else:
    #             self.orient = 2
    #     else:
    #         if y==1:
    #             self.orient = 3
    #         else:
    #             self.orient = 4
    #
    # #print(self.orient)
    # if self.train and game_state['step']<8 and self.orient==1:
    #     return ['RIGHT', 'BOMB', 'LEFT', 'DOWN', 'WAIT', 'WAIT', 'BOMB', 'UP', 'RIGHT', 'WAIT'][game_state['step']-1]
    # if self.train and game_state['step']<8 and self.orient==2:
    #     return ['RIGHT', 'BOMB', 'LEFT', 'UP', 'WAIT', 'WAIT', 'BOMB', 'DOWN', 'RIGHT', 'WAIT'][game_state['step']-1]
    # if self.train and game_state['step']<8 and self.orient==3:
    #     return ['DOWN','BOMB', 'UP', 'LEFT', 'WAIT','WAIT','BOMB'][game_state['step']-1]
    # if self.train and game_state['step']<8 and self.orient==4:
    #     return ['LEFT', 'BOMB', 'RIGHT', 'UP', 'WAIT', 'WAIT', 'BOMB', 'DOWN', 'LEFT', 'WAIT'][game_state['step']-1]
    # # todo Exploration vs exploitation
    if np.mod(game_state['step'],20) ==1:
        self.form = np.random.choice(['RULE', 'SELF'],p=[0.3,0.7])
    if self.form == 'RULE':
        return act_ruled(self, game_state)
    random_prob = 0.2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .17, .03])#p=[., .1, .1, .1, .1, .5])#)

    self.logger.debug("Querying model for action.")



    if self.mod:
#        print(self.model)
        qs = self.model[:,:-1].dot(state_to_features(game_state).T)#/np.sum(np.squeeze(self.model[:,:-1].dot(state_to_features(game_state).T)))
        #qs = self.model.dot(state_to_features(game_state).T)/np.sum(self.model.dot(state_to_features(game_state).T))
        T = 0.7*10 #For the agent where everything is multiplied by 10
    #print(qs)
        prob=np.squeeze(softmax(qs/T)).T
        prob = prob.tolist()
        #print((prob))
        #while True:
        #      choice = np.random.choice(ACTIONS, p=prob)
        #      if choice =
        choice = np.random.choice(ACTIONS, p=prob)
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
    coin_value = 5*10
    crate_value = -0.7*10
    bomb_value = -3*10
    wall_value = -1*10
    around_crate_value = 1*10 #Not used at the moment

    _, score, bombs_left, (x, y) = game_state['self']
    #bombs = game_state['bombs']
    #bomb_xys = [xy for (xy, t) in bombs]
    #others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    box_indices = (arena == 1)

    wall_indices = (arena ==-1)
    arena[wall_indices] = wall_value

    #print(target)
    for coin in coins:
        arena[coin] = coin_value
        xc = coin[0]
        yc = coin[1]

        arena[(xc-1,yc)] = arena[(xc-1,yc)] + (arena[(xc-1,yc)]+10) #Gives +1 to the potential of coins
        arena[(xc+1,yc)] = arena[(xc+1,yc)] + (arena[(xc+1,yc)]+10)
        arena[(xc,yc-1)] = arena[(xc,yc-1)] + (arena[(xc,yc-1)]+10)
        arena[(xc,yc+1)] = arena[(xc,yc+1)] + (arena[(xc,yc+1)]+10)

    #New features: boxes, bombs and explosions

    crates = []

    # arena[box_indices] = crate_value

    # for i in range(box_indices.shape[0]):
    #     for j in range(box_indices.shape[1]):
    #         if box_indices[i,j]:
    #             crates.append([i,j])
    #             arena[(i-1,j)] = arena[(i-1,j)] + (arena[(i-1,j)]+1) #Gives +1 to the potential of crates
    #             arena[(i+1,j)] = arena[(i+1,j)] + (arena[(i+1,j)]+1)
    #             arena[(i,j-1)] = arena[(i,j-1)] + (arena[(i,j-1)]+1)
    #             arena[(i,j+1)] = arena[(i,j+1)] + (arena[(i,j+1)]+1)


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

        #Potential for bombs in the tiles of the explosions
        # for i in range(1, 4):
        #     if arena[(xc-i,yc)] == wall_value: break
        #     else:
        #         arena[(xc-i,yc)] = bomb_value + 10*i
        # for i in range(1, 4):
        #     if arena[(xc+i,yc)] == wall_value: break
        #     else:
        #         arena[(xc+i,yc)] = bomb_value  + 10*i
        #
        # for i in range(1, 4):
        #     if arena[(xc,yc-i)] == wall_value: break
        #     else:
        #         arena[(xc,yc-i)] = bomb_value + 10*i
        # for i in range(1, 4):
        #     if arena[(xc,yc+i)] == wall_value: break
        #     else:
        #         arena[(xc,yc+i)] = bomb_value  + 10*i

    #New way of tile values for bombs
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


    #Relative arena
    s = ((3*17),(3*17))
    arena_ext = -10*np.ones(s) #create a matrix with "mantinels"
    arena_ext[16:33,16:33] = arena #insert the real arena into the middle
    neigh = 5
    agent_neigh = np.array(arena_ext[(16+x-neigh):(17+x+neigh),(16+y-neigh):(17+y+neigh)])
    # print(agent_neigh)
    # print('arena', arena)

    ##Bomb potential features
    #+50 for lateral crates
    #-50 if not bombs left, no scape or nothing to win

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

    # print('arena ', agent_neigh)
    # print('bomb pot ', bomb_potential)
    # For example, you could construct several channels of equal shape, ...
    channels = []
    #channels.append(x,y,arena.reshape(17*17,1), )
    neigharea = (2*neigh+1)**2
    # print(agent_neigh)
    #channels = [agent_neigh.reshape(neigharea,1),coins.reshape(2*numcoins,1),np.zeros((2*(9-numcoins),1))]
    channels = [bomb_potential, agent_neigh.reshape(neigharea,1)]
    #channels = [x,y,arena.reshape(17*17,1),coins.reshape(2*numcoins,1),np.zeros((2*(9-numcoins),1))]
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.vstack(channels)
    #print(np.shape(stacked_channels))
    # and return them as a vector
    #print(np.size(stacked_channels.T))
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

import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


from numpy.linalg import lstsq

def omp_regression(X, y, T):
    """
    Orthogonal Matching Pursuit iterates T times to automatically find the weights for the
    1...T most relevant features in the training set X (standardized!) with responses y
    It returns the optimal weight vector for every iteration with dimensions DxT
    """
    # initialization
    dim = X.shape[1]
    beta_hat = np.zeros((dim,T))
    A = []
    B = list(range(X.shape[1]))
    res = y
    # iteration
    for t in range(T): # do iteration
        cor = [np.abs(np.dot(X[:,j].T,res)) for j in B] # 1a.) correlations with the current residual
        j_max = np.argmax(np.array(cor)) # 1b.) find maximum
        A.append(B.pop(j_max)) # 2.) move most important dim/feature to active set
        X_active = X[:,A] # 3.) form the active matrix
        #print(A,len(B))
        beta = lstsq(X_active,y, rcond=None) # 4.) solve least squares problem n
        res = y - np.dot(X_active,beta[0]) # 5.) update the residual
        beta_hat[A,t] = beta[0]
        #print(y)

    return beta_hat[:,-1]

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    #self.alpha_0 = 0.3
    self.alpha_weight = 0.02
    self.tau = 0.1
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    actions_sec_task = ['UP', 'RIGHT', 'DOWN','LEFT','WAIT','BOMB']
    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    # if (e.INVALID_ACTION in events) and self.mod and not self_action == None:
    #     act_index = actions_sec_task.index(self_action)
    #     self.model[act_index,:] = self.model[act_index,:]- self.model[act_index,:]*np.hstack([state_to_features(old_game_state),0])*0.1
    self.exploded_boxes = self.exploded_boxes + events.count('CRATE_DESTROYED')
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model

    ###Paula

    #Find beta for each action
    import numpy as np
    from sklearn import linear_model

    betas_ls = []
    gamma = 0.7#0.7
    invalid = 0
    actions_sec_task = ['UP', 'RIGHT', 'DOWN','LEFT','WAIT','BOMB']
    for a in actions_sec_task:
        instances = [t for t in self.transitions if t.action == a]
        if a == 'WAIT':
            waitnum = len(instances)
        if not instances:
            invalid = 1
            break
        rewards = np.array([t.reward for t in instances])
        gamma_vect = np.array([gamma**i for i in range(len(rewards))])
        Y = [np.sum(gamma_vect[:(len(rewards)-i)]*rewards[i:]) for i in range(0,len(rewards))]

        X = [t.state for t in instances]

        X = np.vstack(X)
        #E = np.eye(np.shape(X)[1])*self.tau
        #O = np.zeros(np.shape(X)[1])
        #Y= np.hstack([Y,O])

        #X = np.vstack([X, E])
        #print(np.shape(X))
        #linear  = linear_model.LinearRegression()
        #beta = linear.fit(X,Y).coef_
        if a =='BOMB': Tomp = 1
        else: Tomp = np.min([self.step-2, 5])
        beta = omp_regression(X,Y,Tomp).T*np.sqrt(self.step/400)
#        print(self.step)
        if a=='BOMB' and not self.mod:
            beta[-1] = +3
        if a=='RIGHT' and not self.mod:
            beta[0] = +3
            beta[2] = +3
        if a=='LEFT' and not self.mod:
            beta[0] = -3
            beta[2] = -3
        if a=='UP' and not self.mod:
            beta[1] = -3
            beta[3] = -3
        if a=='DOWN' and not self.mod:
            beta[1] = +3
            beta[3] = +3
        #print(np.shape(beta))
        betas_ls.append(beta)
    if not invalid:
        betas = np.array(betas_ls)
        #print(betas)
        if self.mod:
            it = np.ones([betas.shape[0],1])*(self.model[0,-1]+1)
            #print('it',it)
            # print('betas shape = ', betas.shape)
            # print('model shape = ', self.model.shape)
            # print('model[:,-1] shape = ', self.model[:,-1].shape)
            self.alpha = self.alpha_0/(1+self.alpha_weight*it[0])
            update_betas = (1-self.alpha)*self.model[:,:-1]+self.alpha*betas
            model = np.hstack([update_betas, it])
            self.model = model
            #print(model.shape)
#            print(self.model)
        else:
            #print('betas shape = ', betas.shape)
            it = np.ones([betas.shape[0],1])
            #print('it shape = ', it.shape)

            model = np.hstack([betas, it])
            print(model.shape)
            self.model = model
            self.mod = 1
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    if self.alpha_0 == 0:
        f = open("results_learn.txt", "a")
        #f.write(str(score) + "_" + str(game_state['step']))
        score = last_game_state['self'][1]
        #if last_game_state['step'] < 400 : score += 1
        f.write(str(score) + "_" + str(self.step)+ "_" + str(self.exploded_boxes) +  "\n")
        f.close()

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        e.MOVED_LEFT: -.2,
        e.MOVED_RIGHT: -.2,
        e.MOVED_UP:-.2,
        e.MOVED_DOWN:-.2,
        e.WAITED:-.7,
        e.INVALID_ACTION:-10,
        e.KILLED_SELF:-30,
        e.COIN_FOUND:5,
        e.CRATE_DESTROYED:10,
        #e.BOMB_EXPLODED: 5,
        e.BOMB_DROPPED:0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

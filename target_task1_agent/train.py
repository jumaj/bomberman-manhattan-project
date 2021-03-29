import pickle
import random
from collections import namedtuple, deque
from typing import List


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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
     self.alpha_0 = 0.7
    self.alpha_weight = 0.02
    # self.tau = 0.2

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

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    #with open("my-saved-model.pt", "wb") as file:
    #    pickle.dump(self.model, file)

    ###Our code

    #Find beta for each action
    import numpy as np
    from sklearn import linear_model

    # print(self.tau, self.alpha_0, self.gamma) #It reads this variables :D

    betas_ls = []
    gamma = 0.7

    num_features = state_to_features(last_game_state).shape[0]

    actions_firts_task = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    for a in actions_firts_task:
        instances = [t for t in self.transitions if t.action == a]

        rewards = np.array([t.reward for t in instances])
        gamma_vect = np.array([gamma**i for i in range(len(rewards))])
        Y = [np.sum(gamma_vect[:(len(rewards)-i)]*rewards[i:]) for i in range(0,len(rewards))]

        X = [t.state for t in instances]

        # print(X[0].shape[1])
        if not X:
            beta = np.zeros(num_features)
        else:
            X = np.vstack(X)
            E = np.eye(np.shape(X)[1])*self.tau
            O = np.zeros(np.shape(X)[1])
            Y= np.hstack([Y,O])

            X = np.vstack([X, E])
            #print(np.shape(X))
            linear  = linear_model.LinearRegression()
            beta = linear.fit(X,Y).coef_
            #print(np.shape(beta))
        betas_ls.append(beta)
    betas = np.array(betas_ls)
    #print(betas)
    if self.mod:
        it = np.ones([betas.shape[0],1])*(self.model[0,-1]+1)

        self.alpha = self.alpha_0/(1+self.alpha_weight*it[0])
        update_betas = (1-self.alpha)*self.model[:,:-1]+self.alpha*betas
        model = np.hstack([update_betas, it])
        self.model = model


    else:
        #print('betas shape = ', betas.shape)
        it = np.ones([betas.shape[0],1])
        #print('it shape = ', it.shape)

        model = np.hstack([betas, it])
        # print(model.shape)
        self.model = model
        self.mod = 1

    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)



    with open("tau_" + str(self.tau) + "_gamma_" + str(gamma) + "_temp_" + str(self.T) + "_epsilon_" + str(self.epsilon) + ".pt", "wb") as file:
        pickle.dump(self.model, file)

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
        e.INVALID_ACTION:-10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

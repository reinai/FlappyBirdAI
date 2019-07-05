"""
    File name: basicQLearningAgent.py
    Author: Nikola Zubic
"""
import random
import math
from FlappyBird.gameInfo import *

LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.99


def flip_coin(p):
    r = random.random()
    return r > p


class IllegalActionException(BaseException):
    def __init__(self, message):
        print(message)


class BasicQLearningAgent(object):
    """
      Basic Q-Learning Agent

      Functions:
        - __init__ constructor
        - is_state_action_explored
        - max_q
        - compute_action_from_q_values
        - update
        - get_state (static)
    """

    def __init__(self):
        self.epsilon = 0.1
        self.Q_matrix = {}

    def is_state_action_explored(self, state):
        """
        For pair (state, action) we check if it is available in Q matrix.
        In other words, have we been in state "state" and decided to perform action A = {0, 1}
        """
        state_action_up = (state, 0)  # for "UP" action
        state_action_steady = (state, 1)  # no key pressed

        if state_action_up in self.Q_matrix and state_action_steady in self.Q_matrix:
            return True
        else:
            if state_action_up not in self.Q_matrix:
                return False, 0
            elif state_action_steady not in self.Q_matrix:
                return False, 1

    def max_q(self, state):
        """
        Idea is to pick the action with biggest (best) Q-Value of next state in order
        to stay in optimal policy Qpi* .
        So, if policy of SARSA is greedy, it becomes Q-learning.
        """

        if (state, 0) not in self.Q_matrix:
            self.Q_matrix[(state, 0)] = 0
        if (state, 1) not in self.Q_matrix:
            self.Q_matrix[(state, 1)] = 0

        max_q = 1

        if self.Q_matrix[(state, 0)] > self.Q_matrix[(state, 1)]:
            max_q = 0

        return max_q

    def compute_action_from_q_values(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, we
          will return None.
        """
        try:

            if (state, 0) not in self.Q_matrix:
                self.Q_matrix[(state, 0)] = 0
            if (state, 1) not in self.Q_matrix:
                self.Q_matrix[(state, 1)] = 0

            if flip_coin(self.epsilon):
                # We will select action with highest Q-Value
                if self.Q_matrix[(state, 0)] > self.Q_matrix[(state, 1)]:
                    return 0
                else:
                    return 1
            else:
                # We will do exploration, by selecting the random action
                return random.randint(0, 1)

        except IllegalActionException:
            return None

    def update(self, state, action, next_state, reward):
        """
          We call this to observe a
          state = action => nextState and reward transition.
          Here is done the Q-Value update.

          For update, we use formula given here within the algorithm:
            https://nthu-datalab.github.io/ml/labs/16-1_Q-Learning/16-1_Q_Learning.html

        """
        sample = (reward + DISCOUNT_FACTOR * self.Q_matrix[(next_state, self.max_q(next_state))] -
                  self.Q_matrix[(state, action)])

        self.Q_matrix[(state, action)] = self.Q_matrix[(state, action)] + LEARNING_RATE * sample

    @staticmethod
    def get_state(state):
        """
        Representation of state.
        state = (agent y position, agent velocity, next pipe top y position, next pipe distance to player)
        """
        next_pipe_distance_to_player = math.floor(state['next_pipe_dist_to_player']/(SCREEN_WIDTH / PIXEL_DISCOUNT))
        next_pipe_top_y = math.floor(state['next_pipe_top_y'] / (SCREEN_HEIGHT / PIXEL_DISCOUNT))
        player_vel = math.floor(state['player_vel'] / (MAX_DROP_SPEED / PIXEL_DISCOUNT))
        player_y = math.floor(state['player_y'] / (SCREEN_HEIGHT / PIXEL_DISCOUNT))

        state = (player_y, player_vel, next_pipe_top_y, next_pipe_distance_to_player)

        return state

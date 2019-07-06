"""
    File name: monteCarloSearch.py
    Author: Nikola Zubic
"""

import math
import random
from FlappyBird.gameInfo import *

"""
Monte Carlo tree search begins with a tree consisting of only one node.
It is always good to represent node as a class in a tree, because it is an element that builds the tree structure.
"""

"""
Idea from Monte Carlo search library.
"""


def get_state(state):
    next_pipe_distance_to_player = math.floor(state['next_pipe_dist_to_player'] * bucket_range_per_feature /
                                              SCREEN_WIDTH)
    player_vel = math.floor((state['player_vel'] - MIN_VELOCITY) * FIRST_NON_ZERO_VELOCITY / MAX_VELOCITY)

    player_y = state['player_y']
    pipe_gap_average = (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) / 2.0
    next_pipe_gap_y_distance_to_player = math.floor((player_y - pipe_gap_average) * bucket_range_per_feature /
                                                    SCREEN_HEIGHT)

    state = (player_vel, next_pipe_distance_to_player, next_pipe_gap_y_distance_to_player)

    return state


class Node(object):
    # One node represents current Flappy Bird state
    def __init__(self, state):
        self.state = state
        self.N = 0  # Visit count for node
        self.parent = None
        self.children = []
        self.is_expanded = False
        self.W = 0  # accumulated value
        self.p = 0  # Probability of choosing action in state represented with Node
        self.exploring_rate = 0.25

    def get_state(self):
        state = get_state(self.state)
        return state

    def add_child(self, child):
        # Add child to current node (state) and assign its parent to Node
        if child is None:
            return
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        # In order to generate all successor states
        for child in children:
            self.add_child(child)

    def update_node_value(self, value):
        """
        Value of node W is propagated back up the tree by increasing parent's value and visit count.
        Accumulated value of parent node is set to be a total of accumulated values of its children.
        """
        self.W += value
        self.N += 1

        if self.parent is not None:
            self.parent.update_node_value(value)

    def refresh_probability(self, new_value):
        self.p = new_value

    def upper_confidence_tree_score(self):
        """
        Main problem in selecting child nodes is maintaining the balance between exploitation and exploration moves
        with few simulations, but with an idea to get high average win rate.
        Formula for balancing exploit and explore phases is called:
            Upper confidence tree (UCT) score
        The formula can be seen here:
                                    https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
        """

        """
        (wi / ni) + c * sqrt(ln(Ni) / ni) for i-th node (state), where:
            c = self.exploring_rate
            Ni = self.parent.N (parent visits)
            ni = self.N (current node visits)
            wi = self.W (accumulated value)
        """

        if self.N < 1:
            self.N = 1

        utc_score = self.W / self.N + self.exploring_rate * math.sqrt((math.log(self.parent.N)) / self.N)

        return utc_score

    def get_best_successors(self):
        best_successors = []
        best_score = -6.0

        for child in self.children:
            score = child.upper_confidence_tree_score()

            if score > best_score:
                best_score = score
                best_successors = [child]
            elif score == best_score:
                best_successors.append(child)

        if len(best_successors) == 0:
            return

        return random.choice(best_successors)


class MonteCarloSearch(object):
    def __init__(self, root_node):
        self.root_node = root_node
        self.children_search = None
        self.compute_node_values = lambda child, monte_carlo: None

    def selection(self):
        """
        Take the current state of the tree and generate next states to a certain depth.
        """
        best_successors = []
        max_n = -1  # max visits count

        for child in self.root_node.children:
            if child.N > max_n:
                max_n = child.N
                best_successors = [child]
            elif child.N == max_n:
                best_successors.append(child)

        if len(best_successors) == 0:
            return None

        return random.choice(best_successors)

    def expansion(self, node):
        """
        Move one step down after generating tree of certain depth with selection,
        expose a new node (state) in the tree
        """
        self.children_search(node, self)

        for child in node.children:
            W = self.compute_node_values(child, self)

            if W is not None:
                child.update_node_value(W)

            if not (child.N or (child.p is not None)):
                self.roll_out(child)
                child.children = []

        if len(node.children):
            node.is_expanded = True

    def simulation(self, expansion_count=1):
        """
        Simulate the value of expanded state.
        In simulation phase, we do completely random decisions from this point until we reach terminal state
        by loss.
        """
        for i in range(expansion_count):
            current_node = self.root_node

            while current_node.is_expanded:
                current_node = current_node.get_best_successors()

            self.expansion(current_node)

    def roll_out(self, node):
        """
        Going from the child node we roll out game by randomly taking moves from the child state until we reach
        terminal state.
        Information is then propagated back from child nodes up to the parent by incrementing W and visit count.
        """
        self.children_search(node, self)
        child = random.choice(node.children)
        node.children = []
        node.add_child(child)
        child_W_value = self.compute_node_values(child, self)

        if child_W_value is not None:
            node.update_node_value(child_W_value)
        else:
            self.roll_out(child)

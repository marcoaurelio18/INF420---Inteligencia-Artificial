'''
Resposta passo 3: 
python3 gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 0 -l 0.1
Epsilon 0 e taxa de aprendizagem 0.1
'''
# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from more_itertools import difference
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # set all (state, action) pair to 0
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # returns a list with the values of q
        q_values = self.QValues[(state, action)]
        return q_values

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # takes the possible actions, if there are none possible, the value of q is zero
        allowed_actions = self.getLegalActions(state)
        if len(allowed_actions) == 0:
            return 0.0
        # based on the selected policy, defines the best action
        max_action = self.getPolicy(state)
        return self.getQValue(state, max_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # select the possible actions in the current state
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        # set to save the values of the actions
        action_vals = {}
        # variable to save the best value of q
        best_q_value = float('-inf')
        # goes through the possible actions
        for action in legal_actions:
            # save the value of q in a variable
            target_q_value = self.getQValue(state, action)
            # puts that value into an array of values
            action_vals[action] = target_q_value
            # defines the best value of q
            if target_q_value > best_q_value:
                best_q_value = target_q_value
        # vector for best actions
        best_actions = []
        for k,v in action_vals.items():
          if v == best_q_value:
               best_actions.append(k)
        # if any action tie, randomly set
        best_actions = random.choice(best_actions)
        # random tie-breaking
        return best_actions

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # takes the possible actions, if there are none possible, the value of q is zero
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        action = None
        if not util.flipCoin(self.epsilon):
            # exploit
            action = self.getPolicy(state)
        else:
            # explore
            action = random.choice(legal_actions)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        old_value = self.getQValue(state,action)
        future_q_value = self.getValue(nextState)
        difference = (reward + (self.discount * future_q_value) - old_value)
        new_q_value = old_value + (self.alpha * difference)
        self.QValues[(state, action)] = new_q_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # features vector
        features = self.featExtractor.getFeatures(state, action)
        return features * self.weights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        oldValue = self.getQValue(state, action)
        futureQValue = self.getValue(nextState)
        difference = (reward + self.discount * futureQValue) - oldValue
        # for each feature i
        for feature in features:
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # print(self.weights)
            pass
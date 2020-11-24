
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend base class by implementing
        Q-Learning : """
class QLearningAgent(My10x10GridWorld):

    """New methods used for QLearning"""
    # TODO: Implement Q-Learning

    def runQLearning(self):
        return 0

"""Let the agent reinforce"""
qlearning_agent = QLearningAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                         rewards, neg_reward_states, pos_reward_states, Walls,
                         Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon)

qlearning_agent.runQLearning()

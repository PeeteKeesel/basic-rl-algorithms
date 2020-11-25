
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend base class by implementing
        Q-Learning : """
class QLearningAgent(My10x10GridWorld):

    """New methods used for QLearning"""

    # Todo: runs forever like this. how to show value function? now I update the value function by taking the max
    #  value of Q(s, all a)

    """New methods used for Q-Learning"""
    def takeMaxAction(self, state):

        # Q-values for all directions of the state
        all_q = self.Q[self.decodeState(state), :]

        # what are the actions with the maximal Q-values
        max_action_indices = np.where(all_q == max(all_q))[0]
        max_actions = self.A[max_action_indices]
        
        # take random action if there are multiple maximal values
        rdm_max_action = np.random.choice(max_actions)

        return rdm_max_action

    def qLearning(self, episodes):

        # initialize Q-table
        self.Q = np.zeros((len(self.states_encoded.keys()), len(self.A)))

        for episode in range(episodes):

            print(f"episode={episode}")

            state = self.starting_state
            a = self.takeMaxAction(state)

            iters = 0

            while True:

                reward = self.getRewardForAction(state)
                if self.isTerminalState(state):
                    state_nxt = self.starting_state.copy()
                else:
                    state_nxt = np.array(self.getIndiceAfterAction(state, a))

                if self.isOutOfGridOrAtWall(state_nxt):
                    state_nxt = state.copy()

                # next action a` = eps-greedy at s`
                a_nxt = self.takeMaxAction(state_nxt)

                # get Q(s', a') from the table
                Q_nxt = self.Q[self.decodeState(state_nxt), self.getIndexInActionList(a_nxt)]
                Q_now = self.Q[self.decodeState(state), self.getIndexInActionList(a)]

                # Update state-action value: Q(s, a) <- Q(s, a) + alpha*[ R + gamma*Q(s', a') - Q(s, a) ]
                self.Q[self.decodeState(state),
                       self.getIndexInActionList(a)] = Q_now + self.Alpha * ( reward + self.Gamma*Q_nxt - Q_now )

                if self.isTerminalState(state):
                    break

                a, state = a_nxt, state_nxt

                iters += 1

            print(f"iters={iters}")

            # improve policy toward greediness wrt q_pi
            self.policyImprovementByQ()
            print(self.pi)

            # (optional): update v_pi according to Q - just to compare with the REINFORCEjs results
            for r in range(self.Q.shape[0]):
                self.v[self.states_encoded[r][0], self.states_encoded[r][1]] = np.max(self.Q[r])
            print(np.round(self.v, 2))

    def runQLearning(self):

        self.qLearning(200)
        self.policyImprovementByQ()
        print(self.pi)

"""Let the agent reinforce"""
qlearning_agent = QLearningAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                         rewards, neg_reward_states, pos_reward_states, Walls,
                         Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon)

qlearning_agent.runQLearning()

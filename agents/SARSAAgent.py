
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend base class by implementing
        SARSA : """
class SARSAAgent(My10x10GridWorld):

    # Todo: runs forever like this. how to show value function? now I update the value function by taking the max
    #  value of Q(s, all a)

    """New methods used for SARSA"""
    def takeEpsGreedyAction(self, state):

        eps_greedy = np.random.randn(1)

        # follow policy (exploit)
        if eps_greedy > self.epsilon:
            policy_actions = (self.pi[state[0], state[1]]).replace(" ", "")
            policy_actions = list(policy_actions)
            policy_action = np.random.choice(policy_actions, size=1)[0]

            return policy_action

        # take random action (explore)
        else:
            random_action = self.takeRandomAction()

            return random_action

    def sarsa(self, episodes):

        # initialize Q-table
        self.Q = np.zeros((len(self.states_encoded.keys()), len(self.A)))

        for episode in range(episodes):

            print(f"episode={episode}")

            state = self.starting_state
            a = self.takeEpsGreedyAction(state)

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
                a_nxt = self.takeEpsGreedyAction(state_nxt)

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

    def runSarsa(self):

        self.sarsa(2)
        self.policyImprovementByQ()
        print(self.pi)


"""Let the agent reinforce"""
sarsa_agent = SARSAAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                         rewards, neg_reward_states, pos_reward_states, Walls,
                         Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon)

sarsa_agent.runSarsa()

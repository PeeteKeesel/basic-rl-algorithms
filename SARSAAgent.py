
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
        if eps_greedy > eps:
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

            print(f"episode={episode}\n{self.pi}")
            #\n{np.round(self.Q, 2)}
            #print(f"-- episode={episode}\n{np.round(self.Q, 2)}")

            state = self.starting_state
            a = self.takeEpsGreedyAction(state)

            iters = 0

            while not self.isTerminalState(state):

                if iters > 10000:
                    break

                reward = self.getRewardForAction(state)
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

                a, state = a_nxt, state_nxt

                iters += 1

            # Todo: how to update the terminal state? since the agent doesnt take an action there.
            state_nxt = self.starting_state
            a_nxt = self.takeEpsGreedyAction(state_nxt)
            Q_start = self.Q[self.decodeState(state_nxt), self.getIndexInActionList(a_nxt)]
            Q_now = self.Q[self.decodeState(state), self.getIndexInActionList(a)]
            self.Q[self.decodeState(state),
                   self.getIndexInActionList(a)] = Q_now + self.Alpha * ( self.getRewardForAction(state) +
                                                                           self.Gamma*Q_start - Q_now )

            # improve policy toward greediness wrt q_pi
            # Todo: is the policy updated in every step of an episode or only at the end of an episode?
            self.policyImprovementByQ()

            for r in range(self.Q.shape[0]):
                self.v[self.states_encoded[r][0], self.states_encoded[r][1]] = np.max(self.Q[r])
            print(np.round(self.v, 2))

    def runSarsa(self):

        self.sarsa(100)


"""Let the agent reinforce"""
sarsa_agent = SARSAAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                         rewards, neg_reward_states, pos_reward_states, Walls,
                         Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon)

sarsa_agent.runSarsa()

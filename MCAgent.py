
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend the base class by implementing
        Monte-Carlo"""
class MCAgent(My10x10GridWorld):

    """New methods used for MC"""
    def firstVisitMCPrediction(self, episodes):
        """
        Estimates the value function by by using first-visit monte-carlo method. This
        averages all first visit returns for each state in the trajectory of an episode.

        :param
            epsiodes: int - number of episodes to run
        """

        # counter for each state - to calculate the average return in the end
        N_s = np.zeros((self.NROWS, self.NCOLS))
        Returns_s = np.zeros((self.NROWS, self.NCOLS))

        # number of episodes = times agent goes back to start state
        for episode in range(episodes):

            print(f"-- episode={episode}\n{np.round(self.v, 2)}")

            # counter for each state per episode - to check first-visits
            N_s_per_episode = np.zeros((self.NROWS, self.NCOLS))
            state = self.starting_state
            state_traj, action_traj, reward_traj = np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)

            # generate state, action, reward trajectory for the episode
            while not self.isTerminalState(state): # until terminal state is reached

                state_traj = np.vstack((state_traj, state))
                rdm_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
                action_traj = np.append(action_traj, rdm_action)
                state_after_action = self.getIndiceAfterAction(state, rdm_action)
                reward_traj = np.append(reward_traj, self.getRewardForAction(state))

                if not self.isOutOfGridOrAtWall(state_after_action):
                    state = state_after_action

            # for the terminal state
            state_traj = np.vstack((state_traj, state))
            action_traj = np.append(action_traj, self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]])
            reward_traj = np.append(reward_traj, self.getRewardForAction(state))

            G_t = 0
            for t, state in enumerate(reversed(state_traj)):
                G_t += reward_traj[t]

                if N_s_per_episode[state[0], state[1]] == 0:
                    N_s_per_episode[state[0], state[1]] += 1
                    N_s[state[0], state[1]] += 1
                    Returns_s[state[0], state[1]] += G_t

            self.v = np.divide(Returns_s, N_s, out=np.zeros_like(Returns_s), where=N_s != 0)

        self.v = np.divide(Returns_s, N_s, out=np.zeros_like(Returns_s), where=N_s!=0)


    def monteCarloFirstVisitPolicyEvaluation(self, vOld):
        """ MC Backup:
            V(S_t) = V(S_t) + alpha * (G_t - V(S_t))
        """
        vNew = np.zeros((self.NROWS, self.NCOLS))

        # following a random policy we update the value function
        for row in range(self.NROWS):
            for col in range(self.NCOLS):

                # TODO: What to do when in terminal state?
                if self.isTerminalState([row, col]):
                    vNew[row, col] = self.getRewardForAction([row, col])
                    continue

                if self.isOutOfGridOrAtWall([row, col]):
                    vNew[row, col] = np.inf
                    continue

                #current_v = self.v[row, col]
                current_v = vOld[row, col]

                # V(S_t) = V(S_t) + lambda * (G_t - V(S_t))
                # with G_t = R_{t+1} + gamma*R_{t+2} + ... + gammaˆ{T-1}*R_{T}
                current_v = current_v + self.Alpha * ( self.getReturnForState([row, col],
                                                                              0,
                                                                              0) - current_v )

                vNew[row, col] = current_v

        return np.round(vNew, 2)

    def runFirstVisitMonteCarlo(self, episodes = 10):
        """Runs first-visit Monte-Carlo Method"""

        self.firstVisitMCPrediction(episodes)
        self.policyImprovement()
        print(self.pi)

    def monteCarloEveryVisitPolicyEvaluation(self):
        """ MC Backup:
            V(S_t) = V(S_t) + lambda * (G_t - V(S_t))
        """
        # TODO: Implement Every-Visit MC
        return 0

"""Let the agent reinforce"""
mc_agent = MCAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                   rewards, neg_reward_states, pos_reward_states, Walls,
                   Gamma, v, pi, piProbs, eps, Alpha, epsilon)

mc_agent.runFirstVisitMonteCarlo(100)

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
                rdm_action = self.takeRandomAction()
                action_traj = np.append(action_traj, rdm_action)
                state_after_action = self.getIndiceAfterAction(state, rdm_action)
                reward_traj = np.append(reward_traj, self.getRewardForAction(state))

                if not self.isOutOfGridOrAtWall(state_after_action):
                    state = state_after_action

            # for the terminal state
            state_traj = np.vstack((state_traj, state))
            action_traj = np.append(action_traj, self.takeRandomAction())
            reward_traj = np.append(reward_traj, self.getRewardForAction(state))

            # to get the correct indice in reverse order
            length = len(reward_traj)

            G_t = 0
            for t, state in enumerate(reversed(state_traj)):
                G_t += reward_traj[length - 1 - t]

                if N_s_per_episode[state[0], state[1]] == 0:
                    Returns_s[state[0], state[1]] += G_t
                    N_s_per_episode[state[0], state[1]] += 1
                    N_s[state[0], state[1]] += 1

            self.v = np.divide(Returns_s, N_s, out=np.zeros_like(Returns_s), where=N_s != 0)

        self.v = np.divide(Returns_s, N_s, out=np.zeros_like(Returns_s), where=N_s!=0)


    def everyVisitMCPrediction(self, episodes):
        """
        Estimates the value function by by using first-visit monte-carlo method. This
        averages returns for all visits for each state in the trajectory of an episode.

        :param
            epsiodes: int - number of episodes to run
        """

        # counter for each state - to calculate the average return in the end
        N_s = np.zeros((self.NROWS, self.NCOLS))
        Returns_s = np.zeros((self.NROWS, self.NCOLS))

        # number of episodes = times agent goes back to start state
        for episode in range(episodes):

            print(f"-- episode={episode}\n{np.round(self.v, 2)}")

            state = self.starting_state
            state_traj, action_traj, reward_traj = np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)

            # generate state, action, reward trajectory for the episode
            while not self.isTerminalState(state): # until terminal state is reached

                state_traj = np.vstack((state_traj, state))
                rdm_action = self.takeRandomAction()
                action_traj = np.append(action_traj, rdm_action)
                state_after_action = self.getIndiceAfterAction(state, rdm_action)
                reward_traj = np.append(reward_traj, self.getRewardForAction(state))

                if not self.isOutOfGridOrAtWall(state_after_action):
                    state = state_after_action

            # for the terminal state
            state_traj = np.vstack((state_traj, state))
            action_traj = np.append(action_traj, self.takeRandomAction())
            reward_traj = np.append(reward_traj, self.getRewardForAction(state))

            # to get the correct indice in reverse order
            length = len(reward_traj)

            G_t = 0
            for t, state in enumerate(reversed(state_traj)):
                G_t += reward_traj[length - 1 - t]
                Returns_s[state[0], state[1]] += G_t
                N_s[state[0], state[1]] += 1

            self.v = np.divide(Returns_s, N_s, out=np.zeros_like(Returns_s), where=N_s!=0)

        self.v = np.divide(Returns_s, N_s, out=np.zeros_like(Returns_s), where=N_s!=0)

    def runFirstVisitMonteCarlo(self, episodes = 10):
        """Runs first-visit Monte-Carlo Method"""

        self.firstVisitMCPrediction(episodes)
        self.policyImprovementByV()
        print(self.pi)

    def runEveryVisitMonteCarlo(self, episodes = 10):
        """Runs every-visit Monte-Carlo Method"""

        self.everyVisitMCPrediction(episodes)
        self.policyImprovementByV()
        print(self.pi)


"""Let the agent reinforce"""
mc_agent = MCAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                   rewards, neg_reward_states, pos_reward_states, Walls,
                   Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon)

whatToDo = input("Press 1 for First-Visit Monte-Carlo:\n"
                 "      2 for Every-Visit Monte-Carlo: ")

if whatToDo == "1":
    mc_agent.runFirstVisitMonteCarlo(20)
elif whatToDo == "2":
    mc_agent.runEveryVisitMonteCarlo(20)

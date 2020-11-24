
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend base class by implementing
        TD(0) : 1-step-lookahead"""
class TD0Agent(My10x10GridWorld):

    """New methods used for TD(0)"""
    def getTdTargetFor1StepLookahead(self, state):
        """
        Returns the state in which the agent ends up after taking a random action as well as the
        TD-target (=the actual reward the agent is getting for that move).

        :param
            state: 2d-list - current state to evaluate
        :return
            2d-list, - the state in which the agent ends up after taking the action
            float    - the TD-target = R_{t+1} + gamma * V(S_{t+1})
        """

        if self.isTerminalState(state):
            return self.starting_state, self.getRewardForAction(state) + self.Gamma * self.v[self.starting_state[0],
                                                                                             self.starting_state[1]]

        random_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
        state_after_action = self.getIndiceAfterAction(state, random_action)
        reward = self.getRewardForAction(state)

        if self.isOutOfGridOrAtWall(state_after_action):
            return state_after_action, reward + self.Gamma * self.v[state[0], state[1]]
        else:
            return state_after_action, reward + self.Gamma * self.v[state_after_action[0], state_after_action[1]]

    def TD0(self, episodes):
        """
        Evaluate a given policy by running TD(0) Backup = doing 1 step look-aheads:
            V(S_t) = V(S_t) + alpha * (R_{t+1} + gamma*V(S_{t+1}) - V(S_t))

        :param
            episodes: int - number of episodes to run
        """
        # totol number of iterations per episode
        totIters = []

        # number of episodes = times agent goes back to start state
        for episode in range(episodes):

            print(f"-- episode={episode}\n{np.round(self.v, 2)}")

            # iterate until maximal number of iterations is reached
            no_of_iter, state = 0, self.starting_state
            for _ in range( 10000 ):

                current_value = self.v[state[0], state[1]]
                next_state, tdTarget = self.getTdTargetFor1StepLookahead([state[0], state[1]])

                # TD(0) backup
                self.v[state[0], state[1]] = current_value + self.Alpha * (tdTarget - current_value)

                if self.isTerminalState(state):
                    break

                # stay at current state if next state is out of grid or at a wall
                if not self.isOutOfGridOrAtWall(next_state):
                    state = next_state

                no_of_iter += 1

            totIters.append(no_of_iter)

        # improve policy online
        self.policyImprovementByV()
        print(f"final policy:\n{self.pi}")

    def runTD0(self):
        """Runs TD(0)"""
        print(f"-- k=0\n{self.v}")

        self.TD0(100)


    def get1StepTdTargetForStateEpsGreedy(self, state):
        """Do 1-step lookahead by taking a random action"""

        # sample from std normal
        eps_greedy = np.random.randn(1)

        # follow policy pi with prob 1-eps
        if eps_greedy > eps:
            # get actions from policy for current state
            current_pi_actions_for_state = (self.pi[state[0], state[1]]).replace(" ", "")
            current_pi_actions_for_state = list(current_pi_actions_for_state)

            # take random action if multiple are in the current policy
            if len(current_pi_actions_for_state) > 1:
                random_action = np.random.choice(current_pi_actions_for_state, size=1)[0]
                state_after_action = self.getIndiceAfterAction(state, random_action)

            # otherwise just take the action from the policy
            elif len(current_pi_actions_for_state) == 1:
                state_after_action = self.getIndiceAfterAction(state, current_pi_actions_for_state[0])
            else:
                print(f"ERROR: the length {len(current_pi_actions_for_state)} is wrong.")
                return "Error", np.NINF

            # reward for leaving the current state
            reward = self.getRewardForAction(state)

            if self.isOutOfGridOrAtWall(state_after_action):
                return state_after_action, reward + self.Gamma * self.v[state[0], state[1]]
            else:
                return state_after_action, reward + self.Gamma * self.v[state_after_action[0], state_after_action[1]]

        # epsilon-greedy step: take random action with prob eps
        else:
            random_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
            state_after_action = self.getIndiceAfterAction(state, random_action)
            reward = self.getRewardForAction(state)

            if self.isOutOfGridOrAtWall(state_after_action):
                return state_after_action, reward + self.Gamma * self.v[state[0], state[1]]
            else:
                return state_after_action, reward + self.Gamma * self.v[state_after_action[0], state_after_action[1]]

        """OLD IDEA
        random_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
        state_after_action = self.getIndiceAfterAction(state, random_action)
        reward = self.getRewardForAction(state_after_action)

        if self.isOutOfGridOrAtWall(state_after_action):
            return reward + self.Gamma * vOld[state[0], state[1]]
        else:
            return reward + self.Gamma * vOld[state_after_action[0], state_after_action[1]]
        """

    def getNStepReturn(self, state, n):
        """ Returns the return obtained after randomly walking n-steps into the future.

        @input
            state: 2d-array - starting state
            n    : int      - steps to take into the future
        @output:
            state: 2d-array - the state the agent ended up in after taking n random actions
            G_t  : float    - the value of the return obtained after n steps
        """
        G_t = 0
        current_state = next_state = state
        for j in range(1, n+1):
            G_t += self.Gamma**(j-1) * self.getRewardForAction(current_state)

            if self.isTerminalState(current_state):
                break

            # take random action and set state to the state the agent ends up in
            random_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
            next_state = self.getIndiceAfterAction(current_state, random_action)

            if self.isOutOfGridOrAtWall(next_state):
                continue
            else:
                current_state = next_state

        G_t += self.Gamma**n * self.v[state[0], state[1]]
        return state, G_t

    def TDn(self, episodes, n):
        """ V(S_t) = V(S_t) + alpha * (R_t^m - V(S_t))
            Runs Temporal Difference learning algo where the TD target looks n steps into the future.

        @input:
            episodes: int - number of episodes the agent should run
            n       : int - number of steps TD target looks into the future
        """

        for episode in range(episodes):

            state = self.starting_state

            # TODO: Use different n's and average via G_t^lambda

            while True:

                current_value = self.v[state[0], state[1]]

                # n-step return
                state_after_n, G_t_n = self.getNStepReturn(state, n)

                # update value of current state
                self.v[state[0], state[1]] = current_value + self.Alpha * (G_t_n - current_value)

                if self.isTerminalState(state):
                    break

                state = state_after_n

    def getLambdaReturn(self, lmbda, G_t_n, t):

        G_row = G_t_n[t, :]
        G_t_lmbda = 0

        # iterate over the rows
        for n in range(1, len(G_row)+1):
            G_t_lmbda += lmbda**(n-1) * G_row[n]

        return (1 - lmbda) * G_t_lmbda

    def TDLambda(self, lmbda, G_t_n):
        """ G_t^lambda = (1 - lambda) * sum_{n=1}^inf (lambda^{n-1} * G_T^{(n)})
            Updates the lambda-return G_t^lambda by combining all n-step returns G_t^{(n)}.

        @input:
            lmbda: int                 - weight for averaging the returns over different n
            G_t_n: epsiodes x n matrix - matrix of all n-step forward-view returns
                                         column = n = how many steps to look into the future
                                         row    = t = from which timestep is the agent starting to look into the future
        """

        # iterate over episodes
            # start in starting_state
                # iterate over different n
                    # for every n remember the return you got
                # update state value using all G_t^n you obtained
                # Note: continue this for every state yopu ended up and take the appropriate n-step until in
                #       terminal state


"""Let the agent reinforce"""
td0_agent = TD0Agent([NROWS, NCOLS], starting_state, terminal_states, A,
                    rewards, neg_reward_states, pos_reward_states, Walls,
                    Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon)

td0_agent.runTD0()
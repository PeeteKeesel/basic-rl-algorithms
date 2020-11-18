"""
Rewrite the 'GridWorld: TD' example from https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html
"""

import numpy as np

"""Define the environment"""
# Actions
A = ["n", "w", "s", "e"]

# Policy = random uniform
piProbs = {A[0]: .25, A[1]: .25, A[2]: .25, A[3]: .25}

# Rewards
rewards = {"pos": 1, "neg": -1}
neg_reward_states = np.array([[3, 3], [4, 5], [4, 6], [5, 6], [5, 8], [6, 8], [7, 3], [7, 5], [7, 6]])
pos_reward_states = np.array([[5, 5]])

# Walls
Walls = np.array([[2, 1], [2, 2], [2, 3], [2, 4],
                  [3, 4], [4, 4], [5, 4], [6, 4], [7, 4],
                  [2, 6], [2, 7], [2, 8]])

# Undiscounted episodic MDP
Gamma = 0.9

# Grid dimension
NROWS, NCOLS = 10, 10

# Value Function
v = np.zeros((NROWS, NCOLS))
pi = np.full([NROWS, NCOLS], "nwse")

terminal_states = np.array([[5, 5]])
starting_state = np.array([[0, 0]])

# Stopping criterion
eps = 1e-4

# Learning rate for value function
Alpha = 0.1

# Initial epsilon for epsilon-greedy policy
epsilon = 0.2


########################################################################################################################


class My10x10GridWorld:

    def __init__(self, shape, starting_state, terminal_states, A,
                 rewards, neg_reward_states, pos_reward_states, Walls,
                 Gamma, v, pi, piProbs, eps, Alpha, epsilon):
        self.NROWS = shape[0]
        self.NCOLS = shape[1]
        self.v = np.zeros((self.NROWS, self.NCOLS))
        self.starting_state = starting_state
        self.terminal_states = terminal_states

        self.A = A
        self.rewards = rewards
        self.neg_reward_states = neg_reward_states
        self.pos_reward_states = pos_reward_states
        self.walls = Walls

        self.Gamma = Gamma

        self.v = v
        self.pi = pi
        self.piProbs = piProbs

        self.eps = eps
        self.Alpha = Alpha
        self.epsilon = epsilon

    @staticmethod
    def getIndiceAfterAction(current_state, a):
        """Get the indice for an action."""
        if a == "n":
            return [current_state[0] - 1, current_state[1]]
        if a == "w":
            return [current_state[0],     current_state[1] - 1]
        if a == "s":
            return [current_state[0] + 1, current_state[1]]
        if a == "e":
            return [current_state[0],     current_state[1] + 1]
        else:
            print(f"Action {a} is not a feasible input ('n', 'w', 's', 'e').")

    @staticmethod
    def isIn(possible_elem_of_set, set):
        return next((True for elem in set if np.array_equal(elem, possible_elem_of_set)), False)

    @staticmethod
    def countCharsInString(string):
        return len(string.replace(" ", ""))

    def isOutOfGridOrAtWall(self, current_state):
        """Check if current_state is out of the GridWorld or in a wall."""
        return (not ((0 <= current_state[0] <= self.NROWS - 1) and (0 <= current_state[1] <= self.NCOLS - 1))) or \
               self.isIn(current_state, self.walls)

    def getRewardForAction(self, next_state):
        if self.isIn(next_state, self.neg_reward_states):
            return self.rewards['neg']
        elif self.isIn(next_state, self.pos_reward_states):
            return self.rewards['pos']
        else:
            return 0

    def isTerminalState(self, s):
        return self.isIn(s, self.terminal_states)

    def policyImprovement(self, vk):
        """Greedy Policy Improvement: Update Policy greedily for each value function at time-step k"""
        piUpdate = np.empty([self.NROWS, self.NCOLS], dtype="<U10")

        # following a random policy we update the value function
        for row in range(self.NROWS):
            for col in range(self.NCOLS):

                if self.isTerminalState([row, col]):
                    piUpdate[row, col] = "OOOO"
                    continue

                if self.isOutOfGridOrAtWall([row, col]):
                    piUpdate[row, col] = "XXXX"
                    continue

                vSuccessors = np.zeros(len(self.A)) + np.NINF

                for i, a in enumerate(self.A):
                    # get indice of the state after taking the action a
                    i_after_Action = self.getIndiceAfterAction([row, col], a)

                    if self.isOutOfGridOrAtWall(i_after_Action):
                        continue

                    # Get value of successor state
                    vSuccessor = vk[row, col] if self.isOutOfGridOrAtWall([i_after_Action[0], i_after_Action[1]]) \
                        else vk[i_after_Action[0], i_after_Action[1]]

                    vSuccessors[i] = vSuccessor

                # find the indice(s) of the maximal successor values
                maxIndices = [ind for ind, vSuccessor in enumerate(vSuccessors) if vSuccessor == max(vSuccessors)]
                # get the corresponding direction
                directions = [A[ind] for ind in maxIndices]

                piUpdate[row, col] = "{:<4}".format("".join(directions))

        return piUpdate

    def get1StepTdTargetForState(self, state, vOld):
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
                random_action = np.random.choice(current_pi_actions_for_state, size=1)
                state_after_action = self.getIndiceAfterAction(state, random_action)

            # otherwise just take the action from the policy
            elif len(current_pi_actions_for_state) == 1:
                state_after_action = self.getIndiceAfterAction(state, current_pi_actions_for_state)
            else:
                print(f"ERROR: the length {len(current_pi_actions_for_state)} is wrong.")
                return np.NINF

            # reward for leaving the current state
            reward = self.getRewardForAction(state)

            if self.isOutOfGridOrAtWall(state_after_action):
                return reward + self.Gamma * vOld[state[0], state[1]]
            else:
                return reward + self.Gamma * vOld[state_after_action[0], state_after_action[1]]

        # epsilon-greedy step: take random action with prob eps
        else:
            random_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
            state_after_action = self.getIndiceAfterAction(state, random_action)
            reward = self.getRewardForAction(state)

            if self.isOutOfGridOrAtWall(state_after_action):
                return reward + self.Gamma * vOld[state[0], state[1]]
            else:
                return reward + self.Gamma * vOld[state_after_action[0], state_after_action[1]]

        """OLD IDEA
        random_action = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
        state_after_action = self.getIndiceAfterAction(state, random_action)
        reward = self.getRewardForAction(state_after_action)

        if self.isOutOfGridOrAtWall(state_after_action):
            return reward + self.Gamma * vOld[state[0], state[1]]
        else:
            return reward + self.Gamma * vOld[state_after_action[0], state_after_action[1]]
        """

    def TD0(self, vOld):
        """ TD(0) Backup:
            V(S_t) = V(S_t) + alpha * (R_{t+1} + gamma*V(S_{t+1}) - V(S_t))
            Look 1 step ahead
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
                current_v = current_v + self.Alpha * ( self.get1StepTdTargetForState([row, col],
                                                                                     vOld) - current_v )

                vNew[row, col] = current_v

        return np.round(vNew, 2)

    def runTD0(self, whenToPrint, iter):
        """Does Policy Evaluation"""
        vOld = self.v.copy()
        print(f"-- k=0\n{vOld}")

        for k in range(1, iter+1):
            vNew = self.TD0(vOld)
            if k in whenToPrint:
                print(f"-- k={k}\n{vNew}")
                self.pi = self.policyImprovement(vNew)
                print(self.pi)


            # check for convergence via stopping criterion
            #if np.abs(np.sum(vNew - vOld)) < eps:
            #    print(f"Policy Evaluation converged after k={k} iteration using eps={eps}.")
            #    break

            vOld = vNew.copy()

########################################################################################################################
GridWorld = My10x10GridWorld([NROWS, NCOLS], starting_state, terminal_states, A,
                             rewards, neg_reward_states, pos_reward_states, Walls,
                             Gamma, v, pi, piProbs, eps, Alpha, epsilon)

whenToPrint = np.array([1, 2, 3, 4, 5, 10, 100, 500, 100000])
noOfIters = 100000
GridWorld.runTD0(whenToPrint, noOfIters)
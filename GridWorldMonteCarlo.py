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
                 Gamma, v, pi, piProbs, eps, Alpha):
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

    ########################
    # Monte-Carlo Learning #
    ########################

    def getReturnForState(self, state, count, totReturn):

        totReturn += self.Gamma**count * self.getRewardForAction(state)
        count += 1

        if self.isTerminalState(state):
            return totReturn
        else:
            # choose action randomly
            rdm_a = self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]
            state_after_action = self.getIndiceAfterAction(state, rdm_a)

            if self.isOutOfGridOrAtWall(state_after_action):
                return self.getReturnForState(state, count, totReturn)
            else:
                return self.getReturnForState(state_after_action, count, totReturn)

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

    def runMonteCarlo(self, whenToPrint, iter):
        """Does Policy Evaluation"""
        vOld = self.v.copy()
        print(f"-- k=0\n{vOld}")

        for k in range(1, iter):
            vNew = self.monteCarloFirstVisitPolicyEvaluation(vOld)
            print("Hallo")
            if k in whenToPrint:
                print(f"-- k={k}\n{vNew}")

            # check for convergence via stopping criterion
            #if np.abs(np.sum(vNew - vOld)) < eps:
            #    print(f"Policy Evaluation converged after k={k} iteration using eps={eps}.")
            #    break

            vOld = vNew.copy()

    def monteCarloEveryVisitPolicyEvaluation(self):
        """ MC Backup:
            V(S_t) = V(S_t) + lambda * (G_t - V(S_t))
        """
        return 0

########################################################################################################################
GridWorld = My10x10GridWorld([NROWS, NCOLS], starting_state, terminal_states, A,
                             rewards, neg_reward_states, pos_reward_states, Walls,
                             Gamma, v, pi, piProbs, eps, Alpha)

whenToPrint = np.array([1, 2, 3, 4, 5, 10, 100])
noOfIters = 2
GridWorld.runMonteCarlo(whenToPrint, noOfIters)
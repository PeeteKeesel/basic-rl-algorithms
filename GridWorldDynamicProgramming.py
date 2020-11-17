"""
Rewrite the GridWorld: DP example from https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html
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

########################################################################################################################
# Policy Iteration: Policy Evaluation & Policy Improvement
########################################################################################################################


class My10x10GridWorld:

    def __init__(self, shape, starting_state, terminal_states, A,
                 rewards, neg_reward_states, pos_reward_states, Walls,
                 Gamma, v, pi, piProbs, eps):
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

    def policyEvaluation(self, vOld):
        """Iterative Policy Evaluation: Do one update of the value function of Iterative Policy Evaluation"""
        vNew = np.zeros((self.NROWS, self.NCOLS))

        # following a random policy we update the value function
        for row in range(self.NROWS):
            for col in range(self.NCOLS):

                if self.isTerminalState([row, col]):
                    vNew[row, col] = self.getRewardForAction([row, col])
                    continue

                if self.isOutOfGridOrAtWall([row, col]):
                    vNew[row, col] = np.inf
                    continue

                current_policy_directions = self.pi[row, col]
                count_of_directions = self.countCharsInString(current_policy_directions)

                # sum over all actions
                for a in A:

                    # get the probabilities depending on the current policy
                    # Ex: if "nse" the prob. would be 1/3
                    if a not in current_policy_directions:
                        continue
                    else:
                        pi_a_given_s = 1 / count_of_directions

                    # get indice of the state after taking the action a
                    i_after_Action = self.getIndiceAfterAction([row, col], a)

                    # Reward from current state s taking Action a
                    # Note: same reward for all actions from state s
                    R_s_a = self.getRewardForAction([row, col])

                    # Get vOld of successor state
                    vOldSuccessor = vOld[row, col] if self.isOutOfGridOrAtWall([i_after_Action[0], i_after_Action[1]]) \
                        else vOld[i_after_Action[0], i_after_Action[1]]

                    # sum over all successor states - NOTE: here only 1 successor state and prob is 1 to
                    # transfer from current state to that state taking action a
                    vUpdate = pi_a_given_s * (R_s_a + Gamma * 1 * vOldSuccessor)

                    # fill in the new value
                    vNew[row, col] += vUpdate

        return np.round(vNew, 2)

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

                vSuccessors = np.zeros(len(self.A))

                for i, a in enumerate(self.A):
                    # get indice of the state after taking the action a
                    i_after_Action = self.getIndiceAfterAction([row, col], a)

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

    def valueIteration(self, vOld):
        """Does Value Iteration: Find optimal Policy + then update policy"""
        vNew = np.zeros((self.NROWS, self.NCOLS))

        # following a random policy we update the value function
        for row in range(self.NROWS):
            for col in range(self.NCOLS):

                if self.isTerminalState([row, col]):
                    vNew[row, col] = self.getRewardForAction([row, col])
                    continue

                if self.isOutOfGridOrAtWall([row, col]):
                    vNew[row, col] = np.inf
                    continue

                # save the value of the Bellmann eq. for each action to then take the max of it
                tempBellmannEqValues = np.array([])

                # iterate over all actions
                for a in A:

                    # get indice of the state after taking the action a
                    i_after_Action = self.getIndiceAfterAction([row, col], a)

                    # Reward from current state s taking Action a
                    # Note: same reward for all actions from state s
                    R_s_a = self.getRewardForAction([row, col])

                    # Get vOld of successor state
                    vOldSuccessor = vOld[row, col] if self.isOutOfGridOrAtWall([i_after_Action[0], i_after_Action[1]]) \
                        else vOld[i_after_Action[0], i_after_Action[1]]

                    # sum over all successor states - NOTE: here only 1 successor state and prob is 1 to
                    # transfer from current state to that state taking action a
                    vUpdate = R_s_a + Gamma * 1 * vOldSuccessor
                    tempBellmannEqValues = np.append(tempBellmannEqValues, vUpdate)

                # fill in the value of the action with the maximal value
                vNew[row, col] = np.max(tempBellmannEqValues)

        return np.round(vNew, 2)

    def runPolicyEvaluation(self, whenToPrint, iter):
        """Does Policy Evaluation"""
        vOld = self.v.copy()
        print(f"-- k=0\n{vOld}")

        for k in range(1, iter):
            vNew = self.policyEvaluation(vOld);
            if k in whenToPrint:
                print(f"-- k={k}\n{vNew}")

            # check for convergence via stopping criterion
            #if np.abs(np.sum(vNew - vOld)) < eps:
            #    print(f"Policy Evaluation converged after k={k} iteration using eps={eps}.")
            #    break

            vOld = vNew.copy()

    def runPolicyImprovement(self, whenToPrint, iter):
        """Does Policy Improvememt = Greedy Policy Improvement"""
        pi0 = np.full([NROWS, NCOLS], "nwse")
        vOld = self.v.copy()
        print(f"-- k=0\n{pi0}")

        for k in range(1, iter):
            vNew = self.policyEvaluation(vOld)
            piNew = self.policyImprovement(vNew)
            if k in whenToPrint:
                print(f"-- k={k}\n{piNew}")

            vOld = vNew.copy()

    def runPolicyIteration(self, whenToPrint, iter):
        """Does Policy Iteration = Policy Evaluation + Greedy Policy Improvement"""
        v = np.zeros((10, 10))
        pi0 = np.full([NROWS, NCOLS], "nwse")
        vOld = self.v.copy()
        print(f"-- k=0\n{self.v}\n{pi0}")

        for k in range(1, iter + 1):
            vNew = self.policyEvaluation(vOld)
            self.v = vNew
            self.pi = self.policyImprovement(vNew)
            if k in whenToPrint:
                print(f"-- k={k}\n{self.v}\n{self.pi}")

            vOld = vNew.copy()

    def runValueIteration(self, whenToPrint, iter):
        vOld = self.v.copy()
        print(f"-- k=0\n{vOld}")

        for k in range(1, iter + 1):
            vNew = self.valueIteration(vOld)
            if k in whenToPrint:
                print(f"-- k={k}\n{vNew}")

            # check for convergence via stopping criterion
            #if np.abs(np.sum(vNew - vOld)) < eps:
            #    print(f"Policy Evaluation converged after k={k} iteration using eps={eps}.")
            #    break

            vOld = vNew.copy()


########################################################################################################################
GridWorld = My10x10GridWorld([NROWS, NCOLS], starting_state, terminal_states, A,
                             rewards, neg_reward_states, pos_reward_states, Walls,
                             Gamma, v, pi, piProbs, eps)

whenToPrint = np.array([1, 2, 3, 4, 5, 10, 100])
noOfIters = 4

whatToDo = input("Press 1 for Policy Evaluation:\n"
                 "      2 for Policy Improvement:\n"
                 "      3 for Policy Iteration:\n"
                 "      4 for Value Iteration: ")

if whatToDo == "1":
    GridWorld.runPolicyEvaluation(whenToPrint, noOfIters)
elif whatToDo == "2":
    GridWorld.runPolicyImprovement(whenToPrint, noOfIters)
elif whatToDo == "3":
    GridWorld.runPolicyIteration(whenToPrint, noOfIters)
elif whatToDo == "4":
    GridWorld.runValueIteration(whenToPrint, noOfIters)

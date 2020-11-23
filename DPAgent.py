
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend base class by implementing 
        Policy Iteration : Iteratively Policy Evaluation & Policy Improvement
        Value Iteration  : Find opt. Value Function & then update Policy"""
class DPAgent(My10x10GridWorld):

    """New methods used for DP"""
    def policyEvaluation(self, vOld):
        """
        Iterative Policy Evaluation: Do one update of the value function of Iterative Policy Evaluation.
        Note: Takes the sum of all successor states.
        """
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

    def policyImprovementExternal(self, vk):
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
        """
        Does Value Iteration: Find optimal value function --> optimal policy + then update policy.
        Note: Always takes the action which leads to the state with the maximal value.

        """
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
            vNew = self.policyEvaluation(vOld)
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
            piNew = self.policyImprovementExternal(vNew)
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
            self.pi = self.policyImprovementExternal(vNew)
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


"""Let the agent reinforce"""
dp_agent = DPAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                  rewards, neg_reward_states, pos_reward_states, Walls,
                  Gamma, v, pi, piProbs, eps, Alpha, epsilon)

whenToPrint = np.array([1, 2, 3, 4, 5, 10, 100])
noOfIters = 4

whatToDo = input("Press 1 for Policy Evaluation:\n"
                 "      2 for Policy Improvement:\n"
                 "      3 for Policy Iteration:\n"
                 "      4 for Value Iteration: ")

if whatToDo == "1":
    dp_agent.runPolicyEvaluation(whenToPrint, noOfIters)
elif whatToDo == "2":
    dp_agent.runPolicyImprovement(whenToPrint, noOfIters)
elif whatToDo == "3":
    dp_agent.runPolicyIteration(whenToPrint, noOfIters)
elif whatToDo == "4":
    dp_agent.runValueIteration(whenToPrint, noOfIters)

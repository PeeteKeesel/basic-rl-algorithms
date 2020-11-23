
from GlobalParams import *
from BaseGridWorldClass import My10x10GridWorld


"""Extend the base class by implementing
        Monte-Carlo"""
class MCAgent(My10x10GridWorld):

    """New methods used for MC"""
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
mc_agent = MCAgent([NROWS, NCOLS], starting_state, terminal_states, A,
                   rewards, neg_reward_states, pos_reward_states, Walls,
                   Gamma, v, pi, piProbs, eps, Alpha, epsilon)

whenToPrint = np.array([1, 2, 3, 4, 5, 10, 100])
noOfIters = 2
mc_agent.runMonteCarlo(whenToPrint, noOfIters)
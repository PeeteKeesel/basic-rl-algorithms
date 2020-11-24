"""
Rewrite the 'GridWorld: DP' example from https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html
"""

from GlobalParams import np


"""The class"""
class My10x10GridWorld(object):

    def __init__(self, shape, starting_state, terminal_states, A,
                 rewards, neg_reward_states, pos_reward_states, Walls,
                 Gamma, v, pi, piProbs, states_encoded, Q, eps, Alpha, epsilon):
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

        self.states_encoded = states_encoded
        self.Q = Q

        self.eps = eps
        self.Alpha = Alpha
        self.epsilon = epsilon

    @staticmethod
    def getIndiceAfterAction(current_state, a):
        """
        Get the indice of the state after taking an action.

        :param
            current_state: 2d-list - the current state the agent is in
            a            : str     - action to take
        :return
            2d-list - the state the agent ends up after taking action
        """
        if a == "n":
            return [current_state[0] - 1, current_state[1]]
        if a == "w":
            return [current_state[0],     current_state[1] - 1]
        if a == "s":
            return [current_state[0] + 1, current_state[1]]
        if a == "e":
            return [current_state[0],     current_state[1] + 1]
        else:
            print(f"Action {a} from state {current_state} is not a feasible input ('n', 'w', 's', 'e').")

    @staticmethod
    def isIn(possible_elem_of_quantity, quantity):
        """
        Checks if a given element lies in a set.

        :param
            possible_elem_of_quantity: np.array              - the element to check
            quantity                 : np.array of np.arrays - the quantity in which the element may lie
        :return
            boolean - True, if the element is in the quantity, False, if not
        """
        return next((True for elem in quantity if np.array_equal(elem, possible_elem_of_quantity)), False)

    @staticmethod
    def countCharsInString(string):
        """
        Counts the chars in a given string.

        :param
            string: str - some string
        :return
            int - the number of chars in the string, without any whitespaces
        """
        return len(string.replace(" ", ""))

    def getIndexInActionList(self, a):
        return self.A.index(a)

    def isOutOfGridOrAtWall(self, state):
        """
        Check if current_state is out of the GridWorld or in a wall.

        :param
            state: 2d-list - a state
        :return
            boolean - True, if the state is in a wall or outside the grid, False, otherwise
        """
        return (not ((0 <= state[0] <= self.NROWS - 1) and (0 <= state[1] <= self.NCOLS - 1))) or \
               self.isIn(state, self.walls)

    def getRewardForAction(self, state):
        """
        Returns the reward an agent gets when leaving its current state.
        Note: In the current scenario the agent always gets the same reward leaving a state no matter which action
              it takes.

        :param
            state: 2d-list - the state the agent is in
        :return
            int - the reward the agent gets when leaving the current state
        """
        if self.isIn(state, self.neg_reward_states):
            return self.rewards['neg']
        elif self.isIn(state, self.pos_reward_states):
            return self.rewards['pos']
        else:
            return 0

    def isTerminalState(self, state):
        """
        Returns if the input state is a terminal state.

        :param
            state: 2d-list - some state
        :return
            boolean - True, if the agent is in a terminal state, False, otherwise
        """
        return self.isIn(state, self.terminal_states)

    def policyImprovementByV(self):
        """
        Does Greedy Policy Improvement using the current state-value function v(s).
        """
        for row in range(self.NROWS):
            for col in range(self.NCOLS):

                if self.isOutOfGridOrAtWall([row, col]):
                    self.pi[row, col] = "XXXX"
                    continue

                vSuccessors = np.zeros(len(self.A)) + np.NINF

                for i, a in enumerate(self.A):
                    # get indice of the state after taking the action a
                    i_after_Action = self.getIndiceAfterAction([row, col], a)

                    if self.isOutOfGridOrAtWall(i_after_Action):
                        continue

                    # Get value of successor state
                    vSuccessor = self.v[row, col] if self.isOutOfGridOrAtWall([i_after_Action[0], i_after_Action[1]]) \
                        else self.v[i_after_Action[0], i_after_Action[1]]

                    vSuccessors[i] = vSuccessor

                # find the indice(s) of the maximal successor values
                maxIndices = [ind for ind, vSuccessor in enumerate(vSuccessors) if vSuccessor == max(vSuccessors)]
                # get the corresponding direction
                directions = [self.A[ind] for ind in maxIndices]

                self.pi[row, col] = "{:<4}".format("".join(directions))

    def takeRandomAction(self):
        return self.A[np.random.choice(np.array(range(0, len(self.A))), size=1)[0]]

    def decodeState(self, state):
        """
        Returns the key for state which is the row of the Q(s,a) table

        :param
            state: 2d-list - some state in the grid
        :return
            int - the key which encodes the state in the dictionary states_decoded
        """
        temp = 0
        for i, state_enc in enumerate(self.states_encoded.values()):
            if np.array_equal(state_enc, state):
                temp = i
        return list(self.states_encoded.keys())[temp]

    def policyImprovementByQ(self):
        """
        Does Greedy Policy Improvement using the current state-action-value function q(s, a).
        """
        for row in range(self.NROWS):
            for col in range(self.NCOLS):

                if self.isOutOfGridOrAtWall([row, col]):
                    self.pi[row, col] = "XXXX"
                    continue
                    
                # choose the action for which q(s, a) is taking the max value
                q_s = self.Q[self.decodeState(np.array([row, col])), :]

                # only those indices (columns=actions) which lead to legal states
                indices_of_legal_actions = [i for i, q in enumerate(q_s) if not
                self.isOutOfGridOrAtWall(self.getIndiceAfterAction([row, col], self.A[i]))]

                # all values of the row in q
                qs = [q for i, q in enumerate(q_s) if self.isIn(i, indices_of_legal_actions)]
                indices_of_max_elems = [m for i, m in enumerate(indices_of_legal_actions)
                                        if q_s[m] == np.max(qs)]

                all_max_actions = []
                for i, a in enumerate(self.A):
                    if self.isOutOfGridOrAtWall(self.getIndiceAfterAction([row, col], a)):
                        continue
                    if i in indices_of_max_elems:
                        all_max_actions.append(a)
                        #self.pi[row, col] += a

                self.pi[row, col] = "{:<4}".format("".join(all_max_actions))


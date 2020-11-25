# KaparthyReinforceJSRedo

Rewriting the code from [.../reinforcejs/gridworld_dp.html](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html). 
Formulas are decoded from [here](https://www.codecogs.com/latex/eqneditor.php).

Implement

- [x] Policy Iteration
- [x] Value Iteration 
- [x] First-Visit Monte Carlo
- [x] Every-Visit Monte Carlo
- [x] TD-0
- [ ] TD(lambda) 
- [x] SARSA
- [x] Q-Learning

ToDo (optional): 
- [ ] write tests
    - [ ] Policy Iteration
    - [ ] Value Iteration
    - [ ] First- and Every-Visit Monte-Carlo
    - [ ] TD-0
    - [ ] SARSA
    - [ ] Q-Learning
    

### 1. Dynamic Programming: Implement Policy Iteration and Value Iteration.

![img](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cgets%20%5Cmathbb%7BE%7D_%7B%5Cpi%7D%5Br_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20V%28S_%7Bt%7D%29%5D)

Implementing code for the simulations on [GridWorld: DP](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)

#### 1.1 __Policy Iteration__: 

- converges to `v_*(s)`

Iteratively doing 

1. Getting ![img](http://latex.codecogs.com/svg.latex?v_i) by doing _Policy Evaluation_ of policy 
![img](http://latex.codecogs.com/svg.latex?%5Cpi_%7Bi-1%7D)
 using ![img](http://latex.codecogs.com/svg.latex?v_%7Bi-1%7D)
followed by 

2. Updating the policy to 
![img](http://latex.codecogs.com/svg.latex?%5Cpi_i) by doing _Policy Improvement_ using 
![img](http://latex.codecogs.com/svg.latex?v_i)

#### 1.2 __Value Iteration__:

- converges to `v_*(s)`

1. Finding the _optimal value function_ 
![img](http://latex.codecogs.com/svg.latex?v_%2A) using only 
the action which leads to the successor state with maximal value followed by 

2. One update of the policy using the optimal found policy from 1.
    
__Investigations__: Doing iteratively first policy evaluation followed by policy improvement
and then using that improvement policy in the next evaluation step leads to the same
value function for each iteration as in value iteration. I.e. Directly updating the policy 
and then evaluation leads to the same as always only taking the action which leads to
the successor state with maximal value.

### 2. Monte Carlo

![img](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cgets%20V%28S_t%29%20&plus;%20%5Calpha%5BR_%7Bt%7D%20-%20V%28S_t%29%5D)

- learn state-value function for a given policy
- update only when end of episode is reached
- converges to `v_pi(s)` as the number of visits to `s` goes to infinity
- only sample experience is available, no complete probability 
distributions of all possible transitions are required

__Idea__: Estimate the value of a state by experience by _averaging the returns observed after 
visiting that state_. As more returns are observed, the average should converge to the 
expected value.

__visit__ to `s` = each occurence of state `s` in an episode.

__First-visit MC method__ = estimates `v_pi(s)` as the average of the returns following
first visits to `s`.    

__Every-visit MC method__ = estimates `v_pi(s)` as the average of the returns following
all visits to `s`.    

### 3. Temporal Difference Learning

- update online, after each step

Implementing code for the simulations on [GridWorld: TD](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html)

#### 3.1 __TD(0)__

- converges to `v_pi(s)`

![img](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cgets%20V%28S_t%29%20&plus;%20%5Calpha%5Br_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20V%28S_%7Bt&plus;1%7D%29%20-%20V%28S_t%29%5D)

- learn state-value function for a given policy

__Investigations__: _TD(0)_ converges to the __correct__ answer (value function/policy)
but not to the optimal policy!

#### 3.2 __SARSA__

- converges to `Q_*(s, a)`

![img](https://latex.codecogs.com/gif.latex?Q%28S_t%2C%20A_t%29%20%5Cgets%20Q%28S_t%2C%20A_t%29%20&plus;%20%5Calpha%20%5B%20R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20Q%28S_%7Bt&plus;1%7D%2C%20A_%7Bt&plus;1%7D%29-%20Q%28S_t%2C%20A_t%29%20%5D)

#### 3.3 __Q-Learning__

- converges to `Q_*(s, a)`

![img](https://latex.codecogs.com/gif.latex?Q%28S_t%2C%20A_t%29%20%5Cgets%20Q%28S_t%2C%20A_t%29%20&plus;%20%5Calpha%20%5B%20R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%7DQ%28S_%7Bt&plus;1%7D%2C%20A_%7Bt&plus;1%7D%29-%20Q%28S_t%2C%20A_t%29%20%5D)

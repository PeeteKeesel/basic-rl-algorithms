# KaparthyReinforceJSRedo

Rewriting the code from [.../reinforcejs/gridworld_dp.html](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html). 

Implement

- [x] Policy Iteration
- [x] Value Iteration 
- [ ] First-Visit Monte Carlo
- [ ] Every-Visit Monte Carlo
- [x] TD-0
- [ ] TD(lambda) 
- [ ] SARSA
- [ ] Q-Learning

### 1. Dynamic Programming: Implement Policy Iteration and Value Iteration.

![img](https://latex.codecogs.com/gif.latex?V%28S_t%29%20%5Cgets%20%5Cmathbb%7BE%7D_%7B%5Cpi%7D%5Br_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20V%28S_%7Bt%7D%29%5D)

Implementing code for the simulations on [GridWorld: DP](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)

__Policy Iteration__: 
Iteratively doing 

1. Getting ![img](http://latex.codecogs.com/svg.latex?v_i) by doing _Policy Evaluation_ of policy 
![img](http://latex.codecogs.com/svg.latex?%5Cpi_%7Bi-1%7D)
 using ![img](http://latex.codecogs.com/svg.latex?v_%7Bi-1%7D)
followed by 

2. Updating the policy to 
![img](http://latex.codecogs.com/svg.latex?%5Cpi_i) by doing _Policy Improvement_ using 
![img](http://latex.codecogs.com/svg.latex?v_i)

__Value Iteration__:

1. Finding the _optimal value function_ 
![img](http://latex.codecogs.com/svg.latex?v_%2A) using only 
the action which leads to the successor state with maximal value followed by 

2. One update of the policy using the optimal found policy from 1.
    
__Investigations__: Doing iteratively first policy evaluation followed bu policy improvement
and then using that improvement policy in the next evaluation step leads to the same
value function for each iteration as in value iteration. I.e. Directly updating the policy 
and then evaluation leads to the same as always only taking the action which leads to
the successor state with maximal value.

### 2. Monte Carlo



### 3. Temporal Difference Learning

Implementing code for the simulations on [GridWorld: TD](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html)

__TD(0)__

- start in a start state
- update current state using a 1-step-lookahead
- continue value estmation with the state you
ended up after the 1-step-lookahead
- directly update the policy = online
- continue procedure until final state 
or specifid number of iterations is reached
- go back to start state and continue 

__Investigations__: _TD(0)_ converges to the __correct__ answer (value function/policy)
but not to the optimal policy!

__SARSA__

__Q-Learning__
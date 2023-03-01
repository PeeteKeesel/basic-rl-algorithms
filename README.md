# :robot: Basic RL Algorithms

Writing code to obtain the results of the simulations from Kaparthy's [.../reinforcejs/gridworld_dp.html](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html). 

## :books: Table of Contents

- [:robot: Basic RL Algorithms](#robot-basic-rl-algorithms)
  - [:books: Table of Contents](#books-table-of-contents)
  - [:triangular\_ruler: Description of the Mathematical Components](#triangular_ruler-description-of-the-mathematical-components)
  - [:tractor: Dynamic Programming](#tractor-dynamic-programming)
    - [Policy Iteration](#policy-iteration)
    - [Value Iteration](#value-iteration)
  - [:blue\_car: Monte Carlo](#blue_car-monte-carlo)
  - [:railway\_car: Temporal Difference (TD) Learning](#railway_car-temporal-difference-td-learning)
    - [TD(0)](#td0)
    - [SARSA](#sarsa)
    - [Expected SARSA](#expected-sarsa)
    - [Q-Learning](#q-learning)
  - [:dart: Summaries](#dart-summaries)
    - [Problems](#problems)
    - [Algorithms](#algorithms)
  - [:calendar: ToDo](#calendar-todo)

## :triangular_ruler: Description of the Mathematical Components

| Component | Description |
| --------- | ----------- |
| $S$ | State in the enviroment |
| $A$ | Action of a policy |
| $Q(S,A)$ | Estimated value of taking action $A$ in state $S$ |
| $Q(S_{t+1},A_{t+1})$ | Estimated value of taking the next action $A_{t+1}$ in the next state $S_{t+1}$ under the current policy $\pi$ |
| $\pi$ | Policy of the RL agent, which maps states $S$ to actions $A$ |
| $\pi(A_{t+1} \| S_{t+1})$ | Probability of selecting action $A_{t+1}$ in state $S_{t+1}$ under the current policy $\pi$ |
| $R_t$ | Immediate reward received from taking action $A$ in state $S$ |
| $R_{t+1}$ | Received reward after taking action $A_{t+1}$ in state $S_{t+1}$ |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $\sum_a$ | Sum over all possible actions $A$ in state $S$ |

## :tractor: Dynamic Programming

$$
    V(S_t) \leftarrow \mathbb{E}_{\pi} [ r_{t+1} + \gamma V(S_t) ]
$$

Implementing code for the simulations on [GridWorld: DP](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)

### Policy Iteration

- converges to $v_*(s)$

<u>Steps</u>: Iteratively

1. Getting $v_i$ by doing _Policy Evaluation_ of policy $\pi_{i-1}$ using $v_{i-1}$.

2. Updating the policy to $\pi_i$ by doing _Policy Improvement_ using $v_i$.

### Value Iteration

- converges to $v_*(s)$

<u>Steps</u>:

1. Finding the _optimal value function_ $v_*$ using only the action which leads to the successor state with maximal value.

2. One update of the policy using the optimal found policy from 1.
    
<u>Investigations</u>: Doing iteratively first policy evaluation followed by policy improvement and then using that improvement policy in the next evaluation step leads to the same value function for each iteration as in value iteration. I.e. Directly updating the policy and then evaluation leads to the same as always only taking the action which leads to the successor state with maximal value.

## :blue_car: Monte Carlo

$$
V(S_t) \leftarrow V(S_t) + \alpha [ R_t -V(S_t) ]
$$

- learn state-value function for a given policy
- update only when end of episode is reached
- converges to $v_{\pi}(s)$ as the number of visits to $s$ goes to infinity
- only sample experience is available, no complete probability 
distributions of all possible transitions are required

<u>Idea</u>: Estimate the value of a state by experience by _averaging the returns observed after 
visiting that state_. As more returns are observed, the average should converge to the 
expected value.

- <u>visit</u> to $s$ = each occurence of state $s$ in an episode.

- <u>First-visit MC method</u> = estimates  $v_{\pi}(s)$ as the average of the returns following
first visits to $s$.    

- <u>Every-visit MC method</u> = estimates  $v_{\pi}(s)$ as the average of the returns following
all visits to $s$.    

## :railway_car: Temporal Difference (TD) Learning

- update online, after each step

Implementing code for the simulations on [GridWorld: TD](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html)

### TD(0)

- converges to $v_{\pi}(s)$

$$
    V(S_t) \leftarrow V(S_t) + \alpha \left[ r_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
$$

- learn state-value function for a given policy

<u>Investigations</u>: _TD(0)_ converges to the __correct__ answer (estimated value function for a given policy) but not to the optimal policy!

### SARSA
- used to estimate optimal action-value function $Q_*(s, a)$
- converges to $Q_*(s, a)$

$$
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ] 
$$

### Expected SARSA 
- used to estimate optimal action-value function $Q_*(s, a)$
- converges to $Q_*(s, a)$

$$
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma \sum_a \pi(A_{t+1} | S_{t+1}) Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ]
$$

- similar to SARSA
- but takes expected value over all possible actions $a$ instead of using the actual next action to estimate the next state-action value

### Q-Learning

- converges to $Q_*(s, a)$

$$
    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma \max_a Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ] 
$$


---

## :dart: Summaries

### Problems

| Problem    | Goal                                                           | Examples                               |
| :--------- |:-------------------------------------------------------------- | :------------------------------------- |
| Prediction | __evaluate__ a given policy  <br> _How much reward are we going to get for a given policy?_  | Iterative Policy Evaluation, TD(lambda) <br> First-Visit MC, Every-Visit MC|
| Control    | find the __optimal__ policy  <br> _What is the most total reward we are getting out of our MDP?_ | Policy Iteration, Value Iteration, <br>SARSA, Q-Learning, <br> MC Exploring Starts, On-Policy first-visit MC control                |

### Algorithms

| Algorithm                   | Update Equation | Type       | Description | 
| :-------------------------- | :---------------| :--------- | :-----------|
| Iterative Policy Evaluation | $V(s) \leftarrow \sum_a \pi(a \| s) \sum_{s', r} p(s', r \| s, a) [ r + \gamma V(s') ]$ | Synchronous DP | evaluate a given $\pi$ <br> - there is an explicit policy  |  
| Policy Iteration            | 1. Policy Evaluation <br> $V(s) \leftarrow \sum_a \pi(a \| s) \sum_{s', r} p(s', r \| s, \pi(s)) [ r + \gamma V(s') ]$ <br> 2. Policy Improvement <br> $\pi{s} \leftarrow \max_a \sum_{s', r} p(s', r \| s, a)[ r + \gamma V(s') ]$ | Synchronous DP | evaluate a given $\pi$ via _Bellmann Expectation Eq._ + update $\pi$ <br> - there is an explicit policy | 
| Value Iteration             |  $V(s) \leftarrow \max_a \sum_{s', r} p(s', r \| s, a)[ r + \gamma V(s') ]$ | Synchronous DP | evaluate a given a $\pi$ via _Bellmann Optimality Eq._ <br> - there is no explicit policy|
| First-Visit-MC              | ... | MC-Learning    | estimates $v(s)$ as the average of the returns following first-visits to $s$ |`
| Every-Visit-MC              | ... | MC-Learning    | estimates $v(s)$ as the average of the returns following every-visits to $s$ |
| TD(0)                       | $V(S_t) \leftarrow V(S_t) + \alpha \left[ r_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$ | TD-Learning    | |
| n-step TD                   | ... | TD-Learning    | |
| SARSA                       | $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ]$ | TD-Learning    | estimate $q_{\pi}$ following $\pi$ + update $\pi$ <br> - performs on-policy updates <br> - randomly select $A_{t+1}$  | 
| Q-Learning                  | $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma \max_a Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) ]$ | TD-Learning    | estimate $q_{\pi}$ following optimal next state-actions <br> - performs off-policy updates (approx. $q^*$ ind. of policy) <br> - select $argmax_a Q(S_{t+1}, A_{t+1})$ | 
| Expected SARSA              | $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma \sum_a \pi(a \| S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) ]$ | TD-Learning    | estimate $q_{\pi}$ using expected value of next state-actions <br> - performs off-policy updates <br> - randomly select $A_{t+1}$ <br> - moves _deterministically_ in same direction as SARSA moves in _expectation_ |

## :calendar: ToDo
- [x] Policy Iteration
- [x] Value Iteration 
- [x] First-Visit Monte Carlo
- [x] Every-Visit Monte Carlo
- [x] TD-0
- [ ] TD(lambda) 
- [x] SARSA
- [x] Q-Learning
- [ ] Include eligibility traces
- [x] Update readme
  - [x] include formulas and descriptions
- [ ] Include picture of the grid world
- [ ] Make separate main which runs the specific agent simulation
- [ ] Investigate the slowliness of SARSA

<u>Optional</u>
- [ ] Write unit-tests
    - [ ] Policy Iteration
    - [ ] Value Iteration
    - [ ] First- and Every-Visit Monte-Carlo
    - [ ] TD-0
    - [ ] SARSA
    - [ ] Q-Learning
# KaparthyReinforceJSRedo

Rewriting the code from [.../reinforcejs/gridworld_dp.html](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html). 

#### Dynamic Programming: Implement Policy Iteration and Value Iteration.

__Policy Iteration__: 
Iteratively doing 

1. Getting $v_i$ by doing _Policy Evaluation_ of policy 
<img src="https://render.githubusercontent.com/render/math?math=\pi_{i-1}">
 using <img src="https://render.githubusercontent.com/render/math?math=v_{i-1}">
followed by 

2. Updating the policy to 
<img src="https://render.githubusercontent.com/render/math?math=\pi_i"> by doing _Policy Improvement_ using 
<img src="https://render.githubusercontent.com/render/math?math=v_i">

__Value Iteration__:

1. Finding the _optimal value function_ 
<img src="https://render.githubusercontent.com/render/math?math=v_*"> using only 
the action which leads to the successor state with maximal value followed by 

2. One update of the policy using the optimal found policy from 1.
    
__Investigations__: Doing iteratively first policy evaluation followed bu policy improvement
and then using that improvement policy in the next evaluation step leads to the same
value function for each iteration as in value iteration. I.e. Directly updating the policy 
and then evaluation leads to the same as always only taking the action which leads to
the successor state with maximal value.
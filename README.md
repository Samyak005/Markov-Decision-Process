# Value-Iteration Algorithm
# Task 1 
Took 124 iterations to converge\
Reaching the end state for the given 2 states: 

(W,0,0,D,100) RIGHT\
(C,0,0,D,100) RIGHT\
(E,0,0,D,100) HIT\
(E,0,0,D,50) HIT\
(E,0,0,D,0)NONE\
Simulation Over

(C,2,0,R,100) UP\
(N,2,0,R,100) CRAFT\
(N,1,2,R,100) CRAFT\
(N,0,3,R,100) STAY\
(N,0,3,D,100) DOWN\
(C,0,3,D,100) RIGHT\
(E,0,3,D,100) SHOOT\
(E,0,2,D,75) HIT\
(E,0,1,D,25) SHOOT\
(E,0,1,D,0) NONE\
Simulation Over

1. If IJ is on centre with no arrows and MM is in dormant state, IJ prefers to go right and hit MM
2. If IJ is on centre with some arrows and MM has 25 health, IJ prefers to shoot, as terminal reward is high
3. If IJ is on centre with even some arrows and MM is in ready state, IJ prefers to go down if he doesn’t have material otherwise up as he is safe there
4. If IJ is in the east position, he prefers to hit when MM has more health or he is without arrows. He prefers to shoot when he has arrows or if MM is running low on health. (Expected damage is more for shoot)
5. If IJ is in the north position he prefers crafting arrows if has material. He prefers down if MM is in dormant state else he stays if MM is in a ready state.
6. If IJ is in south position he prefers gathering or staying when MM is in ready state. He prefers up when MM is in dormant state.
7. If IJ is in west position, he moves right if he has arrows because at centre position probability of arrow success is high. He prefers shoot when MM has less health. 

# TASK 2
## Case 1
Took 125 iterations, 1 more than task 1\
Intuition : As IJ can now go directly to west square from east square when MM goes to ready state where he can’t be hit, so he becomes less afraid of going east.

No change in actions of last iteration is observed from trace file. Optimal policy doesn’t change. 

## Case 2
62 iterations to converge as game has become easy because stay action has no penalty\
Strategy becomes defensive as IJ never hits MM
1. If IJ is on centre he prefers going left
2. If IJ is in west he prefers staying
3. If IJ is in east he shoots if MM health is low, otherwise stays
4. If IJ is in north and has arrows he goes down otherwise prefers staying
5. If IJ is in south and MM is in dormant he goes up else stays when MM is ready. If no arrows prefers staying

End state would never be reached if IJ is on west square with 0 arrows, 0 materials, MM is in ready state. 

## Case 3
If discount factor is higher agent gives more importance to future states and looks much ahead into future.
In this case gamma is significantly lower.

8 iterations to converge
1. If IJ is in centre he directly hits or shoots(if arrows) without thinking that probability of hitting/shooting would increase on going east. He also doesn’t look for making arrows and prefers going west to avoid being hit by MM
2. If IJ is in east he prefers hitting/shooting if MM is in dormant state. He prefers shooting over hitting if MM has low health. Else he moves left if MM in ready state.
3. If IJ is in north, if has material he crafts, if MM in ready state he stays, otherwise he prefers down if has arrows
4. If IJ is in south, he gathers extensively. Sometimes he also moves up.
5. If IJ is in west, he shoots without thinking about where he can go to increase his probability of success. He stays/right otherwise.
6. End state would never be reached if IJ is on west square with 0 arrows, 0 materials, MM is in ready state. 

# Linear Programming
The step cost is ​ -20​.
## Initializing A matrix

For every action state pair we fill the transition probability for the final state in the matrix. We traverse the matrix column-wise. If action is NONE\
we fill 1 where initial and final state match, else 0. The general procedure is to subtract the transition probability where initial state!=final state and add the same value where initial state==final state.

```python
column_index = 0
for initial_state in self.states:
    for action in self.possible_actions[initial_state]:
        action = self.convert_back[action]
        # row_index = 0
        if action == "NONE":
            self.A[self.reverse_hash_state[initial_state]][column_index] = 1
            column_index += 1
            continue

        for final_state in self.transition_prob[initial_state][action].keys():	
            self.A[self.reverse_hash_state[final_state]][column_index] -= self.transition_prob[initial_state][action][final_state]
            self.A[self.reverse_hash_state[initial_state]][column_index] += self.transition_prob[initial_state][action][final_state]
            # row_index += 1
        column_index += 1
```
## Calculating Policy
The LP maximizes the value of r.x where constraints are A.x = alpha and x>=0\
It is solved using cvxpy library.

## Analysis
"C,2,3,R,100,UP" - Start State\
"N,2,3,R,100,STAY"\
"N,2,3,D,100,DOWN"\
"C,2,3,D,100,RIGHT"\
"E,2,3,D,100,SHOOT"\
"E,2,2,D,75,SHOOT"\
"E,2,1,D,50,SHOOT"\
"E,2,0,D,25,HIT"\
"E,2,0,D,0,NONE"\
Simulation Over

Very similar behaviour as compared to VI algorithm analysis
## Can there be multiple policies?
Yes, there can be multiple policies.
1. Commutability: This occurs
when more than one action has the maximum utility. This results
in the existence of multiple optimal policies.
2. The chosen policy depends on how the ties are broken​. Out of the
multiple maximums, any action can be chosen. In our code, it depends on the order in which actions are defined.
3. If step cost/reward for actions is changed slightly the policy would change. Utility values would change only slightly.
4. If the order of actions chosen is different, then the A
matrix would be different(because the column traversal depends on which action comes first), the R matrix would change, the x array would be
different(only some columns corresponding to the 2 actions get swapped). The objective value and alpha matrix would be the same​. So changing
the preference of actions may lead to a change in the chosen
policy.
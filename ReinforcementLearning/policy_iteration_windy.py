# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:54:53 2020

@author: Guru Prasad Mupppana

Optimizing the policy.

We select first random selection for policy and optimize the value function.
Once we optimize the value function, we optimize the policy function.

Note: Action one takes not full deterministic.  In practical, from the same state, you will not be taking
the same deterministic action while there is a preferred action for a given state.

It is similar to like windy scenario. The agents wants to move specific direction but due to heavy wind,
there are chances to move to other directions.

We model this windy scenarios with the probabilities.
For example, we can make the preferred direction with 50% probability and the rest is given
0.5/ len(other options at that given time)

"""

import numpy as np
from Grid import negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')
GAMMA = 0.9 # future decaying rewards

if __name__ == "__main__" : 
    print("The main function starts here")
    
    # Create a negative grid.
    g = negative_grid(step_cost=-1.0)
    # print values
    all_states = g.all_states()

    print_values(g.rewards,g)
    
    V = {} # empty dictionary for holding states
    for s in all_states:
        if s in g.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0 # Terminal states which do not have any actions.
    
#    print_values(V,g)
    
    policy = {}
    for s in all_states:
        if s in g.actions:
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS) # initialize the policies with random actions
            # NOTE: each state has only one action item.
#        else: 
#            policy[s] = None # Do we need this ?
            
    print_policy(policy,g)
        
    
    while True:
        
        # Now find out value function with randomly initized policy.
        count = 0
        while True:
            biggest_change = 0
            
            for s in all_states: # for every state s
                v_old = V[s]
                v_new = 0
    
                if s in policy: # if it is not terminal state. We assume that there is only one action for a given state
                    p = 0 # probabilities
                    # since it is a windy scenario, we will we try to give preferred direction as 0.5
                    # then rest is distributed equally
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/(3) # since we have four actions
                        
                        a = policy[s]
                        g.set_state(s)
                        r = g.move(a)
                        v_new += p*( r + GAMMA*V[g.get_current()])
                    V[s] = v_new
                    biggest_change = max(biggest_change, np.abs(v_new -v_old))
            
            count +=1
            #print("count:",count)
            if (biggest_change < SMALL_ENOUGH ):
                print("Coompleted optimizing the value function")
                break
            
        # end of identification of Value function. 
        #print_values(V,g)
        
        is_policy_converged = True
        
        for s in all_states :
            if s in policy:
                a_old = policy[s]
                a_new = None
                best_value = float('-inf') # Maximum negative value
                #p = 0 
                
                for pre_action in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == pre_action :
                            p = 0.5
                        else:
                            p = 0.5/3

                        g.set_state(s)
                        r = g.move(a)
                        v += p*(r + GAMMA*V[g.get_current()])
                    if v > best_value:
                        best_value = v
                        a_new = pre_action
                
                policy[s] = a_new
  
                if a_old != a_new :
                    is_policy_converged = False
                    
        print("Coompleted optimizing the Policy function")
        if is_policy_converged :
            break
    
    
print("values/rewards:")
print_values(V,g)
print("Optimal Policy")
print_policy(policy,g)    
    
            





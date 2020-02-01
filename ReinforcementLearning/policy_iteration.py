# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:54:53 2020

@author: Guru Prasad Mupppana

Optimizing the policy.

We select first random selection for policy and optimize the value function.
Once we optimize the value function, we optimize the policy function.

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
    g = negative_grid()
    # print values
    all_states = g.all_states()
    
    V = {} # empty dictionary for holding states
    for s in all_states:
        if s in g.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0 # Terminal states which do not have any actions.
    
    print_values(V,g)
    
    policy = {}
    for s in all_states:
        if s in g.actions:
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS) # initialize the policies with random actions
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
    #            v_new = 0
                if s in policy: # if it is not terminal state. We assume that there is only one action for a given state
                    a = policy[s]
                    g.set_state(s)
                    r = g.move(a)
                    V[s] = r + GAMMA*V[g.get_current()]
                    biggest_change = max(biggest_change, np.abs(V[s] -v_old))
            
            count +=1
            #print("count:",count)
            if (biggest_change < SMALL_ENOUGH ):
                print("Coompleted optimizing the value function")
                break
            
        # end of identification of Value function. 
        print_values(V,g)
        
        is_policy_converged = True
        
        for s in all_states :
            if s in policy:
                a_old = policy[s]
                a_new = None
                best_value = float("-inf") # Maximum negative value
                
                for a in ALL_POSSIBLE_ACTIONS:
                    g.set_state(s)
                    r = g.move(a)
                    v = r + GAMMA*V[g.get_current()]
                    if v > best_value:
                        best_value = v
                        a_new = a
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
    
            





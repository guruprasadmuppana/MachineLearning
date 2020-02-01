# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 06:27:36 2020

@author: Guru Prasad Muppana

"""

import numpy as np
from Grid import standard_grid

SMALL_ENOUGH = 1e-3

# print R (rewards ) in the grid
def print_values(V,g):
    
    for i in range(g.rows):
        print("________________________")
        for j in range(g.cols):
            #print("|")
            v = V.get((i,j),0) # get a value from (i,j) and if it has no value, return zero
            if v >= 0 :
                print(" %.2f|" % v,end="")
            else:
                print("%.2f|" % v,end="")
        print("") # next line
    print("________________________")


def print_policy(P,g):
    
    for i in range(g.rows):
        print("________________________")
        for j in range(g.cols):
            #print("|")
            p = P.get((i,j),' ') # get a value from (i,j) and if it has no value, return empty space
            print("  %s  |" % p,end="")
        print("") # next line
    print("________________________")
 




if __name__ == "__main__" :
    print("main function starts here")
    g = standard_grid()
    
    states = g.all_states()
    # V contains all value for each state
    V = {} # it is a disctionary of states with R value
    for s in states:
        V[s] = 0 # initiall all states with zero value.
#        print("(  {0}, {1})".format(s[0], s[1]))
#
#    print("__________")

    #
    #print_values(V,g)
    gamma = 1 # No discoint.
    
    while True:
        biggest_change = 0
        for s in states:
            v_old = V[s] # store previous V[s]
            
            if s in g.actions:
                v_new = 0
                pr = 1.0/(len(g.actions[s])) # uniform distribuion.
                for a in g.actions[s]:
                    
                    g.set_state(s) # Mark s as current state
                    r = g.move(a) # get a reward for taking action at state s and action a
#                    current_state = g.get_current() 
#                    print("(  {0},  {1})".format(current_state[0], current_state[1]))
                    v_new += pr*( r + gamma*V[g.get_current()])

#                    current_state = g.get_current() 
#                    print("( Set State:  {0},  {1})".format(s[0], s[1]))
#                    print("action",a)
#                    r = g.move(a)
#                    print("( Current state:  {0},  {1})".format(current_state[0], current_state[1]))
#                    print("c1 ",g.get_current())
#                    v_new += pr * (r + gamma * V[g.get_current()])
      
                V[s] = v_new # add to old / previous V(s) value
                biggest_change = max(biggest_change,np.abs(v_new - v_old)) # why cannnot use np.max()
            
        
        if biggest_change < SMALL_ENOUGH :
            print("biggest change occured at ({0},{1}) with value {2}".format( s[0], s[1], biggest_change))
            break # when the values convertion
            
    # print V values.
    print("Values for uniformly random actions:")
    print_values(V,g)
    print("\n\n")
    
    
      ### fixed policy ###
    policy = {
             (2, 0): 'U',
             (1, 0): 'U',
             (0, 0): 'R',
             (0, 1): 'R',
             (0, 2): 'R',
             (1, 2): 'R',
             (2, 1): 'R',
             (2, 2): 'R',
             (2, 3): 'U',
    }
    
    print_policy(policy, g)

    V = {} # it is a disctionary of states with R value
    for s in states:
        V[s] = 0 # initiall all states with zero value.
#        print("(  {0}, {1})".format(s[0], s[1]))
#
#    print("__________")

    #
    #print_values(V,g)
    gamma = 0.9 # No discoint.
    
    while True:
        biggest_change = 0
        for s in states:
            v_old = V[s] # store previous V[s]
            
            if s in policy: # Fixed policy instead of uniform distribution.
                v_new = 0
                for a in policy[s]:
                    
                    g.set_state(s) # Mark s as current state
                    r = g.move(a) # get a reward for taking action at state s and action a
#                    current_state = g.get_current() 
#                    print("(  {0},  {1})".format(current_state[0], current_state[1]))
                    
                    # when the policy is fixed, the probability to move for a given position is 1.
                    # hence Pr = 1 ... it is removed.
                    pr = 1
                    v_new += pr*( r + gamma*V[g.get_current()])

      
                V[s] = v_new # add to old / previous V(s) value
                biggest_change = max(biggest_change,np.abs(v_new - v_old)) # why cannnot use np.max()
            
        
        if biggest_change < SMALL_ENOUGH :
            print("biggest change occured at ({0},{1}) with value {2}".format( s[0], s[1], biggest_change))
            break # when the values convertion
            
    # print V values.
    print("Values for uniformly random actions:")
    print_values(V,g)
    print("\n\n")
    
      
    
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:12:23 2020

@author: Grid - Based grid for playing the game. 
In this game, one has to travel to the final poins (Success with R = 1) and the agent can endup in
the failure state (R with -1).  There exist a path to reach succcess state.
At each state, one can make a specific set of actions
We call this Action space. In this case, the action space is (Up(U), Right(R), Left (L), Down (D)).

"""

class Grid:
    # Grid, in this case defined with rows and columns.
    # position of (i,j) tuple represent the state of the grid.
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        # (i,j) is the current state
        self.i = start[0] # first element of start state is i and second elsement is j
        self.j = start[1]
    
    # Set functon initializes each state with appropiate rewards.
    # note that Rewards static in this case (off line training)
    # actions is a set of actions that is possible at the state (s =[i,j])
    def set(self, rewards, actions):
        
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards  # hashed by state. i.e a dictionary with index state and value for the state
        self.actions = actions # hashed by state. i.e a dict with index state and action list is a string.
                            # each string contains actions letters like RUL -? right, up, left
    
    # sets the state s to the current state of the grid.
    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]
        
    # Returns the current state tuple in the grid.
    # ideally, it can get_current_state    
    def get_current(self) :
        return (self.i,self.j)
        
    # returns bool value based whether the current state is terminal or not.
    # A terminal state is defined by "not having any action set"
    # It will also contains the rewards of Max or Min.
    def is_terminal(self, s):
        #return (self.i,self.j)
        return s not in self.actions # Note actions is a dictionary of actions for hashed by state s
    
    # check if action is possible. if it is possible, it moves the current state.
    # after moving, it will return reward point in the new current state.
    def move(self, action):
        if action in self.actions[(self.i,self.j)] :
            # i, j starts (0,0) at the top left cornor of the grid.
            # i moves from top to down ... value increase from 0 to max value of i.
            # Up move is noting but decrementing value in 'i' value.
            if action == 'R' :
                self.j +=1 # increment i value by 1 to move right
            if action == 'L' :
                self.j -=1
            if action == 'U' :
                self.i -=1
            if action == 'D' :
                self.i +=1
                
        return self.rewards.get((self.i,self.j),0)
        
        
    def unmove(self,action):
        # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
          self.i += 1
        elif action == 'D':
          self.i -= 1
        elif action == 'R':
          self.j -= 1
        elif action == 'L':
          self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert(self.current_state() in self.all_states())
        
    def game_over(self): # return ture value when there is "no" action list in the state.
        # if the current state has no actions:
        #return self.actions[(self.i,self.j)] - > return null value. not bool 
        return (self.i,self.j) not in self.actions
    
    def all_states(self):
        # returns all set of states. It is set of stats without terminal (success and failure)
        return set(self.actions.keys() | self.rewards.keys
                   ()) # return the unique values only.
    
# Grid class ends here.    
    
def standard_grid():
   # . . . 1
   # . x . -1
   # s . . .
   g = Grid(3,4,(2,0)) # a grid with 3x4 with start state is (2,0)
   # There are two cells have rewards.
#   g.rewards[(0,3)] = 1
#   g.rewards[(1,3)] = -1
   rewards = {(0, 3): 1, (1, 3): -1}
   actions = {
           (0,0) : ('R','D'),
           (1,0) : ('U','D'),
           (2,0) : ('U','R'),
           
           (0,1) : ('L','R'),
           #(1,1) : not valid position
           (2,1) : ('L','R'),
           
           (0,2) : ('L','R','D'),
           (1,2) : ('U','R','D'),
           (2,2) : ('L','R','U'),
           
           #(0,3) : ('L','R','D') Success point
           #(1,3) : ('U','R','D') # failure point.
           (2,3) : ('U','L')
           
    #        (0, 0): ('D', 'R'),
    #        (0, 1): ('L', 'R'),
    #        (0, 2): ('L', 'D', 'R'),
    #        (1, 0): ('U', 'D'),
    #        (1, 2): ('U', 'D', 'R'),
    #        (2, 0): ('U', 'R'),
    #        (2, 1): ('L', 'R'),
    #        (2, 2): ('L', 'R', 'U'),
    #        (2, 3): ('L', 'U'),
           
           }   
   
   g.set(rewards,actions)   
   return g

def negative_grid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  return g





   
    
    
    
    
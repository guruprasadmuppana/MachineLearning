# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:42:50 2020

@author: Guru Prasad Muppana
"""

# imports:

import numpy as np

LENGTH = 3 # represting 3x3 matric board

# Differnent type of classes

# Agent uses Epsilon-Greedy algorithm
class Agent :
    def __init__(self, eps=0.1,alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []
        
    def setV(self, V) : # V is value of Value function
        self.V = V
        
    def set_symbol(self, sym): # is it X or O
        self.sym = sym
    
    def set_verbose(self, v):
        self.verbose = v
    
    def reset_history(self): # Once the game is over, we need to reset the state_ history
        self.state_history= []
        
    def take_action(self, env): # Make next move. Use epsilon greedy algo.
        # Take random state values is less than eps. Take the action from the possible moves 
        # else take best value from state history
        
        # Taking action random from given state.
        r = np.random.rand() # gives a random number but it is not normalized.
        best_state = None
        if r < self.eps:
            if self.verbose == True:
                print("Taking a random action")
            possible_moves = [] # The list of possions which are empty.
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_moves.append((i,j)) # (i,j )th position : Checking all possible postions across the board.
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else : # case of chossing the best action.
            # Chose the best action based on teh current values of states.
            pos2value = {} # Curly brackets?
            next_move = None
            best_value = - 1 # Why ?
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        env.board[i,j] = self.sym
                        state = env.get_state()
                        env.board[i,j] = 0 # reset to the orginal value i.e empty value
                        pos2value[(i,j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i,j)
            if self.verbose :
                print ("Taking a greedy action ")
                for i in range(LENGTH):
                    print("____________________")
                    for j in range(LENGTH):
                        if env.is_empty(i,j):
                            print("%.2f |" % pos2value[(i,j)], end="" )
                        else:
                            print("   ", end="")
                            if env.board[i,j] == env.x :
                                print("X  |", end="")
                            elif env.board[i,j] == env.o :
                                print("O  |", end="")
                            else: 
                                print("  |", end="")
                    print("")
                print("____________________")
    
        #Make a mext move :
        env.board[next_move[0],next_move[1]] = self.sym
        
    def update_state_history(self,s):
        
        self.state_history.append(s)
        
    def update(self,env):
        # Backtrack over the states:
        
        # V[prev_state] = V[prev_state] + aplph*(v[next_state] - V[prev_state])
        
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed (self.state_history):
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()
        
class Environment:
    def __init__(self):
        self.board = np.zeros((LENGTH,LENGTH))
        self.x =  -1  # why ? player 1
        self.o = 1 # player 2
        self.winner = False
        self.ended = False
        self.num_states = 3**(LENGTH*LENGTH) # Around 19 K
        
    def is_empty(self,i,j):
        return self.board[i,j] == 0 
        
        
    def reward(self, sym):
        # no rewaerd until the game is over. Delayed rewarding
        if not self.game_over():
            return 0
        
        # game is over case: TO_STUDY
        return 1 if self.winner == sym else 0
        
    def get_state(self):
        
        # TO DO - Add comments:
        h = 0
        k = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i,j] == 0:
                    v = 0
                elif self.board[i,j] == self.x:
                    v = 1
                elif self.board[i,j] == self.o:
                    v = 2
                h += (3**k)*v
                k +=1
        return h
        
    def game_over(self, force_recalculate=False):
        
        if not force_recalculate and self.ended :
            return self.ended
        
        # Check for rows
        # check for columns
        # check for diagonals
        # check for draw
        
        #check for rows:
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if (self.board[i].sum() == player*LENGTH):
                    self.winner = player
                    self.ended = True
                    return True
        
        # check for columns
        for j in range(LENGTH):
            for player in (self.x, self.o):
                if (self.board[:,j].sum() == player*LENGTH):
                    self.winner = player
                    self.ended = True
                    return True

        # check for diagonals
        for player in (self.x, self.o):
            # top right to left bottom
            if (self.board.trace() == player*LENGTH):
                self.winner = player
                self.ended = True
                return True
           # top right to left bottom. flip left to right (fliplr)
            if (np.fliplr(self.board).trace() == player*LENGTH):
                self.winner = player
                self.ended = True
                return True

        # check for draw
        if np.all( (self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True
        
        # if game is not over:
        self.winner = None
        return False
    
    
    def is_draw(self):
        return self.ended and self.winner is None
    
    
    def draw_board(self):
        for i in range(LENGTH):
            print("____________________")
            for j in range(LENGTH):
                print("   ", end="")
                if self.board[i,j] == self.x :
                    print("X  |", end="")
                elif self.board[i,j] == self.o :
                    print("O  |", end="")
                else: 
                    print("  |", end="")
            print("")
        print("____________________")
        
        
      
class Human:
    def __init__(self):
        pass
    
    def set_symbol(self,sym):
        self.sym = sym
        
    def take_action(self,env):
        while True:
            move = input("Enter coordinates i,j for your next move (i,j=0,1,2  :")
            i, j  = move.split(",")
            i = int(i)
            j = int(j)
            if env.is_empty(i,j):
                env.board[i,j] = self.sym
                break
            
    def update(self, env):
        pass
    def update_state_history(self,s):
        pass
    
    
def get_state_hash_and_winner(env, i=0, j=0):
    results = []
    
    for v in (0,env.x, env.o):
        env.board[i,j] = v
        if j == 2 :
            # j oges back to 0,  increse i , unless i = 2, then we are done
            if i == 2 :
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state,winner,ended))
            else:
                results += get_state_hash_and_winner(env,i+1,0)
        else:
            results += get_state_hash_and_winner(env, i, j+1)
        
    return results
        

def initialV_x(env, state_winner_triples):
    
    V = np.zeros(env.num_states) # 19 K len.
    for state, winner, ended in state_winner_triples:
        if ended :
            if winner == env.x:
                v = 1 # 
            else:
                v = 0
        else:
            v = 0.5
            
        V[state] = v
    return V




def initialV_o(env, state_winner_triples):
    
    V = np.zeros(env.num_states) # 19 K len.
    for state, winner, ended in state_winner_triples:
        if ended :
            if winner == env.o:
                v = 1 # 
            else:
                v = 0
        else:
            v = 0.5
            
        V[state] = v
    return V

def play_game(p1, p2, env, draw=False):
    current_player = None
    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
    
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        current_player.take_action(env)
        
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
 
    if draw:
        env.draw_board()
        
    p1.update(env)
    p2.update(env)
    
    
    
    
if __name__ == "__main__" :
    p1 = Agent()
    p2 = Agent()
    
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)
    
    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)
    
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)
    
    T = 1000
    
    
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1,p2, Environment())
        
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
#        play_game(human,p1,Environment(),draw=2)
        play_game(p1,human,Environment(),draw=2)
        
        answer = input("Play again ? [Y/n]")
        if answer and answer.lower()[0] == 'n':
            break
    
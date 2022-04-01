# just in case I screwed up
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np


@register_agent("student_agent")
class ChangAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(ChangAgent, self).__init__()
        self.name = "ChangAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        
    # part of minimax algorithm
    def minimax_decision():
        return
    
    # part of minimax algorithm
    def minimax_value():
        return
    
    # check if the game ends
    # copied from world -> check_endgame
    def check_endgame(self, board, my_pos, adv_pos):
        father = dict()
        board_size = board.shape[0]
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)
        
        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]
        
        def union(pos1, pos2):
            father[pos1] = pos2
            
        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(self.moves[1:3]):
                    if board[r, c, dir+1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find(r+move[0], c+move[1])
                    if pos_a != pos_b:
                        union(pos_a, pos_b)
                        
        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        else:
            return True, p0_score, p1_score

    
    def set_barrier(self, board, r, c, dir):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board
    
    
    # find all possible steps given current board
    def all_steps(self, board, my_pos, adv_pos, max_step):
        x1, y1 = my_pos
        x2, y2 = adv_pos
        b = board.copy()
        board_size = board.shape[0]
        lx, ly, lb = [], [], []
        ub = np.maximum(x1 - max_step, 0)
        rb = np.minimum(y1 + max_step, board_size-1)
        db = np.minimum(x1 + max_step, board_size-1)
        lb = np.maximum(y1 - max_step, 0)
        if x1!=x2 and y1!=y2: # case adv is not on our path
            # iterate through current position to each boundary
            # if there is no barrier add to the list

            for i in range(x1-1, ub-1, -1): # direction: up
                if not(board[i, y1, 2]):
                    for j in range(4):
                        if not(b[i, y1, j]):
                            b = self.set_barrier(b, i, y1, j)
                            lx.append(i)
                            ly.append(y1)
                            lb.append(j)
                else:
                    break
            
            for i in range(y1+1, rb+1, 1): # direction； right
                if not(board[x1, i, 3]):
                    for j in range(4):
                        if not(b[x1, i, j]):
                            b = self.set_barrier(b, x1, i ,j)
                            lx.append(x1)
                            ly.append(i)
                            lb.append(j)
                else:
                    break
                            
            for i in range(x1+1, db+1, 1): # direction: down
                if not(board[i, y1, 0]):
                    for j in range(4):
                        if not(b[i, y1, j]):
                            b = self.set_barrier(b, i, y1, j)
                            lx.append(i)
                            ly.append(y1)
                            lb.append(j)
                else:
                    break
            
            for i in range(y1-1, lb-1, -1): # direction left
                if not(board[x1, i, 1]):
                    for j in range(4):
                        if not(b[x1, i, j]):
                            b = self.set_barrier(b, x1, i, j)
                            lx.append(x1)
                            ly.append(i)
                            lb.append(j)
                else:
                    break
                
        elif x1==x2: # case adv is on the same row
            if y1>y2: # adv is on the left side
                for i in range(x1-1, ub-1, -1): # direction: up
                    if not(board[i, y1, 2]):
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1+1, rb+1, 1): # direction； right
                    if not(board[x1, i, 3]):
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i ,j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                            
                for i in range(x1+1, db+1, 1): # direction: down
                    if not(board[i, y1, 0]):
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1-1, lb-1, -1): # direction left
                    if not(board[x1, i, 1]) and y2!=i:
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i, j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                
            else: # adv is on the right
                for i in range(x1-1, ub-1, -1): # direction: up
                    if not(board[i, y1, 2]):
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1+1, rb+1, 1): # direction； right
                    if not(board[x1, i, 3]) and y2!=i:
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i ,j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                            
                for i in range(x1+1, db+1, 1): # direction: down
                    if not(board[i, y1, 0]):
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1-1, lb-1, -1): # direction left
                    if not(board[x1, i, 1]):
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i, j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                    
        else: # case adv is on the same column
            if x1>x2: # adv is on top
                for i in range(x1-1, ub-1, -1): # direction: up
                    if not(board[i, y1, 2]) and x2!=i:
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1+1, rb+1, 1): # direction； right
                    if not(board[x1, i, 3]):
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i ,j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                            
                for i in range(x1+1, db+1, 1): # direction: down
                    if not(board[i, y1, 0]):
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1-1, lb-1, -1): # direction left
                    if not(board[x1, i, 1]):
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i, j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                
            else: # adv is under us
                for i in range(x1-1, ub-1, -1): # direction: up
                    if not(board[i, y1, 2]):
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1+1, rb+1, 1): # direction； right
                    if not(board[x1, i, 3]):
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i ,j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                            
                for i in range(x1+1, db+1, 1): # direction: down
                    if not(board[i, y1, 0]) and x2!=i:
                        for j in range(4):
                            if not(b[i, y1, j]):
                                b = self.set_barrier(b, i, y1, j)
                                lx.append(i)
                                ly.append(y1)
                                lb.append(j)
                    else:
                        break
            
                for i in range(y1-1, lb-1, -1): # direction left
                    if not(board[x1, i, 1]):
                        for j in range(4):
                            if not(b[x1, i, j]):
                                b = self.set_barrier(b, x1, i, j)
                                lx.append(x1)
                                ly.append(i)
                                lb.append(j)
                    else:
                        break
                
        result = np.array([lx, ly, lb])
        return result
    
    # find a list of successor board given current board
    def successors():
        return

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return

        return my_pos, self.dir_map["u"]
    
    
    
    

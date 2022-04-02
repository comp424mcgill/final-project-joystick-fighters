# just in case I screwed up
from os import listdir
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np


@register_agent("chang_agent")
class ChangAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(ChangAgent, self).__init__()
        self.name = "ChangAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    
    
    # part of minimax algorithm
    def minimax_value(self, board, my_pos, adv_pos, max_turn, depth):
        result, util = self.check_endgame(board, my_pos, adv_pos)
        if result:
            return util
        suc_values =np.array([], dtype='int16')
        if max_turn:
            list_step = self.all_steps(board, my_pos, adv_pos)
            list_new_board, list_new_my_pos, _ = self.successors(board, list_step)
            for i in range(len(list_new_board)):
                new_value = self.minimax_value(list_new_board[i], list_new_my_pos[i], adv_pos, False, depth+1)
                suc_values = np.append(suc_values, new_value) 
            return np.max(suc_values)
        else:
            list_step = self.all_steps(board, adv_pos, my_pos)
            list_new_board, list_new_my_pos, _ = self.successors(board, list_step)
            for i in range(len(list_new_board)):
                new_value = -self.minimax_value(list_new_board[i], list_new_my_pos[i], my_pos, True, depth+1)
                suc_values = np.append(suc_values, new_value) 
            return np.min(suc_values)

    
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
                    pos_b = find((r+move[0], c+move[1]))
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
            return False, 0
        elif p0_score != p1_score: # player 0 wins
            return True, (p0_score - p1_score)
        else: # tie
            return True, 0

    
    def set_barrier(self, board, r, c, dir):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board
    
    
    def check_valid_step(self, board, start_pos, end_pos, barrier_dir, adv_pos):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        # adv_pos = self.p0_pos if self.turn else self.p1_pos
        
        max_step = (board.shape[0] + 1) // 2

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
    
    
    # find all possible steps given current board
    def all_steps(self, board, my_pos, adv_pos):
        list_step = []
        board_size = board.shape[0]
        for i in range(board_size):
            for j in range(board_size):
                for k in range(4):
                    if self.check_valid_step(board, np.array(my_pos), np.array([i,j]), k, adv_pos):
                        list_step.append(((i,j),k))
        return list_step
    
    
    # find a list of successor board given current board
    def successors(self, board, list_step):
        list_new_board, list_new_pos, list_new_dir = [], [], []
        for i in range(len(list_step)):
            temp = board.copy()
            (x, y), dir = list_step[i]
            temp = self.set_barrier(temp, x, y, dir)
            list_new_board.append(temp)
            list_new_pos.append((x,y))
            list_new_dir.append(dir)
        return list_new_board, list_new_pos, list_new_dir


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
        
        list_op = self.all_steps(chess_board, np.array(my_pos), adv_pos)
        list_new_board, list_new_my_pos, list_new_dir = self.successors(chess_board, list_op)
        list_val = np.array([], dtype='int16')
        for i in range(len(list_new_board)):
            temp_new_board = list_new_board[i]
            temp_new_my_pos = list_new_my_pos[i]
            list_val = np.append(list_val, self.minimax_value(temp_new_board, temp_new_my_pos, adv_pos, True, 0))
        
        best_idx = np.argmax(list_val)
        best_my_pos = list_new_my_pos[best_idx]
        best_dir = list_new_dir[best_idx]

        return best_my_pos, best_dir
    
    
    
    

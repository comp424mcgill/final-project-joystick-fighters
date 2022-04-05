# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.max_my_reach = 2
        self.max_adv_reach = 2
        self.my_barrier_score = np.array([-5,-4,-3,-2,-1])
        self.adv_barrier_score = np.array([5,4,3,2,1])

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
        
        return self.greedy_decision(chess_board, my_pos, adv_pos)
    
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
        if p0_r == p1_r: # not end
            return False, -1
        elif p0_score > p1_score: # player 0 wins
            return True, 1
        elif p0_score < p1_score: # player 1 wins
            return True, 0
        else: # tie
            return True, 0.5

    def set_barrier(self, board, r, c, dir):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board
    
    # given board, start_pos, end_pos, barrier_dir, adv_pos, check if the move is valid
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
    
    # given board, my_pos, adv_pos: return list[((x,y), dir)]
    def all_steps(self, board, my_pos, adv_pos):
        list_step = []
        board_size = board.shape[0]
        for i in range(board_size):
            for j in range(board_size):
                for k in range(4):
                    if self.check_valid_step(board, np.array(my_pos), np.array([i,j]), k, adv_pos):
                        list_step.append(((i,j),k))
        return list_step
    
    # given board, my_pos, adv_pos, my_turn, return all possible next state as:
    # list[new_board], list[new_pos], list[new_dir]
    def all_next_state(self, board, my_pos, adv_pos, my_turn):
        list_new_board, list_new_pos, list_new_dir = [], [], []
        if my_turn:
            list_step = self.all_steps(board, my_pos, adv_pos)
            for i in range(len(list_step)):
                temp = deepcopy(board)
                (x, y), dir = list_step[i]
                temp = self.set_barrier(temp, x, y, dir)
                list_new_board.append(temp)
                list_new_pos.append((x,y))
                list_new_dir.append(dir)
        else:
            list_step = self.all_steps(board, adv_pos, my_pos)
            for i in range(len(list_step)):
                temp = deepcopy(board)
                (x, y), dir = list_step[i]
                temp = self.set_barrier(temp, x, y, dir)
                list_new_board.append(temp)
                list_new_pos.append((x,y))
                list_new_dir.append(dir)
        return list_new_board, list_new_pos, list_new_dir
    
    # given current state, evaluate each position and return the best state with highest heuristic score
    def greedy_decision(self, board, my_pos, adv_pos):
        list_new_board, list_new_pos, list_new_dir = self.all_next_state(board, my_pos, adv_pos, True)
        list_rm = []
        for i in range(len(list_new_board)):
            end_result, end_score = self.check_endgame(list_new_board[i], list_new_pos[i], adv_pos)
            if (end_result, end_score) == (True, 1): # if the position is winning return this
                return list_new_pos[i], list_new_dir[i]
            elif (end_result, end_score) == (True, 0):
                list_rm.append(i)
        list_rm.reverse()
        for i in range(len(list_rm)): # removing losing position
            list_new_board.pop(list_rm[i])
            list_new_pos.pop(list_rm[i])
            list_new_dir.pop(list_rm[i])
        # we prefer:
        # position that is far away from wall and adv near wall
        # my_reach: get the number of barrier within the reach as a score, lower the better
        # adv_reach: get the number of barrier within the adversary's reach as a score, higher the better
        state_score = np.array([])
        board_size = list_new_board[i].shape[0]
        for i in range(len(list_new_board)):
            score_i = 0
            r, c = list_new_pos[i]
            for reach_row in range(self.max_my_reach*2):
                if (r-self.max_my_reach-reach_row)>=0 and (r-self.max_my_reach+reach_row)<board_size:
                    for reach_col in range(self.max_my_reach*2):
                        if (c-self.max_my_reach-reach_col)>=0 and (c-self.max_my_reach+reach_col)<board_size:
                            score_i += self.my_barrier_score[np.abs(reach_row-self.max_my_reach)+np.abs(reach_col-self.max_my_reach)]*np.count_nonzero((list_new_board[i])[r-self.max_my_reach, c-self.max_my_reach])
            r, c = adv_pos
            for reach_row in range(self.max_adv_reach*2):
                if (r-self.max_adv_reach-reach_row)>=0 and (r-self.max_adv_reach+reach_row)<board_size:
                    for reach_col in range(self.max_adv_reach*2):
                        if (c-self.max_adv_reach-reach_col)>=0 and (c-self.max_adv_reach+reach_col)<board_size:
                            score_i += self.adv_barrier_score[np.abs(reach_row-self.max_adv_reach)+np.abs(reach_col-self.max_adv_reach)]*np.count_nonzero((list_new_board[i])[r-self.max_adv_reach, c-self.max_adv_reach])
            state_score = np.append(state_score, score_i)
        best_idx = np.argmax(state_score)
        best_pos = list_new_pos[best_idx]
        best_dir = list_new_dir[best_idx]
        return best_pos, best_dir
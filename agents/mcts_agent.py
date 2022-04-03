# Student agent: Add your own agent here
from hashlib import new
from logging import root
from agents.agent import Agent
from constants import MAX_BOARD_SIZE
from store import register_agent
import sys
from copy import deepcopy
import numpy as np


@register_agent("mcts_agent")
class MCTSAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"
        self.autoplay = True
        
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.max_exp = 10 # max number of exploration
        self.max_depth = 1 # max number of depth of the tree
        

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
        tree_root = self.MCTSNode(chess_board, my_pos, adv_pos, True, None, None)
        for i in range(self.max_exp):
            node, _ = self.select_node(tree_root, tree_root.uct_val(), 0)
            node.explored = True
            self.add_children(node, self.find_all_children(chess_board, my_pos, adv_pos, node.my_turn))
            rst = self.random_walk(chess_board, my_pos, adv_pos)
            while(node!=None):
                node.number_of_visits += 1
                node.value += rst
                node = node.parent
        
        new_pos, new_dir, rtn_val = None, None, -1
        for i in range(len(tree_root.children)):
            child = tree_root.children[i]
            child_val = child.value/(child.number_of_visits+0.001)
            if child_val > rtn_val:
                new_pos, new_dir, rtn_val = child.my_pos, child.new_dir, child_val
        return new_pos, new_dir
    
    
    class MCTSNode():
        def __init__(self, board, my_pos, adv_pos, my_turn, parent=None, new_dir=None):
            self.board = board
            self.my_pos = my_pos
            self.adv_pos = adv_pos
            self.my_turn = my_turn # True or False
            self.parent = parent
            self.new_dir = new_dir # new barrier direction chosen by parent
            self.children = []
            self.explored = False
            self.number_of_visits = 0 # n(s), Q(s,a) denominator
            self.value = 0 # Q(s,a) numerator
            return
        
        def uct_val(self):
            if self.parent==None:
                return self.value/(self.number_of_visits+0.001)
            return (self.value/(self.number_of_visits+0.001) + np.sqrt(2*self.parent.number_of_visits/(self.number_of_visits+0.001)))
        
        
    def select_node(self, node, node_val, depth):
        rtn_node, rtn_val = node, node_val
        if node.explored==True and depth<self.max_depth:
            for i in range(len(node.children)):
                child = node.children[i]
                child_val = child.uct_val()
                if child_val > rtn_val:
                    rtn_node, rtn_val = self.select_node(child, child_val, depth+1)
        return rtn_node, rtn_val

    
    def add_children(self, node, list_children):
        cur_my_pos = node.my_pos
        cur_adv_pos = node.adv_pos
        cur_my_turn = node.my_turn
        list_new_board, list_new_pos, list_new_dir = list_children
        for i in range(len(list_new_board)):
            new_board, new_pos, new_dir = list_new_board[i], list_new_pos[i], list_new_dir[i]
            if cur_my_turn:
                child = self.MCTSNode(new_board, new_pos, cur_adv_pos, False, node, new_dir)
            else:
                child = self.MCTSNode(new_board, cur_my_pos, new_pos, True, node, new_dir)
            node.children.append(child)
            
    
    def random_walk(self, board, my_pos, adv_pos):
        temp=deepcopy(board)
        result, util = self.check_endgame(temp, my_pos, adv_pos)
        while (result != True):
            myposstep=self.all_steps(temp,my_pos,adv_pos)
            choice1 = np.random.randint(0, (len(myposstep)))
            (x, y), dir = myposstep[choice1]
            temp=self.set_barrier(temp, x, y, dir)
            my_pos= (x, y)
            result, util = self.check_endgame(temp, my_pos, adv_pos)
            if (result == True):
                return util
            adsteps=self.all_steps(temp,adv_pos,my_pos)
            choice2 = np.random.randint(0, (len(adsteps)))
            (x, y), dir = adsteps[choice2]
            temp=self.set_barrier(temp, x, y, dir)
            adv_pos = (x, y)
            result, util = self.check_endgame(temp, my_pos, adv_pos)
        return util
    
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
    
    
    def all_steps(self, board, my_pos, adv_pos):
        list_step = []
        board_size = board.shape[0]
        for i in range(board_size):
            for j in range(board_size):
                for k in range(4):
                    if self.check_valid_step(board, np.array(my_pos), np.array([i,j]), k, adv_pos):
                        list_step.append(((i,j),k))
        return list_step
    
    # Given current state, return the next state
    def next_state(self, board, my_pos, adv_pos, step, my_turn):
        (r, c), dir = step
        new_board = self.set_barrier(board, r, c, dir)
        if my_turn:
            return new_board, (r, c), adv_pos, False
        else:
            return new_board, my_pos, (r, c), True
    
    
    def successors(self, board, list_step):
        list_new_board, list_new_pos, list_new_dir = [], [], []
        for i in range(len(list_step)):
            temp = deepcopy(board)
            (x, y), dir = list_step[i]
            temp = self.set_barrier(temp, x, y, dir)
            list_new_board.append(temp)
            list_new_pos.append((x,y))
            list_new_dir.append(dir)
        return list_new_board, list_new_pos, list_new_dir
    
    
    def find_all_children(self, board, my_pos, adv_pos, my_turn):
        if my_turn:
            list_children = self.all_steps(board, my_pos, adv_pos)
        else:
            list_children = self.all_steps(board, adv_pos, my_pos)
        return self.successors(board, list_children)

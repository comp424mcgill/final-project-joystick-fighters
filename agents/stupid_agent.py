# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy


@register_agent("stupid_agent")
class StupidAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StupidAgent, self).__init__()
        self.name = "StupidAgent"
        self.autoplay = True
        
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.tree_root = None
        self.max_first_exp = 100
        self.max_exp = 10

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
        if self.tree_root==None:
            self.tree_root = self.MCTSNode(chess_board, my_pos, adv_pos, True, None, None)
            for i in range(self.max_first_exp): # for our first run, try to exploit the 30 seconds
                node = self.uct_node(self.tree_root)
                if node.my_turn:
                    sim_rst = self.random_walk(chess_board, my_pos, adv_pos)
                else:
                    sim_rst = self.random_walk(chess_board, adv_pos, my_pos)
                while(node!=None):
                    node.n += 1
                    node.v += sim_rst
                    node = node.parent
        else:
            for i in range(len(self.tree_root.children)):
                if np.array_equal(chess_board, self.tree_root.children[i].board):
                    if self.tree_root.children[i].adv_pos==adv_pos:
                        self.tree_root = self.tree_root.children[i]
                        self.tree_root.parent = None
                        break
            for i in range(self.max_exp):
                node = self.uct_node(self.tree_root)
                if node.my_turn:
                    sim_rst = self.random_walk(chess_board, my_pos, adv_pos)
                else:
                    sim_rst = self.random_walk(chess_board, adv_pos, my_pos)
                while(node!=None):
                    node.n += 1
                    node.v += sim_rst
                    node = node.parent
        
        new_root, new_pos, new_dir, rtn_val = self.tree_root, None, None, -1
        for i in range(len(self.tree_root.children)):
            child = self.tree_root.children[i]
            child_val = child.v/(child.n+0.001)
            if child_val > rtn_val:
                new_root, new_pos, new_dir, rtn_val = child, child.my_pos, child.new_dir, child_val
        
        self.tree_root = new_root
        self.tree_root.parent = None
        return new_pos, new_dir
    
    
    class MCTSNode():
        def __init__(self, board, my_pos, adv_pos, my_turn, parent=None, new_dir=None):
            self.board = board
            self.my_pos = my_pos
            self.adv_pos = adv_pos
            self.my_turn = my_turn
            self.parent = parent
            self.new_dir = new_dir
            self.children = []
            self.n = 0
            self.v = 0
            return
        
        
    # from the root, find the node to do simulation
    # use this function uct_node(self.tree_root)
    def uct_node(self, node):
        if node.n==0: # it is a node that has not been visited
            return node
        elif len(node.children)==0: # visited but not expanded, add children and choose from children
            list_children = self.find_all_children(node.board, node.my_pos, node.adv_pos, node.my_turn)
            node.children = self.children_nodes(list_children)
            max_val = 0
            max_node = None
            for i in range(len(node.children)):
                node.children[i].parent = node
                cur_val = (node.children[i].v/(node.children[i].n+0.001) + np.sqrt(2*node.n/(node.children[i].n+0.001)))
                if cur_val>max_val:
                    max_val = cur_val
                    max_node = node.children[i]
            return max_node
        else: # visited and expanded, need to choose from its children, then continue
            max_val = 0
            max_node = None
            for i in range(len(node.children)):
                cur_val = (node.children[i].v/(node.children[i].n+0.001) + np.sqrt(2*node.n/(node.children[i].n+0.001)))
                if cur_val>max_val:
                    max_val = cur_val
                    max_node = node.children[i]
            return self.uct_node(max_node)
            
    
    # return a list of MCTS nodes, from a list of children
    def children_nodes(self, list_children):
        (list_new_board, list_new_pos, list_new_dir), fixed_pos, turn = list_children
        rtn_list = []
        for i in range(len(list_new_board)):
            new_board, new_pos, new_dir = list_new_board[i], list_new_pos[i], list_new_dir[i]
            if turn:
                child = self.MCTSNode(new_board, new_pos, fixed_pos, False, None, new_dir)
            else:
                child = self.MCTSNode(new_board, fixed_pos, new_pos, True, None, new_dir)
            rtn_list.append(child)
        return rtn_list

        
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
            return self.successors(board, list_children), adv_pos, my_turn
        else:
            list_children = self.all_steps(board, adv_pos, my_pos)
            return self.successors(board, list_children), my_pos, my_turn
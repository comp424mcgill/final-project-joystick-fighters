# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import numpy as np
import time


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
        self.root_node = None

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
        start_time = time.time()
        self.board_size = chess_board.shape[0]
        list_new_board, list_new_pos, list_new_dir = self.all_next_state(chess_board, my_pos, adv_pos, True)
        list_rm = []
        for i in range(len(list_new_board)):
            end_result, end_score = self.check_endgame(list_new_board[i], list_new_pos[i], adv_pos)
            if (end_result, end_score) == (True, 1): # if the position is winning return this
                return list_new_pos[i], list_new_dir[i]
            elif (end_result, end_score) == (True, 0):
                list_rm.append(i)
            elif (end_result, end_score) == (True, 0.5):
                continue
            '''
            else:
                list_adv_new_board, list_adv_new_pos, list_adv_new_dir = self.all_next_state(list_new_board[i], list_new_pos[i], adv_pos, False)
                for j in range(len(list_adv_new_board)):
                    adv_result, adv_score = self.check_endgame(list_adv_new_board[j], list_adv_new_pos[j], list_new_pos[i])
                    if (adv_result, adv_score) == (True, 1):
                        list_rm.append(i)
                        break
            '''
        list_rm.reverse()
        for k in range(len(list_rm)): # removing losing position
            list_new_board.pop(list_rm[k])
            list_new_pos.pop(list_rm[k])
            list_new_dir.pop(list_rm[k])
            
        self.center = (self.board_size-1)/2
        
        zip_list = list(zip(list_new_board, list_new_pos, list_new_dir))
        np.random.shuffle(zip_list)
        shuffle_new_board, shuffle_new_pos, shuffle_new_dir = zip(*zip_list)
        shuffle_new_board = np.array(shuffle_new_board)
        shuffle_new_pos = list(shuffle_new_pos)
        shuffle_new_dir = list(shuffle_new_dir)
        rand_pos, rand_dir = shuffle_new_pos[0], shuffle_new_dir[0]
        
        score = np.array([])
        for i in range(len(list_new_board)):
            x1, y1 = list_new_pos[i]
            x2, y2 = adv_pos
            score = np.append(score, np.abs(x1-x2)+np.abs(y1-y2)+np.abs(x1-self.center)+np.abs(x2-self.center))
            if x1 > self.center:
                if y1 > self.center:
                    if list_new_dir[i]==0 or list_new_dir[i]==3:
                        score[i] -= 1
                elif y1 == self.center:
                    if list_new_dir[i]==0:
                        score[i] -= 1
                else:
                    if list_new_dir[i]==0 or list_new_dir[i]==1:
                        score[i] -= 1
            elif x1 == self.center:
                if y1 > self.center:
                    if list_new_dir[i]==3:
                        score[i] -= 1
                else:
                    if list_new_dir[i]==1:
                        score[i] -= 1
            else:
                if y1 > self.center:
                    if list_new_dir[i]==2 or list_new_dir[i]==3:
                        score[i] -= 1
                elif y1 == self.center:
                    if list_new_dir[i]==2:
                        score[i] -= 1
                else:
                    if list_new_dir[i]==1 or list_new_dir[i]==2:
                        score[i] -= 1
        list_center_idx = np.argsort(score)
        center_board, center_pos, center_dir = list_new_board[list_center_idx[0]], list_new_pos[list_center_idx[0]], list_new_dir[list_center_idx[0]]
        if self.root_node == None:
            self.root_node = self.MCTSNode(center_board, center_pos, adv_pos, False, None, None)
            self.root_node.n += 1
            print("Time1:", time.time()-start_time)
            self.root_node.v += self.rand_simulation(center_board, center_pos, adv_pos, False, max_step)
            print("Time2:", time.time()-start_time)
            list_new_adv_board, list_new_adv_pos, list_new_adv_dir = self.all_next_state(chess_board, my_pos, adv_pos, False)
            for i in range(len(list_new_adv_board)):
                self.root_node.children.append(self.MCTSNode(list_new_adv_board[i], center_pos, list_new_adv_pos[i], True, self.root_node, list_new_adv_dir[i]))
            it = 0
            while (time.time()-start_time) < 29.0:
                it += 1
                print("Iteration:", it, "at time:", time.time()-start_time)
                node = self.uct(self.root_node)
                sim_rst = self.rand_simulation(node.board, node.my_pos, node.adv_pos, node.my_turn, max_step)
                while(node!=None):
                    node.n += 1
                    node.v += sim_rst
                    node = node.parent
                max_node_val = self.root_node.children[0].v
                max_node = self.root_node.children[0]
                for j in range(len(self.root_node.children)):
                    if (self.root_node.children[j].v/(self.root_node.children[j].n+0.001)) > max_node_val:
                        max_node_val = (self.root_node.children[j].v/(self.root_node.children[j].n+0.001))
                        max_node = self.root_node.children[j]
            print("total iteration in 30s:", it)
            max_node.parent = None
            self.root_node = max_node
            return center_pos, center_dir
        else:
            while (time.time()-start_time) <1.8:
                
                pass
        
        end_time = time.time()
        print(end_time-start_time)
        
        return center_pos, center_dir
    
    
    class MCTSNode():
        def __init__(self, board, my_pos, adv_pos, my_turn, parent, new_dir):
            self.board = board
            self.my_pos = my_pos
            self.adv_pos = adv_pos
            self.my_turn = my_turn
            self.parent = parent
            self.children = []
            self.new_dir = new_dir
            self.n = 0
            self.v = 0
            
        # for a node, return the uct node to do random simulation
    def uct(self, node):
        if node.n==0:
            return node
        uct_val = np.array([])
        for i in range(len(node.children)):
            temp = node.children[i].v/(node.children[i].n+0.001) + np.sqrt(2*node.n/(node.children[i].n+0.001))
            uct_val = np.append(uct_val, temp)
        max_idx = np.argmax(uct_val)
        if node.children[max_idx].n==0:
            list_new_board, list_new_pos, list_new_dir = self.all_next_state(node.children[max_idx].board, node.children[max_idx].my_pos, node.children[max_idx].adv_pos, (node.children[max_idx].my_turn))
            for i in range(len(list_new_board)):
                if node.children[max_idx].my_turn:
                    node.children.append(self.MCTSNode(list_new_board[i], list_new_pos[i], node.children[max_idx].adv_pos, False, node.children[max_idx], list_new_dir[i]))
                else:
                    node.children.append(self.MCTSNode(list_new_board[i], node.children[max_idx].my_pos, list_new_pos[i], True, node.children[max_idx], list_new_dir[i]))
            return node.children[max_idx]
        else:
            return self.uct(node.children[max_idx])
        
        
    def rand_simulation(self, board, my_pos, adv_pos, my_turn, max_step):
        temp = deepcopy(board)
        if my_turn:
            
            pass
        
        if my_turn:
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
        else:
            result, util = self.check_endgame(temp, my_pos, adv_pos)
            while (result != True):
                myposstep=self.all_steps(temp,adv_pos,my_pos)
                choice1 = np.random.randint(0, (len(myposstep)))
                (x, y), dir = myposstep[choice1]
                temp=self.set_barrier(temp, x, y, dir)
                adv_pos= (x, y)
                result, util = self.check_endgame(temp, my_pos, adv_pos)
                if (result == True):
                    return util
                adsteps=self.all_steps(temp,my_pos, adv_pos)
                choice2 = np.random.randint(0, (len(adsteps)))
                (x, y), dir = adsteps[choice2]
                temp=self.set_barrier(temp, x, y, dir)
                my_pos = (x, y)
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
    
    # given current state, return the winning state if any, else delete all losing state and shuffle the list
    def preprocess_next_state(self, board, my_pos, adv_pos):
        list_new_board, list_new_pos, list_new_dir = self.all_next_state(board, my_pos, adv_pos, True)
        list_rm = []
        for i in range(len(list_new_board)):
            end_result, end_score = self.check_endgame(list_new_board[i], list_new_pos[i], adv_pos)
            if (end_result, end_score) == (True, 1): # if the position is winning return this
                return list_new_pos[i], list_new_dir[i]
            elif (end_result, end_score) == (True, 0):
                list_rm.append(i)
            elif (end_result, end_score) == (True, 0.5):
                continue
            '''
            else:
                list_adv_new_board, list_adv_new_pos, list_adv_new_dir = self.all_next_state(list_new_board[i], list_new_pos[i], adv_pos, False)
                for j in range(len(list_adv_new_board)):
                    adv_result, adv_score = self.check_endgame(list_adv_new_board[j], list_adv_new_pos[j], list_new_pos[i])
                    if (adv_result, adv_score) == (True, 1):
                        list_rm.append(i)
                        break
            '''
        list_rm.reverse()
        for k in range(len(list_rm)): # removing losing position
            list_new_board.pop(list_rm[k])
            list_new_pos.pop(list_rm[k])
            list_new_dir.pop(list_rm[k])
        zip_list = list(zip(list_new_board, list_new_pos, list_new_dir))
        np.random.shuffle(zip_list)
        shuffle_new_board, shuffle_new_pos, shuffle_new_dir = zip(*zip_list)
        shuffle_new_board = np.array(shuffle_new_board)
        shuffle_new_pos = list(shuffle_new_pos)
        shuffle_new_dir = list(shuffle_new_dir)
        return shuffle_new_board, shuffle_new_pos, shuffle_new_dir
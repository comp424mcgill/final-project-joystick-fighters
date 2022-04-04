# Student agent: Add your own agent here
import random

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
        list_step1=self.all_steps_possible(chess_board,my_pos, adv_pos)
        temp=self.choice(chess_board,list_step1, adv_pos)
        (x, y), dir=list_step1[temp]
        return (x, y), dir

    def choice(self, board, list_step1, adv_pos):
        list_utility = [0] * len(list_step1)
        list_res=[False]*len(list_step1)
        for i in range(len(list_step1)):  # my steps
            temp = board.copy()
            (x, y), dir = list_step1[i]
            temp = self.set_barrier(temp, x, y, dir)
            mypos1 = (x, y)
            result, util = self.check_endgame(temp, mypos1, adv_pos)
            if util==1:
                return i
            list_utility[i] = util * 100
            list_res[i]=result

        for i in range(len(list_res)):
            if  list_res[i]==False:
                (x, y), dir = list_step1[i]
                temp = board.copy()
                temp = self.set_barrier(temp, x, y, dir)
                mypos1 = (x, y)
                advsteps = self.all_steps_possible(temp, adv_pos, mypos1)  # adversary steps
                if len(advsteps) > 0:
                    list_utility1 = [0] * len(advsteps)
                    list_res2 = [False] * len(advsteps)
                    for j in range(len(advsteps)):
                        temp1 = temp.copy()
                        (x1, y1), dir1 = advsteps[j]
                        advpos1 = (x1, y1)
                        temp1 = self.set_barrier(temp1, x1, y1, dir1)
                        result1, util1 = self.check_endgame(temp1, mypos1, advpos1)
                        if util1==-1:
                            list_utility[i]=util1*100
                            break
                        list_utility1[j] = util1 * 100
                        list_res2[j] = result1
                        list_utility[i]=self.findminind(list_utility1)
        mustfail=True
        temp=0
        for i in range(len(list_step1)):
            if list_utility[i]==100:
                return i
            if list_utility[i]>=0:
                mustfail =False
        if  mustfail:
            return random.randint(0,(len(list_step1)-1))
        else:
            found=False
            while not found:
                temp=random.randint(0,(len(list_step1)-1))
                if list_utility[temp]==0:
                    found=True
        return temp

    def findminind(selfself, listint):
        min=listint[0]
        for i in range(len(listint)):
            if listint[i]<min:
                min=listint[i]
        return min

    def all_steps_possible(self, board, my_pos, adv_pos):
        list_step = []
        board_size = board.shape[0]
        for i in range(board_size):
            for j in range(board_size):
                for k in range(4):
                    if self.check_valid_step(board, np.array(my_pos), np.array([i, j]), k, adv_pos):
                        temp = board.copy()
                        temp = self.set_barrier(temp, i, j, k)
                        result, util = self.check_endgame(temp, (i,j), adv_pos)
                        if util>=0:
                            list_step.append(((i, j), k))
        return list_step

    def set_barrier(self, board, r, c, dir):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board

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
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
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
        if p0_score != p1_score:  # player 0 wins
            if p0_score > p1_score:
                return True, 1
            else:
                return True, -1
        else:  # tie
            return True, 0

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
            if (cur_step) == max_step:
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

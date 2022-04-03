from math import log

import numpy as np
from copy import deepcopy

from numpy import sqrt

from agents.agent import Agent
from store import register_agent

"""
minimax with monte carlo, 3 layers of choices with 10 branches of monte carlo algorithm

"""


# Important: you should register your agent with a name
@register_agent("lin_agent")
class LinAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(LinAgent, self).__init__()
        self.name = "LinAgent"
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
        # Moves (Up, Right, Down, Left)
        possiblesteps=self.all_steps_possible(chess_board,my_pos, adv_pos,0)
        index= self.choice(chess_board, possiblesteps, adv_pos)
        (x, y), dir = possiblesteps[index]
        my_pos=(x, y)
        return my_pos, dir

    # find all possible steps given current board

    def all_steps_possible(self, board, my_pos, adv_pos, depth):
        list_step = []
        board_size = board.shape[0]
        for i in range(board_size):
            for j in range(board_size):
                for k in range(4):
                    if self.check_valid_step(board, np.array(my_pos), np.array([i, j]), k, adv_pos, depth):
                        list_step.append(((i, j), k))
        return list_step

    # find a list of successor board given current board
    def choice(self, board, list_step1, adv_pos):
        list_new_pos, list_new_dir = [], []
        list_utility = [0] * len(list_step1)
        for i in range(len(list_step1)):  # my steps
            temp = board.deepcopy()
            (x, y), dir = list_step1[i]
            temp = self.set_barrier(temp, x, y, dir)
            mypos1 = (x, y)
            result, util = self.check_endgame(temp, mypos1, adv_pos)
            list_utility[i] = util * 100
            if not result:
                advsteps = self.all_steps_possible(temp, adv_pos, mypos1, 1)  # adversary steps
                if len(advsteps) > 0:
                    list_utility1 = [0] * len(advsteps)
                    for j in range(len(advsteps)):
                        temp1 = temp.deepcopy()
                        (x1, y1), dir1 = advsteps[j]
                        advpos1 = (x1, y1)
                        temp1 = self.set_barrier(temp1, x1, y1, dir1)
                        result1, util1 = self.check_endgame(temp1, advpos1, mypos1)
                        list_utility1[j] = -util1 * 100
                        if not result1:
                            mysteps = self.all_steps_possible(temp1, mypos1, advpos1, 1)  # my steps
                            if len(mysteps) > 0:
                                list_utility2 = [0] * len(mysteps)
                                for k in range(len(mysteps)):
                                    temp2 = temp1.deepcopy()
                                    (x2, y2), dir2 = mysteps[k]
                                    temp2 = self.set_barrier(temp2, x2, y2, dir2)
                                    result2, util2 = self.check_endgame(temp, (x2, y2), advpos1)
                                    list_utility2[k] = util2 * 100
                                    if not result2:
                                        temputil = 0
                                        for z in range(100):
                                            temputil += self.random_walk(board, (x2, y2), advpos1)*80
                                        list_utility2[k]=temputil/100+2*sqrt(log(100)/100)
                                list_utility1[j] =self.findmaxind(list_utility2)
                    list_utility[i]=self.findminind(list_utility1)
        return self.findmaxid(list_utility)

    def set_barrier(self, board, r, c, dir):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board

    def random_walk(self, board, my_pos, adv_pos):
        """
        Randomly walk until reach end of game

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        temp = board.deepcopy()
        result, util = self.check_endgame(temp, my_pos, adv_pos)
        depth = 2
        mstp=True
        while not result:
            mstp = False
            advposstep = self.all_steps_possible(temp,adv_pos, my_pos, depth)
            if len(advposstep)>0:
                choice1 = np.random.randint(0, (len(advposstep) - 1))
                (x, y), dir = advposstep[choice1]
                temp = self.set_barrier(temp, x, y, dir)
                my_pos = (x, y)
                result, util = self.check_endgame(temp,  adv_pos, my_pos)
                if result:
                    return -util
                mysteps = self.all_steps_possible(temp, adv_pos, my_pos, depth)
                mstp = True
                if len(mysteps)>0:
                    choice2 = np.random.randint(0, (len(mysteps) - 1))
                    (x1, y1), dir2 = mysteps[choice2]
                    temp = self.set_barrier(temp, x1, y1, dir2)
                    adv_pos = (x, y)
                    result, util = self.check_endgame(temp, my_pos, adv_pos)
                    if result:
                        return util
                else:
                    return 0
            else:
                return 0
            depth += 1
        if mstp:
            return util
        else:
            return -util

    def findmaxind(selfself, listint):
        max=listint[0]
        for i in range(len(listint)):
            if listint[i]>max:
                max=listint[i]
        return max

    def findminind(selfself, listint):
        min=listint[0]
        for i in range(len(listint)):
            if listint[i]<min:
                min=listint[i]
        return min

    def findmaxid(selfself, listint):
        max=listint[0]
        ind=0
        for i in range(len(listint)):
            if listint[i]>max:
                max=listint[i]
                ind=i
        return ind

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
        elif p0_score != p1_score:  # player 0 wins
            if p0_score > p1_score:
                return True, 1
            else:
                return True, -1
        else:  # tie
            return True, 0

    def check_valid_step(self, board, start_pos, end_pos, barrier_dir, adv_pos, depth):
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
            if (cur_step + depth) == max_step:
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

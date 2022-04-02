import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

"""
minimax with monte carlo, 3 layers of choices with 10 branches of monte carlo algorithm

"""


# Important: you should register your agent with a name
@register_agent("Lin_agent")
class LinAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(LinAgent, self).__init__()
        self.name = "LinAgent"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    # find all possible steps given current board

    def all_steps_possible (self, board, my_pos, adv_pos):
        list_step = []
        board_size = board.shape[0]
        for i in range(board_size):
            for j in range(board_size):
                for k in range(4):
                    if self.check_valid_step(board, np.array(my_pos), np.array([i, j]), k, adv_pos):
                        list_step.append(((i, j), k))
        return list_step

    # find a list of successor board given current board
    def scores(self, board, list_step1, adv_pos):
        list_new_pos, list_new_dir = [], []
        list_utility=[]
        for i in range(len(list_step1)):
            temp = board.deepcopy()
            (x, y), dir = list_step1[i]
            temp=self.set_barrier(temp, x, y, dir)
            advsteps=self.all_steps_possible(temp, adv_pos, (x, y))
            for j in range(len(advsteps)):
                temp1=temp.deepcopy()
                (x1, y1), dir1 = advsteps[i]
                temp1 = self.set_barrier(temp1, x1, y1, dir1)
                mystep= self.all_steps_possible(temp, (x, y), (x1, y1))



        return list_utility

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
        temp=board.deepcopy()
        result, util = self.check_endgame(temp, my_pos, adv_pos)
        while (result != True):
            myposstep=self.all_steps_possible(temp,my_pos,adv_pos)
            choice1 = np.random.randint(0, (len(myposstep)-1))
            (x, y), dir = myposstep[choice1]
            temp=self.set_barrier(temp, x, y, dir)
            my_pos= (x, y)
            result, util = self.check_endgame(temp, my_pos, adv_pos)
            if (result == True):
                return util
            adsteps=self.all_steps_possible(temp,adv_pos,my_pos)
            choice2 = np.random.randint(0, (len(adsteps) - 1))
            (x, y), dir = myposstep[choice2]
            temp=self.set_barrier(temp, x, y, dir)
            adv_pos = (x, y)
            result, util = self.check_endgame(temp, my_pos, adv_pos)
        return util

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
            return True, (p0_score - p1_score)
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
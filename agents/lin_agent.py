import random
from copy import deepcopy
from math import log

import numpy as np

from numpy import sqrt

from agents.agent import Agent
from store import register_agent

import time
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
        self.root_node = None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        start_time = time.time_ns()
        (x, y), dir = self.choice(chess_board, adv_pos, my_pos, start_time)

        return (x, y), dir

    # find all possible steps given current board
    def all_steps_possible(self, board, my_pos, adv_pos):
        list_step = []
        board_size = board.shape[0]
        max_step = (board.shape[0] + 1) // 2
        x = my_pos[0]
        y = my_pos[1]
        lx = 0
        ux = board_size
        ly = lx
        uy = ux
        if (x - max_step) > 0:
            lx = x - max_step
        if (y - max_step) > 0:
            ly = y - max_step
        if (x + max_step) < board_size:
            ux = x + max_step + 1
        if (y + max_step) < board_size:
            uy = y + max_step + 1
        for i in range(lx, ux):
            for j in range(ly, uy):
                if (abs(i - x) + abs(j - y)) <= max_step:
                    for k in range(4):
                        if self.check_valid_step(board, np.array(my_pos), np.array([i, j]), k, adv_pos):
                            list_step.append(((i, j), k))

        return list_step

    def choice(self, board,  adv_pos, my_pos, start_time):
        list_new_board, list_new_pos, list_new_dir = self.all_next_state(board, my_pos, adv_pos, True)
        list_utility = [0] * len(list_new_pos)
        list_res = [False] * len(list_new_pos)
        list_res1=[False] * len(list_new_pos)
        ind1=0
        mustfail = True
        if self.root_node==None:
            timeconstraint = 29600000000
        else:
            timeconstraint=1600000000
        self.root_node=1
        for i in range(len(list_new_dir)):  # my steps
            result, util = self.check_endgame(list_new_board[i], list_new_pos[i], adv_pos)
            if util == 1:
                pos = list_new_pos[i]
                dir = list_new_dir[i]
                print(time.time())
                print('success')
                return pos, dir
            if util == 0 and result:
                mustfail = False
            list_utility[i] = util * 5
            list_res[i] = result
            list_res1[i] = result
            if (time.time_ns() - start_time) >= timeconstraint:
                ind1=i
                break
        sndlayindex=[]
        sndpos=[]
        sndstate=[]
        branchcount=[]
        if (time.time_ns() - start_time) >= timeconstraint and not mustfail:
                i = random.randint(0, (ind1))
                while list_utility[i]<0:
                    i = random.randint(0, (ind1))
                pos = list_new_pos[i]
                dir = list_new_dir[i]
                print(time.time())
                print('no time')
                return pos, dir
        if (time.time_ns() - start_time) >= timeconstraint and mustfail:
            i = random.randint((ind1), (len(list_new_dir)-1))
            pos = list_new_pos[i]
            dir = list_new_dir[i]
            print(time.time())
            print('no time')
            return pos, dir

        for i in range(len(list_res)):
            if list_res[i] == False:
                list_new_board2, list_new_pos2, _ = self.all_next_state(list_new_board[i], list_new_pos[i], adv_pos, False)
                nottie=False
                if len(list_new_pos2) > 0:
                    list_utility[i]=100
                    for j in range(len(list_new_pos2)):
                        result1, util1 = self.check_endgame(list_new_board2[j], list_new_pos[i], list_new_pos2[j])
                        if util1 == -1:
                            list_utility[i] = util1 * 100
                            list_res[i] = result1
                            break
                        elif list_utility[i] != -100 and not list_res[i] and util1 == 0 and not result1:
                            list_utility[i] = 0
                            list_res[i]=False
                            nottie=True
                        elif list_utility[i] != -100 and not list_res[i] and util1 == 0 and  result1 and not nottie:
                            list_utility[i] = 0
                            list_res[i]=True
                    if list_utility[i] > 0:
                        pos = list_new_pos[i]
                        dir = list_new_dir[i]
                        print(time.time())
                        print('success2')
                        return pos, dir
                    if list_utility[i]==0:
                        mustfail=False
                        sndlayindex.append(i)
                        sndpos.append(list_new_pos2)
                        sndstate.append(list_new_board2)
                        branchcount.append(len(list_new_pos2))
                    if (time.time_ns() - start_time) >= timeconstraint:
                        ind1 = i
                        break
        if (time.time_ns() - start_time) >= timeconstraint and not mustfail:
                i = random.randint(0, (ind1))
                while list_utility[i]<0:
                    i = random.randint(0, (ind1))
                pos = list_new_pos[i]
                dir = list_new_dir[i]
                print(time.time())
                print('no time')
                return pos, dir
        if (time.time_ns() - start_time) >= timeconstraint and mustfail:
            i = random.randint((ind1), (len(list_new_dir)-1))
            pos = list_new_pos[i]
            dir = list_new_dir[i]
            print(time.time())
            print('no time')
            return pos, dir

        if len(sndlayindex)==0:
            mustfail=True
        if mustfail:
            i=random.randint(0, (len(list_new_dir) - 1))
            while list_res1[i]:
                i = random.randint(0, (len(list_new_dir) - 1))
            pos = list_new_pos[i]
            dir = list_new_dir[i]
            print(time.time())
            print('fail')
            return pos, dir
        #self.rand_simulation(board, my_pos, adv_pos, my_turn, max_step)
        qo=[-1000]*len(sndpos)
        no=[0]*len(sndpos)
        so=[0]*len(sndpos)
        max=-1000
        q1=[]
        n1=[]
        s1=[]
        fv=[0]*len(sndpos)
        min=[1000]*len(sndpos)
        minid=[0]*len(sndpos)
        count=0
        max_step = (board.shape[0] + 1) // 2
        for i in range(len(sndpos)):
            count+=1
            no[i]+=1
            k=sndlayindex[i]
            cnt=branchcount[i]
            initlist1=[0]*cnt
            q1.append(initlist1)
            initlist2=[0]*cnt
            n1.append(initlist2)
            initlist3=[0]*cnt
            s1.append(initlist3)
            rst=self.rand_simulation(list_new_board[k], list_new_pos[k], adv_pos,  False, max_step)
            tmputil =10 * rst
            so[i] =tmputil
            qo[i] = tmputil / no[i] + 10 * sqrt(log(count) / (no[i]))
            if qo[i]>max:
                max=qo[i]
            if (time.time_ns() - start_time) >= timeconstraint:
                ind1=i
                break
        if (time.time_ns() - start_time) >= timeconstraint:
            ind = self.findmaxid(qo)
            bri = 0
            if qo[ind] > 10:
                bri = sndlayindex[ind]
            else:
                i = random.randint(0, (ind1))
                bri = sndlayindex[i]
            pos = list_new_pos[bri]
            dir = list_new_dir[bri]
            print(time.time())
            print('no time')
            return pos, dir
        while (time.time_ns() - start_time) < timeconstraint:
            i=self.findmaxid(qo)
            count += 1
            no[i] += 1
            k = sndlayindex[i]
            statebranch=sndstate[i]
            statepos=sndpos[i]
            sa=s1[i]
            na=n1[i]
            qa=q1[i]
            if(fv[i]<(branchcount[i])):
                j=fv[i]
                fv[i]+=1
                rst = self.rand_simulation(statebranch[j], list_new_pos[k], statepos[j],True, max_step)
                tmputil = 10 * rst
                so[i] += tmputil
                na[j]=1
                sa[j]=tmputil
                qa[j]=tmputil / na[j] + 10 * sqrt(log(count) / (na[j]))
                qo[i] = so[i] / no[i] + 10 * sqrt(log(count) / (no[i]))
            else:
                j=self.findminid(qa)
                rst = self.rand_simulation(statebranch[j], list_new_pos[k], statepos[j],True,max_step)
                tmputil = 10 * rst
                so[i] += tmputil
                na[j]+=1
                sa[j]+=tmputil
                qa[j]=sa[j] / na[j] + 10 * sqrt(log(count) / (na[j]))
                qo[i] = so[i] / no[i] + 10 * sqrt(log(count) / (no[i]))
        ind=self.findmaxid(qo)
        bri=0
        if qo[ind]>8:
            bri=sndlayindex[ind]
        else:
            i=random.randint(0,(len(qo)-1))
            bri = sndlayindex[i]
        pos = list_new_pos[bri]
        dir = list_new_dir[bri]
        print((time.time_ns() - start_time))
        print('dec')
        return pos, dir

    def set_barrier(self, board, r, c, dir):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board

    def findmaxind(selfself, listint):
        max = listint[0]
        for i in range(len(listint)):
            if listint[i] > max:
                max = listint[i]
        return max

    def findminind(selfself, listint):
        min = listint[0]
        for i in range(len(listint)):
            if listint[i] < min:
                min = listint[i]
        return min

    def findminid(selfself, listint):
        min = listint[0]
        ind=0
        for i in range(len(listint)):
            if listint[i] < min:
                min = listint[i]
                ind=i
        return ind
    def findmaxid(selfself, listint):
        max = listint[0]
        ind = 0
        for i in range(len(listint)):
            if listint[i] > max:
                max = listint[i]
                ind = i
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

    def conscience(self, board, list_step1, adv_pos):
        list_utility = [0] * len(list_step1)
        list_res = [False] * len(list_step1)
        mustfail = True
        for i in range(len(list_step1)):  # my steps
            temp = board.copy()
            (x, y), dir = list_step1[i]
            temp = self.set_barrier(temp, x, y, dir)
            mypos1 = (x, y)
            result, util = self.check_endgame(temp, mypos1, adv_pos)
            if util == 1:
                return i
            if util == 0 and result:
                mustfail = False
            list_utility[i] = util * 100
            list_res[i] = result

        for i in range(len(list_res)):
            if list_res[i] == False:
                (x, y), dir = list_step1[i]
                temp = board.copy()
                temp = self.set_barrier(temp, x, y, dir)
                mypos1 = (x, y)
                advsteps = self.all_steps_possible(temp, adv_pos, mypos1)  # adversary steps
                list_utility[i] = 100
                if len(advsteps) > 0:
                    for j in range(len(advsteps)):
                        temp1 = temp.copy()
                        (x1, y1), dir1 = advsteps[j]
                        advpos1 = (x1, y1)
                        temp1 = self.set_barrier(temp1, x1, y1, dir1)
                        result1, util1 = self.check_endgame(temp1, mypos1, advpos1)
                        if util1 == -1:
                            list_utility[i] = util1 * 100
                            break
                        if list_utility[i] != -100 and util1 == 0:
                            list_utility[i] = 0
                    if list_utility[i] >= 0:
                        mustfail = False
        temp = 0
        for i in range(len(list_step1)):
            if list_utility[i] == 100:
                return i
        if mustfail:
            return random.randint(0, (len(list_step1) - 1))
        else:
            found = False
            while not found:
                temp = random.randint(0, (len(list_step1) - 1))
                if list_utility[temp] >= 0:
                    found = True
        return temp

    def all_next_state(self, board, my_pos, adv_pos, my_turn):
        list_new_board, list_new_pos, list_new_dir = [], [], []
        if my_turn:
            list_step = self.all_steps_possible(board, my_pos, adv_pos)
            for i in range(len(list_step)):
                temp = deepcopy(board)
                (x, y), dir = list_step[i]
                temp = self.set_barrier(temp, x, y, dir)
                list_new_board.append(temp)
                list_new_pos.append((x, y))
                list_new_dir.append(dir)
        else:
            list_step = self.all_steps_possible(board, adv_pos, my_pos)
            for i in range(len(list_step)):
                temp = deepcopy(board)
                (x, y), dir = list_step[i]
                temp = self.set_barrier(temp, x, y, dir)
                list_new_board.append(temp)
                list_new_pos.append((x, y))
                list_new_dir.append(dir)
        zip_list = list(zip(list_new_board, list_new_pos, list_new_dir))
        np.random.shuffle(zip_list)
        shuffle_new_board, shuffle_new_pos, shuffle_new_dir = zip(*zip_list)
        shuffle_new_board = list(shuffle_new_board)
        shuffle_new_pos = list(shuffle_new_pos)
        shuffle_new_dir = list(shuffle_new_dir)
        return shuffle_new_board, shuffle_new_pos, shuffle_new_dir

    def rand_simulation(self, board, my_pos, adv_pos, my_turn, max_step):
        if my_turn:
            result, util = self.check_endgame(board, my_pos, adv_pos)
            if result == True:
                return util
            else:
                temp = deepcopy(board)
                r, c = my_pos
                ra, ca = adv_pos
                while not (result):
                    rand_step = np.random.randint(0, max_step + 1)
                    for _ in range(rand_step):
                        # print("current pos before:", r, c)
                        # print("current cell barrier:",temp[r,c])
                        dir = np.random.permutation(4)
                        # print("random perm dir:",dir)
                        for j in range(5):
                            if j == 4:
                                break
                            rand_dir = dir[j]
                            # print("choosen dir dir:",dir[j],"barrier cond:",temp[r,c,rand_dir],"adv_pos:",ra,ca)
                            if not (temp[r, c, rand_dir]):
                                if (r == ra):
                                    if rand_dir == 2 or rand_dir == 0:
                                        break
                                    elif rand_dir == 1:
                                        if ca != (c + 1):
                                            break
                                    elif rand_dir == 3:
                                        if ca != (c - 1):
                                            break
                                elif (c == ca):
                                    if rand_dir == 1 or rand_dir == 3:
                                        break
                                    elif rand_dir == 0:
                                        if ra != (r - 1):
                                            break
                                    elif rand_dir == 2:
                                        if ra != (r + 1):
                                            break
                                else:
                                    break
                        # if j==5, meaning current player has no where to go, simply stop moving and put a barrier, result in lose
                        if j == 4:
                            break
                        # print("choosen random exp dir:", rand_dir)
                        if rand_dir == 0:
                            r -= 1
                        elif rand_dir == 1:
                            c += 1
                        elif rand_dir == 2:
                            r += 1
                        else:
                            c -= 1
                    # print("current position:", r, c)
                    rand_barrier_dir = np.random.permutation(4)
                    # print("random perm:",rand_barrier_dir)
                    for k in range(4):
                        b_dir = rand_barrier_dir[k]
                        if not (temp[r, c, b_dir]):
                            break
                    # print("b_dir:", b_dir)
                    temp = self.set_barrier(temp, r, c, b_dir)
                    result, util = self.check_endgame(temp, (ra, ca), (r, c))
                    if result == True:
                        return -util
                    # print("********")
                    rand_step = np.random.randint(0, max_step + 1)
                    for _ in range(rand_step):
                        # print("current pos before:", ra, ca)
                        # print("current cell barrier:",temp[ra,ca])
                        dir = np.random.permutation(4)
                        # print("random perm dir:",dir)
                        for j in range(5):
                            if j == 4:
                                break
                            rand_dir = dir[j]
                            # print("choosen dir dir:",dir[j],"barrier cond:",temp[ra,ca,rand_dir],"adv_pos", r, c)
                            if not (temp[ra, ca, rand_dir]):
                                if (r == ra):
                                    if rand_dir == 0 or rand_dir == 2:
                                        break
                                    elif rand_dir == 1:
                                        if c != (ca + 1):
                                            break
                                    elif rand_dir == 3:
                                        if c != (ca - 1):
                                            break
                                elif (c == ca):
                                    if rand_dir == 1 or rand_dir == 3:
                                        break
                                    elif rand_dir == 0:
                                        if r != (ra - 1):
                                            break
                                    elif rand_dir == 2:
                                        if r != (ra + 1):
                                            break
                                else:
                                    break
                        # print("choosen random exp dir:", rand_dir)
                        if j == 4:
                            break
                        if rand_dir == 0:
                            ra -= 1
                        elif rand_dir == 1:
                            ca += 1
                        elif rand_dir == 2:
                            ra += 1
                        else:
                            ca -= 1
                    # print("current position:", r, c)
                    rand_barrier_dir = np.random.permutation(4)
                    # print("random perm:",rand_barrier_dir)
                    for k in range(4):
                        b_dir = rand_barrier_dir[k]
                        if not (temp[ra, ca, b_dir]):
                            break
                    # print("b_dir:", b_dir)
                    temp = self.set_barrier(temp, ra, ca, b_dir)
                    result, util = self.check_endgame(temp, (r, c), (ra, ca))
                return util
        else:

            result, util = self.check_endgame(board, adv_pos, my_pos)
            if result == True:
                return - util
            else:
                temp = deepcopy(board)
                r, c = my_pos
                ra, ca = adv_pos
                while not (result):
                    rand_step = np.random.randint(0, max_step + 1)
                    for _ in range(rand_step):
                        # print("current pos before:", ra, ca)
                        # print("current cell barrier:",temp[ra,ca])
                        dir = np.random.permutation(4)
                        # print("random perm dir:",dir)
                        for j in range(5):
                            if j == 4:
                                break
                            rand_dir = dir[j]
                            # print("choosen dir dir:",dir[j],"barrier cond:",temp[ra,ca,rand_dir],"adv_pos:", r,c)
                            if not (temp[ra, ca, rand_dir]):
                                if (r == ra):
                                    if rand_dir == 0 or rand_dir == 2:
                                        break
                                    elif rand_dir == 1:
                                        if c != (ca + 1):
                                            break
                                    elif rand_dir == 3:
                                        if c != (ca - 1):
                                            break
                                elif (c == ca):
                                    if rand_dir == 1 or rand_dir == 3:
                                        break
                                    elif rand_dir == 0:
                                        if r != (ra - 1):
                                            break
                                    elif rand_dir == 2:
                                        if r != (ra + 1):
                                            break
                                else:
                                    break
                        # print("choosen random exp dir:", rand_dir)
                        if j == 4:
                            break
                        if rand_dir == 0:
                            ra -= 1
                        elif rand_dir == 1:
                            ca += 1
                        elif rand_dir == 2:
                            ra += 1
                        else:
                            ca -= 1
                    # print("current position:", r, c)
                    rand_barrier_dir = np.random.permutation(4)
                    # print("random perm:",rand_barrier_dir)
                    for k in range(4):
                        b_dir = rand_barrier_dir[k]
                        if not (temp[ra, ca, b_dir]):
                            break
                    # print("b_dir:", b_dir)
                    temp = self.set_barrier(temp, ra, ca, b_dir)
                    result, util = self.check_endgame(temp, (r, c), (ra, ca))

                    if result == True:
                        return util

                    # print("#############")

                    rand_step = np.random.randint(0, max_step + 1)
                    for _ in range(rand_step):
                        # print("current pos before:", r, c)
                        # print("current cell barrier:",temp[r,c])
                        dir = np.random.permutation(4)
                        # print("random perm dir:",dir)
                        for j in range(5):
                            if j == 4:
                                break
                            rand_dir = dir[j]
                            # print("choosen dir dir:",dir[j],"barrier cond:",temp[r,c,rand_dir],"adv_pos",ra,ca)
                            if not (temp[r, c, rand_dir]):
                                if (r == ra):
                                    if rand_dir == 0 or rand_dir == 2:
                                        break
                                    elif rand_dir == 1:
                                        if ca != (c + 1):
                                            break
                                    elif rand_dir == 3:
                                        if ca != (c - 1):
                                            break
                                elif (c == ca):
                                    if rand_dir == 1 or rand_dir == 3:
                                        break
                                    elif rand_dir == 0:
                                        if ra != (r - 1):
                                            break
                                    elif rand_dir == 2:
                                        if ra != (r + 1):
                                            break
                                else:
                                    break
                        # print("choosen random exp dir:", rand_dir)
                        if j == 4:
                            break
                        if rand_dir == 0:
                            r -= 1
                        elif rand_dir == 1:
                            c += 1
                        elif rand_dir == 2:
                            r += 1
                        else:
                            c -= 1
                    # print("current position:", r, c)
                    rand_barrier_dir = np.random.permutation(4)
                    # print("random perm:",rand_barrier_dir)
                    for k in range(4):
                        b_dir = rand_barrier_dir[k]
                        if not (temp[r, c, b_dir]):
                            break
                    # print("b_dir:", b_dir)
                    temp = self.set_barrier(temp, r, c, b_dir)
                    result, util = self.check_endgame(temp, (ra, ca), (r, c))

                return -util


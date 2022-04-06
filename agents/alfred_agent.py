# Student agent: Add your own agent here
from copy import deepcopy
from secrets import choice
# from xxlimited import new
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time

class Node:
    def __init__(self,current_game_status,parent):
        self.parent = parent
        self.child = []
        self.current_game_status = current_game_status

    def getStatus(self):
        return self.current_game_status

    def getChild(self):
        return self.child

    def addChild(self,input):
        self.child.append(input)

    def getParent(self):
        return self.parent

    def getChildMax(self):
        max_win = 0
        if len(self.child) ==0:
            return None, -1

        indices = np.arange(len(self.child))
        c = list(zip(self.child,indices))
        np.random.shuffle(c)
        self.child, indices = zip(*c)
        #self.child = self.child[indices]

        max_c = self.child[len(self.child)-1]
        max_index = len(self.child)-1
        #print("child has length of: ",len(self.child))
        counter = 0
        for i in range(len(self.child)):
            #print("counter: ", counter)
            counter += 1
            if (self.child[i].getStatus().getVisitCount() != 0):
                cur = self.child[i].getStatus().getWinNumber()/self.child[i].getStatus().getVisitCount()
                #print("win number",self.child[i].getStatus().getWinNumber())
                #print("Cur = ",cur)

                if cur > max_win:
                    #print("updated")
                    max_win = cur
                    max_c = self.child[i]
                    max_index = indices[i]

        return max_c.getStatus().getMyPos(), max_index


class Tree:
    def __init__(self,root):
        self.root = root

    def getRoot(self):
        return self.root

class NodeStatus:
    def __init__(self,chess_board, my_pos, adv_pos, max_step,turn):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.visitCount = 0
        self.winNumber = 0
        self.turn = turn        # 0 for my turn, 1 for opponent turn
        self.move = ((-1, 0), (0, 1), (1, 0), (0, -1))

    ##def getNodeStatus(self):
    def getVisitCount(self):
        return self.visitCount

    def getWinNumber(self):
        return self.winNumber

    def getMyPos(self):
        return self.my_pos

    def getAdvPos(self):
        return self.adv_pos

    def getMaxStep(self):
        return self.max_step

    def changeMaxStep(self,new_step):
        self.max_step = new_step

    def getChessBoard(self):
        return self.chess_board

    def getTurn(self):
        return self.turn

    def increaseVisitCount(self):
        self.visitCount += 1

    def increaseWinNumber(self):
        self.winNumber += 1

    def setTurn(self,turn):
        self.turn = turn

    def switchPlayer(self):
        if self.turn ==0:
            self.turn = 1
        else:
            self.turn = 0

    def updateMyPos(self,pos,dir):
        self.my_pos = pos
        self.chess_board[pos[0]][pos[1]][dir] = True
        move_a = self.move[dir]
        if dir == 0:
            oppose_dir = 2
        elif dir ==1:
            oppose_dir = 3
        elif dir ==2:
            oppose_dir = 0
        else:
            oppose_dir = 1
        self.chess_board[pos[0]+move_a[0]][pos[1]+move_a[1]][oppose_dir] = True

    def updatePos(self,pos,dir):
        if self.turn == 1:
            self.adv_pos = pos
        else:
            self.my_pos = pos
        self.chess_board[pos[0]][pos[1]][dir] = True
        move_a = self.move[dir]
        if dir == 0:
            oppose_dir = 2
        elif dir ==1:
            oppose_dir = 3
        elif dir ==2:
            oppose_dir = 0
        else:
            oppose_dir = 1
        self.chess_board[pos[0]+move_a[0]][pos[1]+move_a[1]][oppose_dir] = True




class MCTS:
    def __init__(self,chess_board,my_pos,adv_pos,max_step,turn):
        self.tree = Tree(Node(NodeStatus(chess_board, my_pos, adv_pos, max_step,turn),None))
        self.root = self.tree.getRoot()
        self.validbar = []
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.chessboard = chess_board

    def getSt(self):
        return self.root.getStatus()

    def getNextStep(self):
        start = time.time_ns()
        all = self.getAllPossibleMove(self.root.getStatus(),self.my_pos,self.max_step)
        #board_status = deepcopy(self.root.getStatus())
        for pos, dir in all:
            board_status = deepcopy(self.root.getStatus())
            board_status.updateMyPos(pos,dir)
            #board_status.switchPlayer()
            isend, result = self.check_endgame(board_status)    # result = 0, my win
            if isend and result==0:
                #print("ever here??")
                return (pos, dir)
        new_step = int(self.max_step/4)
        if new_step ==0:
            new_step += 1
        if(np.abs(self.my_pos[0]-self.adv_pos[0])+np.abs(self.my_pos[1]-self.adv_pos[1])>4):# or list(self.root.getStatus().getChessBoard()[self.my_pos[0]][self.my_pos[1]]).count(True)==3): # maybe 2?
                #print("newly added")
            best = None
            # desired_barrier = []
            # if(self.my_pos[0]-self.adv_pos[0]>0):
            #     desired_barrier.append(0)
            # else:
            #     desired_barrier.append(2)
            # if(self.my_pos[1]-self.adv_pos[1]>0):
            #     desired_barrier.append(3)
            # else:
            #     desired_barrier.append(1)
            #desire_barrier =
            for end_pos, barrier_dir in all:
                #if self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,barrier_dir): # 0 is my turn, 1 is opponent turn
                if best == None:
                    best = (end_pos,barrier_dir)
                else:
                    cur_distance = np.abs(best[0][0]-self.adv_pos[0]) + np.abs(best[0][1]-self.adv_pos[1])
                    #print("current distance: ", cur_distance)
                    bar_around = list(self.root.getStatus().getChessBoard()[best[0][0]][best[0][1]]).count(True)
                    end_pos_bar = list(self.root.getStatus().getChessBoard()[end_pos[0]][end_pos[1]]).count(True)
                    if(end_pos[0]==0 or end_pos[0]==self.root.getStatus().getChessBoard().shape[0]-1):
                        end_pos_bar += 1
                    if(end_pos[1]==0 or end_pos[0]==self.root.getStatus().getChessBoard().shape[0]-1):
                        end_pos_bar += 1
                    if (np.abs(end_pos[0]-self.adv_pos[0]) + np.abs(end_pos[1]-self.adv_pos[1]))<=cur_distance:
                        if (end_pos_bar<=bar_around):
                            #print("best updated")
                            best = (end_pos,barrier_dir)
                            row_dis = end_pos[0]-self.adv_pos[0]
                            col_dis = end_pos[1]-self.adv_pos[1]
                            if(np.abs(row_dis)>np.abs(col_dis)):
                                if(row_dis > 0):
                                    if self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,0):
                                        best = (end_pos,0)
                                else:
                                    if self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,2):
                                        best = (end_pos,2)
                            else:
                                if(col_dis > 0):
                                    if self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,3):
                                        best = (end_pos,3)
                                else:
                                    if self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,1):
                                        best = (end_pos,1)

                            # if (self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,desired_barrier[0])):
                            #     best = (end_pos,desired_barrier[0])
                            # elif (self.check_valid_step(self.root.getStatus(),0,self.root.getStatus().getMyPos(),end_pos,desired_barrier[0])):
                            #     best = (end_pos,desired_barrier[1])
                            # else:
                            #     best = (end_pos,barrier_dir)
            return best
        else:

            good_node = self.root
            self.expand(good_node)
            node_to_explore = good_node
            while(time.time_ns()-start<1800000000):    # need to set limit
                #print("max step is: ", self.max_step)
                #print("distance: ",np.abs(self.my_pos[0]-self.adv_pos[0])+np.abs(self.my_pos[1]-self.adv_pos[1]))
                #good_node = self.getGoodNode(self.root)

                #print("MCTS: after expand")
                if(len(good_node.getChild())>0):
                    node_to_explore = good_node.getChild()[np.random.randint(len(good_node.getChild()))]
                else:
                    tmp  = good_node.getStatus()
                    #tmp.changeMaxStep(self.max_step)
                    while True:
                        output = self.randomstep(tmp)
                        if output != -1 or output != -2:
                            return output
                    #return self.randomstep(tmp)
                #print("MCTS: random child")
                randomplayresult = self.randomplay(node_to_explore)
                # if randomplayresult:
                #     print("win")
                #print("MCTS: after random play")
                self.backpropagation(node_to_explore,randomplayresult)
                #counter += 1
                #print("MCTS step count: ", counter)
            winner, bar_i = self.root.getChildMax()

            return winner, self.validbar[bar_i]

    # def getGoodNode(self,node):
    #     good_node = node
    #     while(len(good_node.getChild())!=0):
    #         good_node = self.getNodeUCT(good_node)

    #     return good_node

    # def calculate_UCT(self,winNumber,visitCount,totalCount):
    #     if (visitCount==0):
    #         return 0# max number
    #     else:
    #         return winNumber/visitCount+np.sqrt(2*np.log(totalCount)/visitCount)

    # def getNodeUCT(self,node):
    #     totalCount = node.getStatus().getVisitCount()
    #     allUCT = [self.calculate_UCT(element.getStatus().getWinNumber(),element.getStatus().getVisitCount(),totalCount) for element in node.getChild()]
    #     #print(allUCT)
    #     #allUCT[allUCT==2147483647]=-1
    #     #print("allUCT: ",len(allUCT))
    #     return node.getChild()[np.argmax(allUCT)]

    def expand(self,node):
        my_pos= node.getStatus().getMyPos()
        step = node.getStatus().getMaxStep()
        #print("expand: ",step)
        if self.max_step<5:
            new_step = self.max_step
        else:
            new_step = 4
        expand_step = 3 if self.max_step>3 else self.max_step
        # new_step = int(self.max_step)
        # if new_step == 0:
        #     new_step += 1
        allpossiblenextstatus = self.getAllPossibleMove(node.getStatus(),my_pos,expand_step)
        for end_pos, barrier_dir in allpossiblenextstatus:
            #if self.check_valid_step(node.getStatus(),0,node.getStatus().getMyPos(),end_pos,barrier_dir): # 0 is my turn, 1 is opponent turn
            if(self.eval_step(node.getStatus().getChessBoard(),end_pos)):
                new_board_status = deepcopy(node.getStatus())
                new_board_status.updateMyPos(end_pos,barrier_dir)
                new_board_status.switchPlayer()
                self.validbar.append(barrier_dir)
                new_board_status.changeMaxStep(new_step)
                node.addChild(Node(new_board_status,node)) # create new node


    def eval_step(self,board,end_pos):
        if list(board[end_pos[0]][end_pos[1]]).count(True)>=2:
            return False
        else:
            return True


    def getAllPossibleMove(self,chess_board_status,pos,step_size):
        moves = []
        for i in range(0,step_size+1):
            for r in range(i+1):
                row_r = pos[0]+r
                column_u = pos[1]+(i-r)
                row_l = pos[0]-r
                column_d = pos[1]-(i-r)
                for b in range(4):
                    if(self.check_valid_step(chess_board_status, chess_board_status.getTurn(),pos, [row_r,column_u], b)):
                        moves.append((np.array([row_r,column_u]),b))
                    if(self.check_valid_step(chess_board_status, chess_board_status.getTurn(),pos, [row_r,column_d], b)):
                        moves.append((np.array([row_r,column_d]),b))
                    if(self.check_valid_step(chess_board_status, chess_board_status.getTurn(),pos, [row_l,column_u], b)):
                        moves.append((np.array([row_l,column_u]),b))
                    if(self.check_valid_step(chess_board_status, chess_board_status.getTurn(),pos, [row_l,column_d], b)):
                        moves.append((np.array([row_l,column_d]),b))
        #print("all possible moves have value of: ", len(moves))
        return moves

    def backpropagation(self,node,result): # result = 0, my win, turn = 1, opponent win
        node.getStatus().increaseVisitCount()
        if(result == 0):
            node.getStatus().increaseWinNumber()

    def randomplay(self,node):
        st = time.time()
        current_chess_board = deepcopy(node.getStatus())
        #current_turn = current_chess_board.getTurn()

        isend, result = self.check_endgame(current_chess_board)
        if isend:
            return result
        
        counter = 0
        while(True):
            #print("need to implement")
            # generate random move

            random_move, random_dir = self.randomstep(current_chess_board)
            if random_dir == -2:
                #print("ever reached?? -2")
                #print("win from -2")
                return 0
            elif random_dir == -1:
                #print("ever reached?? -1")
                #print("lose from -1")
                return 1
            #print("Counter: ",counter,"next position",random_move, "Barrier Direction: ",random_dir)
            # check validity
            # update chessboard
            current_chess_board.updatePos(random_move,random_dir)
            current_chess_board.switchPlayer()
            # updata is end
            isend, result = self.check_endgame(current_chess_board)
            #counter += 1
            if isend:
                #print("ever end??")
                #print("simulated iterations: ",counter)
                break
            #print("infinite loop")
            
            #else:
                #print("lose")
        print("end time:", time.time()-st)
        return result   # 0 my win, 1 adv win, -1 tie

    def randomstep(self,chess_board_status):
        # Moves (Up, Right, Down, Left)
        if chess_board_status.getTurn() == 0:   # my turn
            #ori_pos = deepcopy(chess_board_status.getMyPos())
            my_pos = chess_board_status.getMyPos()
            #adv_pos = chess_board_status.getAdvPos()
        else:
            #ori_pos = deepcopy(chess_board_status.getAdvPos())
            my_pos = chess_board_status.getAdvPos()
            #adv_pos = chess_board_status.getMyPos()
        #ori_pos = deepcopy(chess_board_status.getMyPos())

        #all_valid = self.getAllPossibleMove(chess_board_status,my_pos,chess_board_status.getMaxStep())
        #if len(all_valid)==0:
        #    if chess_board_status.getTurn() == 0:
        #        return my_pos, -1   # my loss
        #    else:
        #        return my_pos, -2 # my win
        #else:
        #    return all_valid[np.random.randint(0,len(all_valid))]

        counter = 0
        direction = [1,-1]
        while(True):
            if counter >= 100:
                if chess_board_status.getTurn() == 0:
                    return my_pos, -1   # my loss
                else:
                    return my_pos, -2 # my win
            else:
                step_size = np.random.randint(0,chess_board_status.getMaxStep())
                b = np.random.randint(0,4)
                r_size = np.random.randint(0,step_size+1)
                c_size = step_size-r_size
                next_pos = (my_pos[0]+np.random.choice(direction)*r_size,my_pos[1]+np.random.choice(direction)*c_size)
                if self.check_valid_step(chess_board_status,chess_board_status.getTurn(),my_pos,next_pos,b):
                    return (next_pos,b)
            counter += 1

        


    def check_valid_step(self,chess_board_status, turn,start_pos, end_pos, barrier_dir):
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
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if r>= chess_board_status.getChessBoard().shape[0] or c>= chess_board_status.getChessBoard().shape[0]:
            return False
        if chess_board_status.getChessBoard()[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = chess_board_status.getMyPos() if turn==1 else chess_board_status.getAdvPos()

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            #("cur_pos from validation",cur_pos)
            r, c = cur_pos
            if cur_step == chess_board_status.getMaxStep():
                break
            for dir, move in enumerate(moves):
                if chess_board_status.getChessBoard()[r, c, dir]:
                    continue
                move_0, move_1 = move
                next_pos = (r+move_0,c+move_1)
                #print("next_pos: ",next_pos)
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def check_endgame(self,chess_board_status):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        #print("in check end game")
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        father = dict()
        for r in range(chess_board_status.getChessBoard().shape[0]):
            for c in range(chess_board_status.getChessBoard().shape[1]):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(chess_board_status.getChessBoard().shape[0]):
            for c in range(chess_board_status.getChessBoard().shape[1]):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board_status.getChessBoard()[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(chess_board_status.getChessBoard().shape[0]):
            for c in range(chess_board_status.getChessBoard().shape[0]):
                find((r, c))
        p0_r = find(tuple(chess_board_status.getMyPos()))
        p1_r = find(tuple(chess_board_status.getAdvPos()))
        p0_score = list(father.values()).count(p0_r)    # my position
        p1_score = list(father.values()).count(p1_r)    # opponent position
        if p0_r == p1_r:
            return False, -2
        player_win = None
        #win_blocks = -1
        if p0_score > p1_score: # my win
            player_win = 0
            #win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            #win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, player_win #, p0_score, p1_score



@register_agent("alfred_agent")
class AlfredAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(AlfredAgent, self).__init__()
        self.name = "AlfredAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
        #print("master:",type(max_step))
        start = time.time_ns()
        algorithm = MCTS(chess_board,my_pos,adv_pos,max_step,0)
        next_pos,barrier = algorithm.getNextStep()
        #next_pos,barrier = algorithm.randomstep(algorithm.getSt())
        time_spent = (time.time_ns()-start)
        if(time_spent>2000000000):
             print("Error: runing out of time")
        #elif(time_spent>1800000000):
             #print("almost out of time")
        # else:
        #     print("time is ok")
        return (tuple(next_pos),barrier)
        #return my_pos, self.dir_map["u"]

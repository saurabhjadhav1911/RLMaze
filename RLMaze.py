#############################################################
#   Author : Saurabh Jadhav                                 #
#   https://github.com/saurabhjadhav1911/RLMaze.git         #
#   saurabhjadhav1911@gmail.com                             #
#############################################################

#   C:\Users\saurabhj\OneDrive\Documents\Python Scripts\RL  #

import numpy as np
import random
import cv2
from copy import deepcopy
N=0
E=1
S=2
W=3
exit=1
dir=['N','E','S','W']


class RLMaze():
    """


    """
    def __init__(self,height=None,width=None,goal=None,nogoal=None,start=None,obstacles=None):
        
        #initialise cordinates and maze parameters

        self.height=height or 3
        self.width= width or 4
        self.goal=goal or [0,3]
        self.nogoal= nogoal or [1,3]
        self.obstacles=obstacles or [[1,1]]
        self.start=start or [2,0]
        self.pos=self.start
        self.maze=self.generate_maze()
        self.reward=self.maze
        self.prev_pos=None
        self.actions=[0,1,2,3]
        self.neighbours=[[-1,0],[0,1],[1,0],[0,-1]]
        #self.move=np.array([[0,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,2],[2,1,1],[1,2,2],[2,2,1],[2,2,3],[3,2,2],[2,3,3],[3,3,2],[3,3,0],[0,3,3],[3,0,0],[0,0,3]])
        
        #learning parameters
        self.movement_cost=-0.04
        self.alpha=0.8
        self.gamma=0.9
        self.epsilon=0.1
        e=5
        
        #learning variables
        self.policy=""
        self.maxpolicy=""
        self.total_reward=-100
        self.iteration=1
        self.best=[""]
        self.epoch_num=1
        self.cycle_num=0
        self.move_reward=0
        self.exit=1
        self.reached=np.zeros([self.height,self.width])
        self.Q=np.zeros((self.height,self.width,len(self.actions)),dtype=np.float32)
        
        
        #image parameters
        self.pixelpercell=100
        self.pause=25
        self.img = np.zeros([self.pixelpercell*self.maze.shape[0],self.pixelpercell*self.maze.shape[1],3], np.uint8)
        self.img=self.generate_background()

    def generate_maze(self):
        maze=np.zeros([self.height,self.width])
        maze[self.goal[0],self.goal[1]]=1
        maze[self.nogoal[0],self.nogoal[1]]=-1
        for ob in self.obstacles:
            maze[ob[0],ob[1]]=-10
        return maze

    def generate_background(self):
        self.img[:,:,:]=255
        c=self.img.shape[0]
        
        for i in range(1,self.maze.shape[0]):
            a=int(c*i/self.maze.shape[0])
            cv2.line(self.img,(0,a),(self.img.shape[1],a),[0,0,0],2)
        c=self.img.shape[1]
        for i in range(1,self.maze.shape[1]):
            a=int(c*i/self.maze.shape[1])
            cv2.line(self.img,(a,0),(a,self.img.shape[0]),[0,0,0],2)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                spl=False
                if [i,j]==self.goal:
                    color=[0,255,0]
                    spl=True
                elif [i,j]==self.nogoal:
                    color=[0,0,255]
                    spl=True
                if [i,j] in self.obstacles:
                    color=[255,0,0]
                    spl=True
                if spl:
                    a=self.pixelpercell*i
                    c=self.pixelpercell*(i+1)
                    b=self.pixelpercell*j
                    d=self.pixelpercell*(j+1)
                    cv2.rectangle(self.img,(b,a),(d,c),color, thickness=-1)
                    #print(a,b,c,d)

        cv2.imshow('Environment',self.img)
        return self.img

    def getQ(self,state,action):
        if state==self.goal or state==self.nogoal:
            return self.Q[state[0],state[1],0]
        else:
            return self.Q[state[0],state[1],action]

    def putQ(self,state,action,q):
        if state==self.goal or state==self.nogoal:
            self.Q[state[0],state[1],0]=q
        else:
            self.Q[state[0],state[1],action]=q

    def action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            #print("randomaction",action)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
            #print("best action",action)
        return action

    def render(self):
        self.env=self.generate_background()
        a=self.pixelpercell*self.pos[0]
        c=self.pixelpercell*(self.pos[0]+1)
        b=self.pixelpercell*self.pos[1]
        d=self.pixelpercell*(self.pos[1]+1)
        cv2.rectangle(self.env,(b+20,a+20),(d-20,c-20),[255,255,0], thickness=-1)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                c=self.pixelpercell*(i+0.54)
                d=self.pixelpercell*(j+0.28)
                for ac in self.actions:
                    a=c+self.pixelpercell*0.225*self.neighbours[ac][0]
                    b=d+self.pixelpercell*0.225*self.neighbours[ac][1]
                    cv2.putText( self.env,str(int(10000*self.Q[i][j][ac])/100.0)[0:-1],(int(b),int(a)),   cv2.FONT_HERSHEY_PLAIN, 0.8,(0, 0, 0), 1 )

        cv2.imshow('Environment',self.env)
        #cv2.imwrite("CustomEnv/{}E{}.jpg".format(self.epoch_num,self.cycle_num),self.env)
        cv2.waitKey(self.pause)

    def MoveDir(self,state,action):
        h=state[0]+self.neighbours[action][0]
        w=state[1]+self.neighbours[action][1]
        if ((-1<h<self.height) and (-1<w<self.width)):
            if(self.maze[h][w]!=-10):
                state[0]=state[0]+self.neighbours[action][0]
                state[1]=state[1]+self.neighbours[action][1]
        #print(self.pos)

        if(state==self.goal) or (state==self.nogoal):
            exit=0
            print("exit")
        else:
            exit=1
        move_reward=(self.reward[state[0]][state[1]]+self.movement_cost)
        return exit,state,move_reward

    def Movec(self,state,act):
        c=""
        exit,state,move_reward=self.MoveDir(state,act)
        c+=str(dir[act])
        self.policy+=c
        self.policy+='|'
        self.reached[state[0]][state]+=1
        return exit,state,move_reward

    def learn(self):
        maxr=self.total_reward
        while(1):
            self.Epoch()
            self.render()
            #print("total reward",self.total_reward)
            self.epoch_num+=1
            if ((self.total_reward) >= maxr):
                maxr=self.total_reward
                
                print("Epoch {} with max rewdrd={} and policy = {}".format(self.epoch_num,maxr,self.policy))

    def Epoch(self):
        self.reset()
        self.exit=1
        n=0
        self.policy=""
        self.cycle_num=0
        while(self.exit==1):
            self.prev_pos=deepcopy(self.pos)
            n+=1
            act=self.action(self.pos)
            Q=self.getQ([self.prev_pos[0],self.prev_pos[1]],act)
            self.exit,pos,self.move_reward=self.Movec(self.pos,act)
            self.total_reward+=self.move_reward
            self.render()
            q = [self.getQ(pos, a) for a in self.actions]
            maxQ = max(q)
            #print(maxQ)
            Q=Q+self.alpha*(self.move_reward+(self.gamma*maxQ)-Q)
            self.putQ([self.prev_pos[0],self.prev_pos[1]],act,Q)
            self.pos=pos
            self.cycle_num+=1
        #print("total reward",self.total_reward)
        return exit

    def reset(self):
        self.total_reward=0
        #print(self.pos)
        self.pos=[self.start[0],self.start[1]]
        #print(self.policy)
        self.render()
        print("reset")
        self.reached=np.zeros([self.height,self.width])
        self.exit=1

########   Simple Default Maze  ########
#rl=RLMaze()

########   Custom Maze  ########
rl=RLMaze(height=6,width=8,goal=[0,7],nogoal=[1,7],start=[5,0],obstacles=[[1,1],[2,3],[4,5]])

rl.learn()

import os
import random as ran
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import copy
import sys
import find_basestation
import networkx as nx

from geopy import distance
from scipy.spatial.distance import cdist
from gym import error, spaces, utils
from gym.utils import seeding
from termcolor import colored
from collections import defaultdict
from random import random

#MOBILE EDGE COMPUTING ENVIRONMENT

class MEC_env(object):
        
    def __init__(self,batch,no_of_users):
        self.info = {}
        self.finish = 0
        self.G = nx.Graph()
        self.T = nx.Graph()
        self.pos_loc = {}
        #self.if_mec = ["yes","no","yes","yes","no","yes","yes","yes","no","yes","no","no","no","yes","no","no","no","no","yes","yes"]
        self.if_mec = ["yes","no","yes","yes","no","yes","yes","yes","no","yes","no","no","no","no","yes","no","no","no","no","yes","yes","no","yes","no","no","yes","no","no","yes","no","yes","yes","yes","yes"]
        #self.if_mec = ['yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no', '', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no']
        self.batch = batch
        self.no_of_users = no_of_users 
        self.no_of_stations = 34
        self.inactive = 0
        self.max_hop = 3
        self.done = False
        self.priority_enabled = False

        self.usercoordinates = np.array([])
        self.global_userlocation = np.array([])
        self.local_userlocation = np.array([])
        self.MEC_location = np.array([])
        self.base_stations = np.array([])
        self.userability = np.array([])
        self.mecability = np.array([])
        self.hop_to_hop = np.zeros((self.no_of_stations,self.no_of_stations))        
        self.previous_USER_base=np.full((self.no_of_users,),-1)
        
        self.user_changed=[i+1 for i in range(self.no_of_users)]
        self.mec_details = defaultdict(list)
        self.exec_mec_details = defaultdict(list)
        self.user_details = defaultdict(list)
        self.mec_dict = defaultdict(list)
        self.user_dict = defaultdict(list)

        self.mec_map = {}
        self.pos_mec = {}
        self.closest_mec = {}

        self.possible = []
        self.MEC = []
        self.impossible = []
        self.previous_action = []
        self.USER_base = []
        self.no_of_MEC = 0
        self.base = []
        #coords = [[1.91,1.08],[7.39,5.81], [18.72,2.51], [0.75,11.36],[1.05, 7.01], [10.11,3.99],[17.26 ,10.53], [15.5,17.0],[19.56,6.], [6.41,11.26], [12.97,18.56], [19.91, 14.44], [0.93,16.24],[9.14, 16.09], [13.31,12.11], [4.95,15.61], [0.38,20.71], [10.30,7.83],[5.24,19.25], [20.88,20.17]]#20stations
        coords = [[1.91,1.08],[7.39,5.81], [18.72,2.51], [0.75,11.36],[1.05, 7.01], [10.11,3.99],[17.26 ,10.53], [15.5,17.0],[19.56,6.], [6.41,11.26], [12.97,18.56],[13.5,22.0], [19.91, 14.44], 
                    [0.93,16.24],[9.14, 16.09], [13.31,12.11], [4.95,15.61], [0.38,20.71], [10.30,7.83],[5.24,19.25], [20.88,20.17],[25.00,21.5],[20.5,9.41],[24.2,10.75],[25.0,5.0],[15.5,24.8],[23.7,25.5],[3.5,24.0],[28.4,25.5],[29.0,15.75],[30.0,7.25],[8.5,23.8],[34.0,16.5],[1.0,30.5]] #34 stations
        '''coords = [[90.02, 51.63], [80.74, 71.11], [10.75, 67.97], [43.29, 39.35], [77.19, 55.84], [59.72, 32.4], [83.37, 98.01], [64.91, 9.0], [49.59, 88.54], [32.92, 63.76], [3.07, 93.8], [94.72, 86.08], [44.96, 65.76], [98.3, 61.37], [8.75, 51.23], [8.87, 39.02], [98.48, 13.29], [54.4, 56.08], [47.81, 44.47], [67.78, 61.94], [99.23, 38.84], [11.17, 18.49], [85.77, 6.59], [25.15, 36.68], [17.63, 87.07], [69.95, 68.52], [62.11, 76.14], [73.13, 38.95], [29.85, 21.95], [72.89, 52.07], [49.2, 50.66], [43.57, 78.83], [51.12, 62.07], [23.55, 43.52], [4.85, 33.96], [78.02, 36.99], [27.62, 51.06], [84.4, 61.87], [52.47, 20.66], [81.85, 84.06], [50.6, 1.65],
         [50.29, 14.76], [78.48, 4.25], [13.55, 26.37], [41.96, 59.74], [25.57, 60.15], [58.09, 26.86], [12.03, 89.86], [28.43, 10.67], [98.01, 20.62]]#50 stations'''

        for i in range(self.no_of_stations):
            self.pos_loc[i] = coords[i]
            self.base_stations = np.append(self.base_stations,self.pos_loc[i])
            if self.if_mec[i] == "yes":
                self.no_of_MEC = self.no_of_MEC + 1
                self.possible.append(self.no_of_MEC)
                self.MEC.append(self.no_of_MEC)
                self.MEC_location = np.append(self.MEC_location,self.pos_loc[i])
                self.pos_mec[i] = self.no_of_MEC
        print("BASE STATIONS ", self.base_stations)
        self.MEC_location = np.append(self.MEC_location,[10.,10.])  #fall back
        print("CLOUD STATIONS WITH FALLBACK ", self.MEC_location)
        self.possible.append(self.no_of_MEC+1)
        self.MEC.append(self.no_of_MEC+1)
        self.nearest_base = find_basestation.NearestNeighborRANModel(self.pos_loc,50)

    def generate_points_with_min_distance(self,n, shape, min_dist):
        # compute grid shape based on number of points
        width_ratio = shape[1] / shape[0]
        num_y = np.int32(np.sqrt(n / width_ratio)) + 1
        num_x = np.int32(n / num_y) + 1
        # create regularly spaced neurons
        x = np.linspace(0., shape[1]-1, num_x, dtype=np.float32)
        y = np.linspace(0., shape[0]-1, num_y, dtype=np.float32)
        coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)
        # compute spacing
        init_dist = np.min((x[1]-x[0], y[1]-y[0]))

        # perturb points
        max_movement = (init_dist - min_dist)/2
        noise = np.random.uniform(low=0, high=max_movement, size=(len(coords), 2))
        coords += noise
        return coords

    def calc_reward(self,r_action):
        action=[]
        self.previous_observation = copy.deepcopy(self.observation)
        unchanged_action ,self.correct_action, mapping= self.execute()
        if len(mapping)==0:
            for i in range(self.no_of_MEC+1):
                mapping[i+1] = i+1
        if self.impossible:
            for i in range(len(self.correct_action)):
                for k,v in self.mec_map.items():
                    if self.correct_action[i] == v:
                        action.append(k) 
        else:
            action = self.correct_action
        base = self.base
        self.finish = self.finish + 1
        self.observation,self.base  = self.next_step(unchanged_action)
        return self.previous_observation,action, self.observation,mapping,base

    def find_base(self,user):
        USER_base = self.nearest_base.update_user_access_points(user)
        return USER_base

    def assign(self,user, nearest_mec_index, dist,base_station):
        selected_mec = self.MECloc[nearest_mec_index[0]]
        selected_mec = self.mec_dict[tuple(selected_mec)]
        if self.satisfy(selected_mec,user,base_station):
            self.closest_mec[user] = selected_mec
            return True
        else:
            return False  

    def satisfy(self,mec,user,base_station):
        if self.exec_mec_details[mec][0] >= self.user_details[user][0]:
            if self.exec_mec_details[mec][1]>0:
                mec_in_bs = list(self.pos_mec.keys())[list(self.pos_mec.values()).index(mec)]
                if self.hop_to_hop[base_station][mec_in_bs] <= self.hop :
                    self.exec_mec_details[mec][0] = self.exec_mec_details[mec][0] - self.user_details[user][0]
                    self.exec_mec_details[mec][1] = self.exec_mec_details[mec][1] - 1
                    return True
        return False 


    def execute(self):
        no_of_users = len(self.local_userlocation)
        self.closest_mec =  {k+1: (self.no_of_MEC+1) for k in range(self.no_of_users)}
        self.exec_mec_details = copy.deepcopy(self.mec_details)
        
        USER_base = self.USER_base[:]
        
        for users in range(len(self.local_userlocation)):
            self.hop = 0
            for i in range(self.no_of_users):
                if all(self.global_userlocation[i] == self.local_userlocation[users]):
                    user_num = i
                    break
            self.userloc = []
            base_station = self.USER_base[user_num]
            self.userloc.append(self.local_userlocation[users])
            while(len(self.userloc)>0):
                self.MECloc = []
                exit = False
                self.MECloc = [self.pos_loc[i] for i in self.hop_dist[tuple((base_station,self.hop))] if self.if_mec[i] == "yes"]
                if self.hop == self.max_hop:
                    self.closest_mec[user_num+1] = self.no_of_MEC+1
                    max_hop=True
                    break
                self.hop = self.hop + 1
                if len(self.MECloc)>0:
                    dist = cdist( self.userloc, self.MECloc, metric="euclidean" )
                else:
                    exit = True
                while(not exit):
                    nearest_mec_index = dist.argmin(axis=1)
                    exit = self.assign(user_num+1,nearest_mec_index,dist,base_station)
                    if not exit:
                        dist[0][nearest_mec_index[0]] = 1000.0
                        if all(dist[0] == 1000.0):
                            exit=True
                    else:
                        del self.userloc[0]

        self.closest_mec = list(self.closest_mec.values())
       
        self.closest_mec_changed = [self.closest_mec[i-1] for i in self.user_changed]
        return self.closest_mec, self.closest_mec_changed, self.mec_map

    def change_loc(self,userlocation):
        user_loc = np.reshape(userlocation,(self.no_of_users,2)).tolist()
        #dir_val = defaultdict(lambda: [0.0,0.0], {'N': [3.5,3.0], 'S': [4.0,-2.5], 'W': [-1.5,5.5], 'E': [2.5,3.0]})
        dir_val = defaultdict(lambda: [0.0,0.0], {'N': [1.5,1.0], 'S': [1.0,-2.5], 'W': [-1.5,1.0], 'E': [2.5,1.0]})
        direction = ['N','S','W','E']
        step = []
        #select_users = np.random.randint(1,3)# IF WE WANT TO SEND EVERY OBSERVATION IRRESPECTIVE OF BASE CHANGE THEN MAKE 1 0
        select_users = 3
        x = ran.sample(range(self.no_of_users), select_users)
        x.sort()
        direction = np.random.choice(direction,select_users)
        dir_val = [dir_val[direction[i]] for i in range(select_users)]
        count=0
        for i in range(self.no_of_users):
            if i in x:
                for index in zip(user_loc[x[count]], dir_val[count]):
                    summation = sum(index)
                    if summation > 40.0:
                        summation = 40.0
                    if summation < 0:
                        summation = 0.0
                    step.append(summation)
                count =  count + 1
            else:
                step.append(user_loc[i][0])
                step.append(user_loc[i][1])
        userlocation = []
        usercoordinates = np.around(np.asarray(step),2)
        return usercoordinates



    def next_step(self,r_action):
        self.remove_mec = False
        self.mec_map = {}
        local_userability = []
        previous_base = []
        if self.finish != 1:
            for i in range(self.no_of_users):
                if self.previous_action[i] != r_action[i]:
                    if r_action[i] != self.no_of_MEC+1:
                        if self.previous_action[i] != self.no_of_MEC+1: 
                            self.mecability[self.previous_action[i]-1] = self.mecability[self.previous_action[i]-1] + self.userability[i]
                            self.mecservice[self.previous_action[i]-1] = self.mecservice[self.previous_action[i]-1] + 1 
                            if self.previous_action[i] in self.impossible and self.mecability[self.previous_action[i]-1] > 0 and self.mecservice[self.previous_action[i]-1] > 0:
                                self.impossible.remove(self.previous_action[i])
                                self.possible.append(self.previous_action[i])
                        self.mecability[r_action[i]-1] = self.mecability[r_action[i]-1] - self.userability[i]
                        self.mecservice[r_action[i]-1] = self.mecservice[r_action[i]-1] - 1

                        if self.mecability[r_action[i]-1] == 0 or self.mecservice[r_action[i]-1]==0:
                            self.possible.remove(r_action[i])
                            self.impossible.append(r_action[i])
                        self.mec_details[self.previous_action[i]] = ([self.mecability[self.previous_action[i]-1],self.mecservice[self.previous_action[i]-1]])
                        self.mec_details[r_action[i]] = ([self.mecability[r_action[i]-1],self.mecservice[r_action[i]-1]])
                        self.previous_action[i] = r_action[i]
        else:
            for i in range(self.no_of_users):
                if r_action[i] != self.no_of_MEC+1:
                    self.mecability[r_action[i]-1] = self.mecability[r_action[i]-1] - self.userability[i]
                    self.mecservice[r_action[i]-1] = self.mecservice[r_action[i]-1] - 1
                    if self.mecability[r_action[i]-1] == 0 or self.mecservice[r_action[i]-1]==0:
                        if r_action[i] in self.possible:
                            self.possible.remove(r_action[i])
                            self.impossible.append(r_action[i])
                    self.mec_details[r_action[i]] = ([self.mecability[r_action[i]-1],self.mecservice[r_action[i]-1]])
                    self.previous_action.append(r_action[i])
                else:
                    self.previous_action.append(self.no_of_MEC+1)
                    
        
        #CHANGE BASE STATIONS ACCORDING TO USER LOCATION
        self.usercoordinates = self.change_loc(self.global_userlocation)
        self.global_userlocation = np.reshape(self.usercoordinates,(self.no_of_users,2))
        self.USER_base = self.find_base(self.global_userlocation)
        while(self.USER_base == self.previous_USER_base):
            self.usercoordinates = self.change_loc(self.global_userlocation)
            self.global_userlocation = np.reshape(self.usercoordinates,(self.no_of_users,2))
            self.USER_base = self.find_base(self.global_userlocation)

        self.USER_base = [i if i!=j and i!=self.inactive else -1 for i,j in zip(self.USER_base,self.previous_USER_base)] 
        self.previous_USER_base = [self.USER_base[i] if self.USER_base[i]!=-1 else self.previous_USER_base[i] for i in range(self.no_of_users)] 
        #print(colored(self.USER_base,"yellow"))
        #print(colored(self.previous_USER_base,"yellow"))
        #INACTIVATE A USER
        if self.finish == self.batch-1:
            self.inactivate_user()
            nodes = nx.single_source_shortest_path_length(self.T, self.previous_USER_base[self.inactive-1], cutoff=2)
            nodes = [k for k,v in nodes.items()]
            for i in range(self.no_of_users):
                if self.previous_USER_base[i] in nodes:
                    self.USER_base[i] = self.previous_USER_base[i]
            self.USER_base[self.inactive-1] = -1
        #SET PRIORITY
        if self.finish == 3:
            step = [i for i, x in enumerate(self.USER_base) if x != -1]
            if step:
                self.priority_set(step)
        self.user_dict = defaultdict(list)
        for i in range(len(self.global_userlocation)):
            self.user_dict[tuple(self.global_userlocation[i])] = i+1
        self.user_changed = [self.user_dict[tuple(self.global_userlocation[i])] for i,j in enumerate(self.USER_base) if j!=-1]  #user location of users where base is changed
        self.local_userlocation = [list(self.global_userlocation[i-1]) for i in self.user_changed]
        local_coordinates = [j for i in self.local_userlocation for j in i]
        local_userability = [self.userability[j-1] for j in self.user_changed]
        previous_base = [self.previous_USER_base[j-1] for j in self.user_changed]
        base_coord = [self.pos_loc[i] for i in previous_base]
        base_coord = np.reshape(np.array(base_coord),(len(previous_base)*2,))
        #FOR INVALID MEC
        if self.impossible:
            #print(self.impossible)
            meclocation = self.meclocation[:]
            mecability = self.mecability[:]
            mecservice = self.mecservice[:]
            add=0
            remove =0
            for i in range(self.no_of_MEC+1):
                if (i+1) in self.impossible:
                    meclocation = np.delete(meclocation,i-remove,0)
                    mecability = np.delete(mecability,i-remove, 0)
                    mecservice = np.delete(mecservice,i-remove, 0)
                    remove = remove + 1
                else:
                    add = add + 1
                    self.mec_map[add] = i+1
            #print(self.mec_map)  
            no_of_mec = (self.no_of_MEC - len(self.impossible)) + 1
            meclocation = np.reshape(meclocation,(2*(no_of_mec),))
            #self.observation = (meclocation,mecability,mecservice, local_userability,local_coordinates,previous_base)
            self.observation = (meclocation,mecability,mecservice, local_userability,local_coordinates,base_coord)
        else:
            #self.observation = (self.MEC_location,self.mecability,self.mecservice, local_userability,local_coordinates,previous_base)
            self.observation = (self.MEC_location,self.mecability,self.mecservice, local_userability,local_coordinates,base_coord)
        return self.observation,previous_base
        
    def priority_set(self,step):
        self.priority_enabled = True
        priority = ran.choice(step)
        #print(colored("PRIORITY SERVICE FOR","green"),priority)
        v = self.global_userlocation[priority]
        self.global_userlocation = np.delete(self.global_userlocation,priority,0)
        self.usercoordinates = np.insert(self.global_userlocation, 0, v)
        self.global_userlocation = (np.reshape(self.usercoordinates ,(self.no_of_users,2)))
        self.USER_base.insert(0, self.USER_base.pop(priority)) 
        self.previous_USER_base.insert(0, self.previous_USER_base.pop(priority)) 
        #self.previous_action = self.previous_action.tolist()
        self.previous_action.insert(0, self.previous_action.pop(priority))
        v = self.userability[priority]
        self.userability = np.delete(self.userability,priority,0)
        self.userability = np.insert(self.userability, 0, v)
        for i in range(self.no_of_users):
            self.user_details[i+1]= ([self.userability[i],1])

    def print_val(self):
        print("USERCOORDINATES", self.usercoordinates )
        print("GLOBAL USER LOCATION", self.global_userlocation)
        print("LOCAL USER LOCATION",self.local_userlocation)
        print("USERABILITY", self.userability)
        print("USER DETAILS", self.user_details)
        print("USER DICT", self.user_dict)
        print("USER BASE", self.USER_base)
        print("PREVIOUS USER BASE", self.previous_USER_base)
        print("PREVIOUS ACTION",self.previous_action)
        #print("MIGRATION NEEDED",self.migration)

    def inactivate_user(self):
        self.bool_inactive = True
        self.inactive = np.random.randint(1,self.no_of_users+1)
        '''while(self.USER_base[self.inactive-1] != -1):
            self.inactive = np.random.randint(1,self.no_of_users+1)'''
        
        print(colored("THE USER LEFT","green"), self.inactive)
        mec = self.previous_action[self.inactive-1]- 1 
        if mec != self.no_of_MEC +1:
            self.mecability[mec] = self.mecability[mec] +  self.userability[self.inactive-1]
            self.mecservice[mec] = self.mecservice[mec] + 1
            self.mec_details[mec+1] = ([self.mecability[mec],self.mecservice[mec]])
        self.userability[self.inactive-1] = 0.0
        self.user_details[self.inactive] = ([0.0,0.0])
        if mec+1 in self.impossible and self.mecability[mec] >0.0 and self.mecservice[mec]>0.0:
            self.impossible.remove(mec + 1)
            self.possible.append(mec + 1)

 
    def reset(self):
        #print("reset")
        self.reward=0
        self.finish = 0
        self.possible = []
        self.impossible = []
        self.previous_action = []
        self.inactive = -1
        self.user_changed=[i+1 for i in range(self.no_of_users)]
        self.usercoordinates = np.around(np.random.uniform(low=0,high=30,size=(2*self.no_of_users,)),2)
        self.userability = np.around(np.random.uniform(low=10,high=20,size=(self.no_of_users,)),0)
        self.mecability = np.around(np.random.uniform(low=50,high=100,size=(self.no_of_MEC,)),0)
        self.mecability = np.append(self.mecability,60.0)
        self.mecservice = np.around(np.random.uniform(low=3,high=3,size=(self.no_of_MEC,)),0)
        self.mecservice = np.append(self.mecservice,6.0)
        self.global_userlocation = (np.reshape(self.usercoordinates,(self.no_of_users,2)))
        self.basestations = np.reshape(self.base_stations,(self.no_of_stations,2))
        self.mec_dict[tuple([10.,10.])] = self.no_of_MEC+1
        self.local_userlocation = copy.deepcopy(self.global_userlocation)
        self.meclocation = np.reshape(self.MEC_location,(self.no_of_MEC+1,2))
        for i in range(self.no_of_MEC+1):
            self.mec_dict[tuple(self.meclocation[i])] = i+1
            self.possible.append(i+1)
        for i in range(len(self.global_userlocation)):
            self.user_dict[tuple(self.global_userlocation[i])] = i+1
        for i in range(self.no_of_MEC+1):
            self.mec_details[i+1] = ([self.mecability[i],self.mecservice[i]])
        for i in range(self.no_of_users):
            self.user_details[i+1] = ([self.userability[i],1])
        for i in range(self.no_of_MEC+1):
            self.mec_map[i+1] = i+1
        self.USER_base = self.find_base(self.local_userlocation)
        self.previous_USER_base = self.USER_base
        self.base = self.USER_base
        base_coord = [self.pos_loc[i] for i in self.previous_USER_base]
        base_coord = np.reshape(np.array(base_coord),(self.no_of_users*2,))
        #self.observation = (self.MEC_location,self.mecability,self.mecservice, self.userability,self.usercoordinates, self.previous_USER_base)
        self.observation = (self.MEC_location,self.mecability,self.mecservice, self.userability,self.usercoordinates,base_coord)
        #print(colored(self.observation,"blue"))
        self.setup_graph()
        return self.observation



    def setup_graph(self):
        color_map=[]
        self.base_stations = np.reshape(self.base_stations,(self.no_of_stations,2))
        dist = cdist( self.base_stations, self.base_stations, metric="euclidean" )
        for i in range(self.no_of_stations): 
            self.G.add_node(i,**{'MEC':self.if_mec[i]})
            if self.if_mec[i] == "yes":
                color_map.append("green")
            else:
                color_map.append("red")
        for i in range(self.no_of_stations):
            for j in range(i+1, self.no_of_stations):
                self.G.add_edge(i, j, weight=dist[i][j])

        self.T = nx.minimum_spanning_tree(self.G)

        for i in range(self.no_of_stations):
            for j in range(self.no_of_stations):
                self.hop_to_hop[i][j] = len(nx.shortest_path(self.T, source=i, target=j))-1
        edges = list(self.T.edges(data=True)) 
        #nx.draw_networkx(self.T,self.pos_loc, node_color=color_map,with_labels=True, node_size = 500)
        self.hop_dist = defaultdict(list)
        for i in range(self.no_of_stations):
            for j in range(self.no_of_stations):
                x = nx.single_source_shortest_path_length(self.T, i, cutoff=j)
                x = [k for k,v in x.items() if v == j]
                self.hop_dist[tuple((i,j))] = x
        X = self.T.copy()
        x=50

        for i in range(len(self.usercoordinates)):
            if i%2 ==0:
                X.add_node(x,pos=(self.usercoordinates[i],self.usercoordinates[i+1]))
                color_map.append("blue")
                x += 1

        pos=nx.get_node_attributes(X,'pos')
        self.pos_loc.update(pos)
        #nx.draw_networkx(X,self.pos_loc, node_color=color_map,with_labels=True)

    def latency(self):
        return self.hop_to_hop,self.pos_mec


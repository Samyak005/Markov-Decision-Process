import numpy as np
import cvxpy as cp
import sys
import json
import os
import shutil
np.set_printoptions(threshold=sys.maxsize)

class LP:
    def __init__(self):
        self.numposState = 5
        self.numMaterials = 3
        self.numArrows = 4
        self.numMMstate = 2
        self.numhealthState = 5

        self.order_of_states = ["pos", "mat", "arrow", "state", "health"]

        self.actions = {
            "UP":0,
            "LEFT":1,
            "DOWN":2,
            "RIGHT":3,
            "STAY":4,
            "SHOOT":5,
            "HIT":6,
            "CRAFT":7,
            "GATHER":8,
            "NONE":9
        }

        self.convert_back = {
            0 : "UP",
            1 :"LEFT",
            2 :  "DOWN",
            3 : "RIGHT",
            4 :  "STAY",
            5 : "SHOOT",
            6 :   "HIT",
            7 : "CRAFT",
            8 :"GATHER",
            9 :  "NONE"
        }

        self.positions = {0:"C",
            1:"E",
            2:"N",
            3:"S",
            4:"W"}
 
        self.healthdescriptor = {0:0,
            1:25,
            2:50,
            3:75,
            4:100}

        self.statedescriptor = {0:'D',
                1:'R'}

        self.hash_state = {}
        self.reverse_hash_state = {}
        self.states = []
        self.A = [ [ 0 for i in range(1936) ] for j in range(600) ]
        self.reward = []
        self.possible_actions = {}
        self.penalty = -20.0 # step cost
        self.alpha = []
        self.total_actions = 0
        self.x = []
        self.objective = 0.0
        self.policy = []

        self.solve_lp()

    def solve_lp(self):
        self.initialize_states()
        self.initialize_possible_actions()
        self.get_transition_probabilities()
        self.initialize_Amatrix()
        self.initialize_reward()
        self.initialize_alpha()
        self.solve()
        self.get_policy()
        self.dump_to_file()

    def initialize_states(self):
        self.states = [(pos, mat, arrow, mmstate, health) for pos in range(self.numposState)
          for mat in range(self.numMaterials)
          for arrow in range(self.numArrows)
          for mmstate in range(self.numMMstate)
          for health in range(self.numhealthState)]
        # print(self.states)
        f= open("state_map.txt","w+")
        for i in range(600):
            self.hash_state[i] = tuple(self.states[i])
            self.reverse_hash_state[self.states[i]] = i
            f.write(str(i) + " " + str(self.states[i]) + '\n')
        f.close()

    def initialize_possible_actions(self):
        i = 0 
        f= open("action_map.txt","w+")

        for cur_state in self.states:
            pos = cur_state[0]
            mat = cur_state[1]
            arrow = cur_state[2]
            mmstate = cur_state[3]
            health = cur_state[4]
            cur_possible_actions = []

            if health == 0:
                cur_possible_actions.append(self.actions["NONE"])

            else:
                if pos==0:
                    cur_possible_actions.append(self.actions["UP"])
                    cur_possible_actions.append(self.actions["DOWN"])
                    cur_possible_actions.append(self.actions["LEFT"])
                    cur_possible_actions.append(self.actions["RIGHT"])
                    cur_possible_actions.append(self.actions["STAY"])
                    cur_possible_actions.append(self.actions["HIT"])
                    
                if pos==1:
                    cur_possible_actions.append(self.actions["LEFT"])
                    cur_possible_actions.append(self.actions["STAY"])
                    cur_possible_actions.append(self.actions["HIT"])

                if pos==2:
                    cur_possible_actions.append(self.actions["DOWN"])
                    cur_possible_actions.append(self.actions["STAY"])
                    if mat>=1:
                        cur_possible_actions.append(self.actions["CRAFT"])

                if pos==3:
                    cur_possible_actions.append(self.actions["UP"])
                    cur_possible_actions.append(self.actions["STAY"])
                    cur_possible_actions.append(self.actions["GATHER"])
                    
                if pos==4:
                    cur_possible_actions.append(self.actions["RIGHT"])
                    cur_possible_actions.append(self.actions["STAY"])

                if arrow > 0 and (pos in {0,1,4}):
                    cur_possible_actions.append(self.actions["SHOOT"])

            cur_possible_actions.sort()
            self.possible_actions[cur_state] = cur_possible_actions
            self.total_actions += len(cur_possible_actions)
            for j in range(len(cur_possible_actions)):
                f.write(str(i) + " " + str(self.states[i]) + " " + str(j) + " " + str(self.convert_back[cur_possible_actions[j]]) + '\n')
            f.write('\n')
            i+=1
        f.close()

    def get_transition_probabilities(self):
        self.transition_prob = {}

        P_MM_GO_READY = 0.2
        P_MM_STAY_DORMANT = 1 - P_MM_GO_READY
        P_MM_STAY_READY = 0.5 # same as attack mode
        P_MM_GO_DORMANT = 1 - P_MM_STAY_READY
        for pos, mat, arrow, mmstate, health in self.states:
            self.transition_prob[tuple([pos, mat, arrow, mmstate, health])] = {}

        for pos, mat, arrow, mmstate, health in self.states:
            state = tuple([pos, mat, arrow, mmstate, health])

            # print(pos, mat, arrow, mmstate, health)

            # <pos> : {C, E, N, S, W}   position of IJ
            # <action> : {UP, LEFT, DOWN, RIGHT,STAY, SHOOT, HIT, CRAFT, GATHER,NONE}

            # UP (and for current state is Center)

            next_state_center = tuple([0, mat, arrow, 0, health])
            next_state_north = tuple([2, mat, arrow, 0, health])
            next_state_west = tuple([4, mat, arrow, 0, health])
            next_state_south = tuple([3, mat, arrow, 0, health])
            next_state_east = tuple([1, mat, arrow, 0, health])

            # for SHOOT action, arrow count reduces, health may or may not come down
            if(arrow>=1):
                next_state_shoot_success = tuple([pos, mat, max(0,arrow-1), 0, max(0,health-1)])
                next_state_shoot_fail = tuple([pos, mat, max(0,arrow-1), 0, health])
            # for HIT action, health may reduce by 50, else the same state continues
            next_state_hit_success = tuple([pos, mat, arrow, 0, max(0, health-2)])
            # next_state_hit_success_tozero = tuple([pos, mat, arrow, 0, 0])
            # for GATHER action, materials may increase by 1 if it was zero else no change
            next_state_gather_success = tuple([pos, min(2, mat + 1), arrow, 0, health])

            # for CRAFT action, number of arrows increase by 1, 2 or 3 (capping of 3 to be implemented)
            next_state_arrow_plus0 = tuple([pos, max(0,mat-1), arrow, 0, health])
            next_state_arrow_plus1 = tuple([pos, max(0,mat-1), min(3,arrow+1), 0, health])
            next_state_arrow_plus2 = tuple([pos, max(0,mat-1), min(3,arrow+2), 0, health])
            next_state_arrow_plus3 = tuple([pos, max(0,mat-1), min(3,arrow+3), 0, health])

            # for MM Attack from Ready State -> MM goes to dormant state
            next_state_MM_attack_hit = tuple(  # hit only on center or east
                [pos, mat, 0, 0, min(4, health+1)])
            next_state_MM_attack_not_hit = tuple(  # not hit for other states
                [pos, mat, arrow, 0, health])
            # next_state_MM_attack_maxhealth = tuple([pos, mat, 0, 0, 4])

            # <pos> : {C, E, N, S, W}   position of IJ
            # all the above states are repeated with mmState to be READY instead of DORMANT
            mmnext_state_center = tuple([0, mat, arrow, 1, health])
            mmnext_state_north = tuple([2, mat, arrow, 1, health])
            mmnext_state_west = tuple([4, mat, arrow, 1, health])
            mmnext_state_south = tuple([3, mat, arrow, 1, health])
            mmnext_state_east = tuple([1, mat, arrow, 1, health])
            mmnext_state_shoot_success = tuple([pos, mat, max(0,arrow-1), 1, max(0,health-1)])
            mmnext_state_shoot_fail = tuple([pos, mat, max(0,arrow-1), 1, health])

            mmnext_state_hit_success = tuple(
                [pos, mat, arrow, 1, max(0, health-2)])
            # mmnext_state_hit_success_tozero = tuple([pos, mat, arrow, 0, 0])

            mmnext_state_gather_success = tuple([pos, min(2, mat + 1), arrow, 1, health])
            mmnext_state_arrow_plus0 = tuple([pos, max(0,mat-1), arrow, 1, health])
            mmnext_state_arrow_plus1 = tuple([pos, max(0,mat-1), min(3,arrow+1), 1, health])
            mmnext_state_arrow_plus2 = tuple([pos, max(0,mat-1), min(3,arrow+2), 1, health])
            mmnext_state_arrow_plus3 = tuple([pos, max(0,mat-1), min(3,arrow+3), 1, health])

            if (health == 0):
                continue

            # <pos> : {C, E, N, S, W}   position of IJ
            # for current position as CENTER
            if (pos == 0) and (mmstate == 0):
                # for movements to all 4 directions
                SUCCESS_PROB = 0.85
                FAIL_PROB = 1-SUCCESS_PROB
                self.transition_prob[state]["UP"] = {
                    next_state_north: SUCCESS_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_north: SUCCESS_PROB*P_MM_GO_READY,
                    next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_GO_READY
                }

                self.transition_prob[state]["DOWN"] = {
                    next_state_south: SUCCESS_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_south: SUCCESS_PROB*P_MM_GO_READY,
                    next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_GO_READY
                }

                self.transition_prob[state]["LEFT"] = {
                    next_state_west: SUCCESS_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_west: SUCCESS_PROB*P_MM_GO_READY,
                    next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_GO_READY
                }

                self.transition_prob[state]["RIGHT"] = {
                    next_state_east: P_MM_STAY_DORMANT,
                    mmnext_state_east: P_MM_GO_READY
                }
                self.transition_prob[state]["STAY"] = {
                    next_state_center: SUCCESS_PROB*P_MM_STAY_DORMANT,#
                    mmnext_state_center: SUCCESS_PROB*P_MM_GO_READY,
                    next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_GO_READY
                }

                if (arrow != 0):
                    SHOOT_SUCCESS_PROB = 0.5
                    SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB
                    self.transition_prob[state]["SHOOT"] = {
                        next_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_DORMANT,
                        next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_DORMANT,
                        mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_GO_READY,
                        mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_READY,
                    }
                # else:
                #     self.transition_prob[state]["SHOOT"] = {}

                HIT_SUCCESS = 0.1
                HIT_FAIL = 1-HIT_SUCCESS
                self.transition_prob[state]["HIT"] = {
                    next_state_center: HIT_FAIL*P_MM_STAY_DORMANT,#
                    mmnext_state_center: HIT_FAIL*P_MM_GO_READY,
                    next_state_hit_success: HIT_SUCCESS*P_MM_STAY_DORMANT,
                    mmnext_state_hit_success: HIT_SUCCESS*P_MM_GO_READY,
                }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = South
            if (pos == 3) and mmstate == 0:
                SUCCESS_PROB = 0.85
                FAIL_PROB = 1-SUCCESS_PROB
                GATHER_SUCCESS = 0.75
                GATHER_FAIL = 1-GATHER_SUCCESS
                self.transition_prob[state]["UP"] = {
                    next_state_center: SUCCESS_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_center: SUCCESS_PROB*P_MM_GO_READY,
                    next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_GO_READY,
                }
                self.transition_prob[state]["STAY"] = {
                    next_state_south: SUCCESS_PROB*P_MM_STAY_DORMANT,#
                    mmnext_state_south: SUCCESS_PROB*P_MM_GO_READY,
                    next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_GO_READY,
                }
                if mat < self.numMaterials-1:
                    self.transition_prob[state]["GATHER"] = {
                        next_state_gather_success: GATHER_SUCCESS*P_MM_STAY_DORMANT,
                        mmnext_state_gather_success: GATHER_SUCCESS * P_MM_GO_READY,
                        next_state_south: GATHER_FAIL*P_MM_STAY_DORMANT,#
                        mmnext_state_south: GATHER_FAIL*P_MM_GO_READY,
                    }
                elif(mat==2):
                    self.transition_prob[state]["GATHER"] = {
                        next_state_south: P_MM_STAY_DORMANT,#
                        mmnext_state_south: P_MM_GO_READY,
                    }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = NORTH
            if (pos == 2) and mmstate == 0:
                SUCCESS_PROB = 0.85
                FAIL_PROB = 1-SUCCESS_PROB

                self.transition_prob[state]["DOWN"] = {
                    next_state_center: SUCCESS_PROB * P_MM_STAY_DORMANT,
                    mmnext_state_center: SUCCESS_PROB * P_MM_GO_READY,
                    next_state_east: FAIL_PROB * P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB * P_MM_GO_READY
                }
                self.transition_prob[state]["STAY"] = {
                    next_state_north: SUCCESS_PROB * P_MM_STAY_DORMANT,#
                    mmnext_state_north: SUCCESS_PROB * P_MM_GO_READY,
                    next_state_east: FAIL_PROB * P_MM_STAY_DORMANT,
                    mmnext_state_east: FAIL_PROB * P_MM_GO_READY
                }

                CRAFT_0_ARROW = 0
                CRAFT_1_ARROW = 0.5
                CRAFT_2_ARROW = 0.35
                CRAFT_3_ARROW = 0.15
                CRAFT_1_ARROW_IF_2 = 1
                CRAFT_1_ARROW_IF_1 = 0.5
                CRAFT_2_ARROW_IF_1 = 0.5
                # TO FIX PROBABILITIES OF GAINING ARROWS.....
                #

                if(mat >= 1):
                    if (arrow == 3):
                        # no arrows get added as max arrows = 3
                        self.transition_prob[state]["CRAFT"] = {
                            next_state_arrow_plus0: P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus0: P_MM_GO_READY
                        }

                    if (arrow == 2):
                        # 1 arrow get added as max arrows = 3
                        self.transition_prob[state]["CRAFT"] = {
                            next_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_GO_READY
                        }

                    if (arrow == 1):
                        self.transition_prob[state]["CRAFT"] = {
                            next_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_GO_READY,
                            next_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_GO_READY
                        }

                    if (arrow == 0):
                        self.transition_prob[state]["CRAFT"] = {
                            next_state_arrow_plus3: CRAFT_3_ARROW * P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus3: CRAFT_3_ARROW * P_MM_GO_READY,
                            next_state_arrow_plus2: CRAFT_2_ARROW * P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus2: CRAFT_2_ARROW * P_MM_GO_READY,
                            next_state_arrow_plus1: CRAFT_1_ARROW * P_MM_STAY_DORMANT,
                            mmnext_state_arrow_plus1: CRAFT_1_ARROW * P_MM_GO_READY
                        }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = WEST
            # action successful always with prob 1
            if (pos == 4) and mmstate == 0:
                self.transition_prob[state]["RIGHT"] = {
                    next_state_center: P_MM_STAY_DORMANT,
                    mmnext_state_center: P_MM_GO_READY
                }

                self.transition_prob[state]["STAY"] = {
                    next_state_west: P_MM_STAY_DORMANT,#
                    mmnext_state_west: P_MM_GO_READY
                }
                SHOOT_SUCCESS_PROB = 0.25
                SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB

                if (arrow != 0):
                    self.transition_prob[state]["SHOOT"] = {
                        next_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_DORMANT,
                        mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_GO_READY,
                        next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_DORMANT,
                        mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_READY,
                    }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = EAST
            if (pos == 1) and (mmstate == 0):
                self.transition_prob[state]["LEFT"] = {
                    next_state_center: P_MM_STAY_DORMANT,
                    mmnext_state_center: P_MM_GO_READY
                }
                self.transition_prob[state]["STAY"] = {
                    next_state_east: P_MM_STAY_DORMANT,#
                    mmnext_state_east: P_MM_GO_READY,
                }

                SHOOT_SUCCESS_PROB = 0.9
                SHOOT_FAIL_PROB = 1-SHOOT_SUCCESS_PROB
                if (arrow != 0):
                    self.transition_prob[state]["SHOOT"] = {
                        next_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_DORMANT,
                        mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_GO_READY,
                        next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_DORMANT,
                        mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_READY

                    }

                HIT_SUCCESS = 0.2
                HIT_FAIL = 1-HIT_SUCCESS
                self.transition_prob[state]["HIT"] = {
                    next_state_hit_success: HIT_SUCCESS*P_MM_STAY_DORMANT,
                    mmnext_state_hit_success: HIT_SUCCESS*P_MM_GO_READY,
                    next_state_east: HIT_FAIL*P_MM_STAY_DORMANT,#
                    mmnext_state_east: HIT_FAIL*P_MM_GO_READY,
                }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = CENTER, MM in active state
            if (pos == 0) and (mmstate == 1):
                SUCCESS_PROB = 0.85
                FAIL_PROB = 1-SUCCESS_PROB
                self.transition_prob[state]["UP"] = {
                    mmnext_state_north: SUCCESS_PROB*P_MM_STAY_READY,
                    mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
                self.transition_prob[state]["DOWN"] = {
                    mmnext_state_south: SUCCESS_PROB*P_MM_STAY_READY,
                    mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }

                self.transition_prob[state]["LEFT"] = {
                    mmnext_state_west: SUCCESS_PROB*P_MM_STAY_READY,
                    mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }

                self.transition_prob[state]["RIGHT"] = {
                    mmnext_state_east: P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }

                self.transition_prob[state]["STAY"] = {
                    mmnext_state_center: SUCCESS_PROB*P_MM_STAY_READY,#
                    mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }

                if arrow > 0:
                    SHOOT_SUCCESS_PROB = 0.5
                    SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB
                    self.transition_prob[state]["SHOOT"] = {
                        mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_READY,
                        mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_READY,
                        next_state_MM_attack_hit: P_MM_GO_DORMANT,
                    }
                # else:
                #     self.transition_prob[state]["SHOOT"] = {
                #         mmnext_state_center: P_MM_STAY_READY,
                #         next_state_MM_attack_hit: P_MM_GO_DORMANT,
                #     }
                HIT_SUCCESS = 0.1
                HIT_FAIL = 1-HIT_SUCCESS
                self.transition_prob[state]["HIT"] = {
                    mmnext_state_hit_success: HIT_SUCCESS*P_MM_STAY_READY,
                    mmnext_state_center: HIT_FAIL*P_MM_STAY_READY,#
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = EAST, MM in active state
            if (pos == 1) and (mmstate == 1):
                self.transition_prob[state]["LEFT"] = {
                    mmnext_state_center: P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
                self.transition_prob[state]["STAY"] = {
                    mmnext_state_east: P_MM_STAY_READY,#
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
                SHOOT_SUCCESS_PROB = 0.9
                SHOOT_FAIL_PROB = 1-SHOOT_SUCCESS_PROB
                if arrow > 0:
                    self.transition_prob[state]["SHOOT"] = {
                        mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_READY,
                        mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_READY,
                        next_state_MM_attack_hit: P_MM_GO_DORMANT,
                    }
                # else:
                #     self.transition_prob[state]["SHOOT"] = {
                #         mmnext_state_east: P_MM_STAY_READY,
                #         next_state_MM_attack_hit: P_MM_GO_DORMANT,
                #     }
                HIT_SUCCESS = 0.2
                HIT_FAIL = 1-HIT_SUCCESS
                self.transition_prob[state]["HIT"] = {
                    mmnext_state_hit_success: HIT_SUCCESS*P_MM_STAY_READY,
                    mmnext_state_east: HIT_FAIL*P_MM_STAY_READY,#
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = WEST, MM in active state
            if (pos == 4) and (mmstate == 1):
                self.transition_prob[state]["RIGHT"] = {
                    mmnext_state_center: P_MM_STAY_READY,
                    next_state_center: P_MM_GO_DORMANT,
                    
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                }

                self.transition_prob[state]["STAY"] = {
                    mmnext_state_west: P_MM_STAY_READY,#

                    next_state_west: P_MM_GO_DORMANT,
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                }
                SHOOT_SUCCESS_PROB = 0.25
                SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB

                if arrow > 0:
                    self.transition_prob[state]["SHOOT"] = {
                        mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_READY,
                        mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_READY,
                        next_state_shoot_success: SHOOT_SUCCESS_PROB * P_MM_GO_DORMANT,
                        next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_DORMANT
                        # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    }
                # else:
                #     self.transition_prob[state]["SHOOT"] = {}

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = NORTH , MM in active state
            if (pos == 2) and mmstate == 1:
                SUCCESS_PROB = 0.85
                FAIL_PROB = 1-SUCCESS_PROB

                self.transition_prob[state]["DOWN"] = {
                    next_state_center: SUCCESS_PROB * P_MM_GO_DORMANT,
                    next_state_east: FAIL_PROB * P_MM_GO_DORMANT,
                    #next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    mmnext_state_center: SUCCESS_PROB * P_MM_STAY_READY,
                    mmnext_state_east: FAIL_PROB * P_MM_STAY_READY
                }
                self.transition_prob[state]["STAY"] = {
                    next_state_north: SUCCESS_PROB*P_MM_GO_DORMANT,
                    next_state_east: FAIL_PROB*P_MM_GO_DORMANT,
                    #next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    mmnext_state_north: SUCCESS_PROB * P_MM_STAY_READY,#
                    mmnext_state_east: FAIL_PROB * P_MM_STAY_READY
                }

                CRAFT_0_ARROW = 0
                CRAFT_1_ARROW = 0.5
                CRAFT_2_ARROW = 0.35
                CRAFT_3_ARROW = 0.15
                CRAFT_1_ARROW_IF_2 = 1
                CRAFT_1_ARROW_IF_1 = 0.5
                CRAFT_2_ARROW_IF_1 = 0.5
                
                if(mat >= 1):
                    if (arrow == 3):
                        # no arrows get added as max arrows = 3
                        self.transition_prob[state]["CRAFT"] = {
                            # next_state_arrow_plus0: CRAFT_0_ARROW * P_MM_STAY_DORMANT,
                            next_state_arrow_plus0: P_MM_GO_DORMANT,
                            mmnext_state_arrow_plus0: P_MM_STAY_READY
                        }

                    if (arrow == 2):
                        # 1 arrow get added as max arrows = 3
                        self.transition_prob[state]["CRAFT"] = {
                            # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                            next_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_GO_DORMANT,
                            mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_STAY_READY
                        }

                    if (arrow == 1):
                        self.transition_prob[state]["CRAFT"] = {
                            # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                            next_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_GO_DORMANT,
                            next_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_GO_DORMANT,
                            mmnext_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_STAY_READY,
                            mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_STAY_READY
                        }

                    if (arrow == 0):
                        self.transition_prob[state]["CRAFT"] = {
                            # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                            next_state_arrow_plus3: CRAFT_3_ARROW * P_MM_GO_DORMANT,
                            next_state_arrow_plus2: CRAFT_2_ARROW * P_MM_GO_DORMANT,
                            next_state_arrow_plus1: CRAFT_1_ARROW * P_MM_GO_DORMANT,
                            mmnext_state_arrow_plus3: CRAFT_3_ARROW * P_MM_STAY_READY,
                            mmnext_state_arrow_plus2: CRAFT_2_ARROW * P_MM_STAY_READY,
                            mmnext_state_arrow_plus1: CRAFT_1_ARROW * P_MM_STAY_READY
                        }

            # <pos> : {C, E, N, S, W}   position of IJ
            # Current position = South
            if (pos == 3) and mmstate == 1:
                SUCCESS_PROB = 0.85
                FAIL_PROB = 1-SUCCESS_PROB
                GATHER_SUCCESS = 0.75
                GATHER_FAIL = 1-GATHER_SUCCESS
                self.transition_prob[state]["UP"] = {
                    mmnext_state_center: SUCCESS_PROB*P_MM_STAY_READY,
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                    next_state_center: SUCCESS_PROB*P_MM_GO_DORMANT,
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    next_state_east: FAIL_PROB*P_MM_GO_DORMANT,
                }
                self.transition_prob[state]["STAY"] = {
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    mmnext_state_south: SUCCESS_PROB*P_MM_STAY_READY,#
                    mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,

                    next_state_south: SUCCESS_PROB*P_MM_GO_DORMANT,
                    next_state_east: FAIL_PROB*P_MM_GO_DORMANT,
                }
                if mat < self.numMaterials-1:
                    self.transition_prob[state]["GATHER"] = {
                        # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                        mmnext_state_gather_success: GATHER_SUCCESS*P_MM_STAY_READY,
                        mmnext_state_south: GATHER_FAIL*P_MM_STAY_READY,#

                        next_state_gather_success: GATHER_SUCCESS*P_MM_GO_DORMANT,
                        next_state_south: GATHER_FAIL*P_MM_GO_DORMANT,
                    }
                elif(mat==2):
                    self.transition_prob[state]["GATHER"] = {
                        # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                        mmnext_state_south: P_MM_STAY_READY,#
                        next_state_south: P_MM_GO_DORMANT,
                    }

        return self.transition_prob

    def initialize_Amatrix(self):
        # # just a check for VI algorithm to see in every dict sum of prob is 1 
        # for initial_state in self.states:
        #     for action in self.possible_actions[initial_state]:
        #         action = self.convert_back[action]
        #         if action=="NONE":
        #             continue
        #         # print(self.transition_prob[initial_state][action].values())
        #         if(round(sum(self.transition_prob[initial_state][action].values()),1)!=1):
        #             print("state: " + str(initial_state[0]) + str(initial_state[1]) + str(initial_state[2]) + str(initial_state[3]) +str(initial_state[4]))
        #             print(str(action))
        #             print(sum(self.transition_prob[initial_state][action].values()))
        column_index = 0
        for initial_state in self.states:
            for action in self.possible_actions[initial_state]:
                action = self.convert_back[action]
                # row_index = 0
                if action == "NONE":
                    self.A[self.reverse_hash_state[initial_state]][column_index] = 1
                    column_index += 1
                    continue

                for final_state in self.transition_prob[initial_state][action].keys():	
                    self.A[self.reverse_hash_state[final_state]][column_index] -= self.transition_prob[initial_state][action][final_state]
                    self.A[self.reverse_hash_state[initial_state]][column_index] += self.transition_prob[initial_state][action][final_state]
                    # row_index += 1
                column_index += 1

    def initialize_reward(self):
        # for cur_state in self.states:
        #     for action in self.transition_prob[cur_state].keys():
        #         # health = cur_state[4]
        #         expected_reward = 0
        #         for final_state in self.transition_prob[cur_state][action].keys():	
        #             if final_state[4] == 0:
        #                 expected_reward += 0
        #             elif(cur_state[3]==1 and final_state[3]==0 and ((cur_state[0]) in (0,1))):
        #                 expected_reward += self.transition_prob[cur_state][action][final_state] * (-40)
        #             else:
        #                 expected_reward += self.transition_prob[cur_state][action][final_state] * (self.penalty)

        #         self.reward.append(expected_reward)

        # for cur_state in self.states:
        #     for action in self.possible_actions[cur_state]:
        #         # health = cur_state[4]
        #         expected_reward = 0
        #         action_string = self.convert_back[action]
        #         if action_string=="NONE":
        #             pass
        #         else:
        #             for final_state in self.transition_prob[cur_state][action_string].keys():	
        #                 if final_state[4] == 0:
        #                     expected_reward += 0
        #                 elif(cur_state[3]==1 and final_state[3]==0 and ((cur_state[0]) in (0,1))):
        #                     expected_reward += self.transition_prob[cur_state][action_string][final_state] * (-60)
        #                 else:
        #                     expected_reward += self.transition_prob[cur_state][action_string][final_state] * (self.penalty)

        #         self.reward.append(expected_reward)
        #         print(expected_reward)

        for cur_state in self.states:
            for action in self.possible_actions[cur_state]:
                if cur_state[4] == 0:
                    self.reward.append(0.0)
                elif((cur_state[0] in (0,1)) and (cur_state[3]==1)):
                    self.reward.append(-40.0)
                else:
                    self.reward.append(self.penalty)
        self.reward = np.array(self.reward)
        self.reward = np.reshape(self.reward, (1, self.total_actions))

    def initialize_alpha(self):
        self.alpha = np.array([1.0 if state == (
            0, 2, 3, 1, 4) else 0.0 for state in self.states])

        self.alpha = np.expand_dims(self.alpha, axis=1)

    def solve(self):
        print(self.total_actions)
        x = cp.Variable((self.total_actions, 1), 'x')
        self.A = np.array(self.A)
        print(self.A.shape)
        print(x.shape)
        print(self.alpha.shape)
        print(self.reward.shape)
        f = open('reward_matrix.txt', 'w+')
        f.write(str(self.reward))
        f.close()
        # print(self.reward)
        constraints = [cp.matmul(self.A, x) == self.alpha, x >= 0]
        objective = cp.Maximize(cp.sum(cp.matmul(self.reward, x)))
        problem = cp.Problem(objective, constraints)

        self.objective = problem.solve()
        print(self.objective)
        # print(x.value)
        # arr = list(x.value)
        # l = [ float(val) for val in arr]
        # return l
        self.x = x.value.reshape(len(x.value))
        self.x = [round(num, 3) for num in self.x]
    def get_policy(self):

        index = 0
        i = 0 
        for state in self.states:
            index_having_max_value = np.argmax(
                self.x[index:index+len(self.possible_actions[state])])

            self.policy.append(
                [[self.positions[self.hash_state[i][0]], self.hash_state[i][1], self.hash_state[i][2], self.statedescriptor[self.hash_state[i][3]], self.healthdescriptor[self.hash_state[i][4]]], self.convert_back[self.possible_actions[state][index_having_max_value]]])

            index += len(self.possible_actions[state])
            i += 1
        self.policy = np.array(self.policy, dtype=object)

    def dump_to_file(self):

        output = {
            "a": self.A.tolist(),
            "r": self.reward.tolist(),
            "alpha": np.squeeze(self.alpha).tolist(),
            "x": self.x,
            "policy": self.policy.tolist(),
            "objective": self.objective
        }

        try:
            if os.path.exists('./outputs'):
                shutil.rmtree('./outputs')

            os.mkdir('./outputs')

        except OSError as error:
            print(error)
            sys.exit()

        with open("./outputs/part_3_output.json", "w") as fp:
            json.dump(output, fp)


if __name__ == "__main__":
    linearProgram = LP()
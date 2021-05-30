import numpy as np
import os

# <pos> : {C, E, N, S, W}   position of IJ

# <mat> : {0, 1, 2}   material with IJ
# <arrow> : {0, 1, 2, 3}  number of arrows
# <state> : {D, R}   ready and dormant state of MM
# <health> : {0,25,50,75,100}  MM’s health
# <action> : {UP, LEFT, DOWN, RIGHT,STAY, SHOOT, HIT, CRAFT, GATHER,NONE}

numposState = 5
numMaterials = 3
numArrows = 4
numMMstate = 2
numhealthState = 5
actions = {"RIGHT", "UP", "LEFT", "DOWN", "STAY",
           "SHOOT", "HIT", "CRAFT", "GATHER", "NONE"}

positions = {0:"C",
            1:"E",
            2:"N",
            3:"S",
            4:"W"}

# <health> : {0,25,50,75,100} 
healthdescriptor = {0:"0",
            1:"25",
            2:"50",
            3:"75",
            4:"100"}

statedescriptor = {0:"D",
                1:"R"}

gamma = 0.999
delta = 1e-3

P_MM_GO_READY = 0.2
P_MM_STAY_DORMANT = 1 - P_MM_GO_READY
P_MM_STAY_READY = 0.5 # same as attack mode
P_MM_GO_DORMANT = 1 - P_MM_STAY_READY

# step costs for position "W"
step_cost = {
    "UP": -20,
    "LEFT": -20,
    "DOWN": -20,
    "RIGHT": -20,
    "STAY": -20,
    "SHOOT": -20,
    "HIT": -20,
    "CRAFT": -20,
    "GATHER": -20,
    "NONE": -20
}

terminal_reward = 50
inf = 1e17

states = [(pos, mat, arrow, mmstate, health) for pos in range(numposState)
          for mat in range(numMaterials)
          for arrow in range(numArrows)
          for mmstate in range(numMMstate)
          for health in reversed(range(numhealthState))]

to_print = []
all_tasks = []


def get_states():
    all_state = {}
    for pos, mat, arrow, mmstate, health in states:
        all_state[tuple([pos, mat, arrow, mmstate, health])] = 0

    return all_state


def get_transition_probabilities(flag):
    transition_prob = {}

    for pos, mat, arrow, mmstate, health in states:
        transition_prob[tuple([pos, mat, arrow, mmstate, health])] = {}

    for pos, mat, arrow, mmstate, health in states:
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
            transition_prob[state]["UP"] = {
                next_state_north: SUCCESS_PROB*P_MM_STAY_DORMANT,
                mmnext_state_north: SUCCESS_PROB*P_MM_GO_READY,
                next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_GO_READY
            }

            transition_prob[state]["DOWN"] = {
                next_state_south: SUCCESS_PROB*P_MM_STAY_DORMANT,
                mmnext_state_south: SUCCESS_PROB*P_MM_GO_READY,
                next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_GO_READY
            }

            transition_prob[state]["LEFT"] = {
                next_state_west: SUCCESS_PROB*P_MM_STAY_DORMANT,
                mmnext_state_west: SUCCESS_PROB*P_MM_GO_READY,
                next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_GO_READY
            }

            transition_prob[state]["RIGHT"] = {
                next_state_east: P_MM_STAY_DORMANT,
                mmnext_state_east: P_MM_GO_READY
            }
            transition_prob[state]["STAY"] = {
                next_state_center: SUCCESS_PROB*P_MM_STAY_DORMANT,
                mmnext_state_center: SUCCESS_PROB*P_MM_GO_READY,
                next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_GO_READY
            }

            if (arrow != 0):
                SHOOT_SUCCESS_PROB = 0.5
                SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB
                transition_prob[state]["SHOOT"] = {
                    next_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_DORMANT,
                    next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_GO_READY,
                    mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_READY,
                }
            # else:
            #     transition_prob[state]["SHOOT"] = {}

            HIT_SUCCESS = 0.1
            HIT_FAIL = 1-HIT_SUCCESS
            transition_prob[state]["HIT"] = {
                next_state_center: HIT_FAIL*P_MM_STAY_DORMANT,
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
            transition_prob[state]["UP"] = {
                next_state_center: SUCCESS_PROB*P_MM_STAY_DORMANT,
                mmnext_state_center: SUCCESS_PROB*P_MM_GO_READY,
                next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_GO_READY,
            }
            transition_prob[state]["STAY"] = {
                next_state_south: SUCCESS_PROB*P_MM_STAY_DORMANT,
                mmnext_state_south: SUCCESS_PROB*P_MM_GO_READY,
                next_state_east: FAIL_PROB*P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_GO_READY,
            }
            if mat < numMaterials-1:
                transition_prob[state]["GATHER"] = {
                    next_state_gather_success: GATHER_SUCCESS*P_MM_STAY_DORMANT,
                    mmnext_state_gather_success: GATHER_SUCCESS * P_MM_GO_READY,
                    next_state_south: GATHER_FAIL*P_MM_STAY_DORMANT,
                    mmnext_state_south: GATHER_FAIL*P_MM_GO_READY,
                }
            elif(mat==2):
                transition_prob[state]["GATHER"] = {
                    next_state_south: P_MM_STAY_DORMANT,
                    mmnext_state_south: P_MM_GO_READY,
                }

        # <pos> : {C, E, N, S, W}   position of IJ
        # Current position = NORTH
        if (pos == 2) and mmstate == 0:
            SUCCESS_PROB = 0.85
            FAIL_PROB = 1-SUCCESS_PROB

            transition_prob[state]["DOWN"] = {
                next_state_center: SUCCESS_PROB * P_MM_STAY_DORMANT,
                mmnext_state_center: SUCCESS_PROB * P_MM_GO_READY,
                next_state_east: FAIL_PROB * P_MM_STAY_DORMANT,
                mmnext_state_east: FAIL_PROB * P_MM_GO_READY
            }
            transition_prob[state]["STAY"] = {
                next_state_north: SUCCESS_PROB * P_MM_STAY_DORMANT,
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
                    transition_prob[state]["CRAFT"] = {
                        next_state_arrow_plus0: P_MM_STAY_DORMANT,
                        mmnext_state_arrow_plus0: P_MM_GO_READY
                    }

                if (arrow == 2):
                    # 1 arrow get added as max arrows = 3
                    transition_prob[state]["CRAFT"] = {
                        next_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_STAY_DORMANT,
                        mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_GO_READY
                    }

                if (arrow == 1):
                    transition_prob[state]["CRAFT"] = {
                        next_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_STAY_DORMANT,
                        mmnext_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_GO_READY,
                        next_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_STAY_DORMANT,
                        mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_GO_READY
                    }

                if (arrow == 0):
                    transition_prob[state]["CRAFT"] = {
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
            transition_prob[state]["RIGHT"] = {
                next_state_center: P_MM_STAY_DORMANT,
                mmnext_state_center: P_MM_GO_READY
            }

            transition_prob[state]["STAY"] = {
                next_state_west: P_MM_STAY_DORMANT,
                mmnext_state_west: P_MM_GO_READY
            }
            SHOOT_SUCCESS_PROB = 0.25
            SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB

            if (arrow != 0):
                transition_prob[state]["SHOOT"] = {
                    next_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_GO_READY,
                    next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_READY,
                }

        # <pos> : {C, E, N, S, W}   position of IJ
        # Current position = EAST
        if (pos == 1) and (mmstate == 0):
            if(flag==0):
                transition_prob[state]["LEFT"] = {
                    next_state_center: P_MM_STAY_DORMANT,
                    mmnext_state_center: P_MM_GO_READY
                }
            else:
                transition_prob[state]["LEFT"] = {
                    next_state_west: P_MM_STAY_DORMANT,
                    mmnext_state_west: P_MM_GO_READY
                }
            transition_prob[state]["STAY"] = {
                next_state_east: P_MM_STAY_DORMANT,
                mmnext_state_east: P_MM_GO_READY,
            }

            SHOOT_SUCCESS_PROB = 0.9
            SHOOT_FAIL_PROB = 1-SHOOT_SUCCESS_PROB
            if (arrow != 0):
                transition_prob[state]["SHOOT"] = {
                    next_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_GO_READY,
                    next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_DORMANT,
                    mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_READY

                }

            HIT_SUCCESS = 0.2
            HIT_FAIL = 1-HIT_SUCCESS
            transition_prob[state]["HIT"] = {
                next_state_hit_success: HIT_SUCCESS*P_MM_STAY_DORMANT,
                mmnext_state_hit_success: HIT_SUCCESS*P_MM_GO_READY,
                next_state_east: HIT_FAIL*P_MM_STAY_DORMANT,
                mmnext_state_east: HIT_FAIL*P_MM_GO_READY,
            }

        # <pos> : {C, E, N, S, W}   position of IJ
        # Current position = CENTER, MM in active state
        if (pos == 0) and (mmstate == 1):
            SUCCESS_PROB = 0.85
            FAIL_PROB = 1-SUCCESS_PROB
            transition_prob[state]["UP"] = {
                mmnext_state_north: SUCCESS_PROB*P_MM_STAY_READY,
                mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }
            transition_prob[state]["DOWN"] = {
                mmnext_state_south: SUCCESS_PROB*P_MM_STAY_READY,
                mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }

            transition_prob[state]["LEFT"] = {
                mmnext_state_west: SUCCESS_PROB*P_MM_STAY_READY,
                mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }

            transition_prob[state]["RIGHT"] = {
                mmnext_state_east: P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }

            transition_prob[state]["STAY"] = {
                mmnext_state_center: SUCCESS_PROB*P_MM_STAY_READY,
                mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }

            if arrow > 0:
                SHOOT_SUCCESS_PROB = 0.5
                SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB
                transition_prob[state]["SHOOT"] = {
                    mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_READY,
                    mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
            # else:
            #     transition_prob[state]["SHOOT"] = {
            #         mmnext_state_center: P_MM_STAY_READY,
            #         next_state_MM_attack_hit: P_MM_GO_DORMANT,
            #     }
            HIT_SUCCESS = 0.1
            HIT_FAIL = 1-HIT_SUCCESS
            transition_prob[state]["HIT"] = {
                mmnext_state_hit_success: HIT_SUCCESS*P_MM_STAY_READY,
                mmnext_state_center: HIT_FAIL*P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }

        # <pos> : {C, E, N, S, W}   position of IJ
        # Current position = EAST, MM in active state
        if (pos == 1) and (mmstate == 1):
            if(flag==0):
                transition_prob[state]["LEFT"] = {
                    mmnext_state_center: P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
            else:
                transition_prob[state]["LEFT"] = {
                    mmnext_state_west: P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
            transition_prob[state]["STAY"] = {
                mmnext_state_east: P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }
            SHOOT_SUCCESS_PROB = 0.9
            SHOOT_FAIL_PROB = 1-SHOOT_SUCCESS_PROB
            if arrow > 0:
                transition_prob[state]["SHOOT"] = {
                    mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_READY,
                    mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_READY,
                    next_state_MM_attack_hit: P_MM_GO_DORMANT,
                }
            # else:
            #     transition_prob[state]["SHOOT"] = {
            #         mmnext_state_east: P_MM_STAY_READY,
            #         next_state_MM_attack_hit: P_MM_GO_DORMANT,
            #     }
            HIT_SUCCESS = 0.2
            HIT_FAIL = 1-HIT_SUCCESS
            transition_prob[state]["HIT"] = {
                mmnext_state_hit_success: HIT_SUCCESS*P_MM_STAY_READY,
                mmnext_state_east: HIT_FAIL*P_MM_STAY_READY,
                next_state_MM_attack_hit: P_MM_GO_DORMANT,
            }

        # <pos> : {C, E, N, S, W}   position of IJ
        # Current position = WEST, MM in active state
        if (pos == 4) and (mmstate == 1):
            transition_prob[state]["RIGHT"] = {
                mmnext_state_center: P_MM_STAY_READY,
                next_state_center: P_MM_GO_DORMANT,
                
                # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
            }

            transition_prob[state]["STAY"] = {
                mmnext_state_west: P_MM_STAY_READY,

                next_state_west: P_MM_GO_DORMANT,
                # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
            }
            SHOOT_SUCCESS_PROB = 0.25
            SHOOT_FAIL_PROB = 1 - SHOOT_SUCCESS_PROB

            if arrow > 0:
                transition_prob[state]["SHOOT"] = {
                    mmnext_state_shoot_success: SHOOT_SUCCESS_PROB*P_MM_STAY_READY,
                    mmnext_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_STAY_READY,
                    next_state_shoot_success: SHOOT_SUCCESS_PROB * P_MM_GO_DORMANT,
                    next_state_shoot_fail: SHOOT_FAIL_PROB*P_MM_GO_DORMANT
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                }
            # else:
            #     transition_prob[state]["SHOOT"] = {}

        # <pos> : {C, E, N, S, W}   position of IJ
        # Current position = NORTH , MM in active state
        if (pos == 2) and mmstate == 1:
            SUCCESS_PROB = 0.85
            FAIL_PROB = 1-SUCCESS_PROB

            transition_prob[state]["DOWN"] = {
                next_state_center: SUCCESS_PROB * P_MM_GO_DORMANT,
                next_state_east: FAIL_PROB * P_MM_GO_DORMANT,
                #next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                mmnext_state_center: SUCCESS_PROB * P_MM_STAY_READY,
                mmnext_state_east: FAIL_PROB * P_MM_STAY_READY
            }
            transition_prob[state]["STAY"] = {
                next_state_north: SUCCESS_PROB*P_MM_GO_DORMANT,
                next_state_east: FAIL_PROB*P_MM_GO_DORMANT,
                #next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                mmnext_state_north: SUCCESS_PROB * P_MM_STAY_READY,
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
                    transition_prob[state]["CRAFT"] = {
                        # next_state_arrow_plus0: CRAFT_0_ARROW * P_MM_STAY_DORMANT,
                        next_state_arrow_plus0: P_MM_GO_DORMANT,
                        mmnext_state_arrow_plus0: P_MM_STAY_READY
                    }

                if (arrow == 2):
                    # 1 arrow get added as max arrows = 3
                    transition_prob[state]["CRAFT"] = {
                        # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                        next_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_GO_DORMANT,
                        mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_2 * P_MM_STAY_READY
                    }

                if (arrow == 1):
                    transition_prob[state]["CRAFT"] = {
                        # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                        next_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_GO_DORMANT,
                        next_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_GO_DORMANT,
                        mmnext_state_arrow_plus2: CRAFT_2_ARROW_IF_1 * P_MM_STAY_READY,
                        mmnext_state_arrow_plus1: CRAFT_1_ARROW_IF_1 * P_MM_STAY_READY
                    }

                if (arrow == 0):
                    transition_prob[state]["CRAFT"] = {
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
            transition_prob[state]["UP"] = {
                mmnext_state_center: SUCCESS_PROB*P_MM_STAY_READY,
                # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,
                next_state_center: SUCCESS_PROB*P_MM_GO_DORMANT,
                # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                next_state_east: FAIL_PROB*P_MM_GO_DORMANT,
            }
            transition_prob[state]["STAY"] = {
                # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                mmnext_state_south: SUCCESS_PROB*P_MM_STAY_READY,
                mmnext_state_east: FAIL_PROB*P_MM_STAY_READY,

                next_state_south: SUCCESS_PROB*P_MM_GO_DORMANT,
                next_state_east: FAIL_PROB*P_MM_GO_DORMANT,
            }
            if mat < numMaterials-1:
                transition_prob[state]["GATHER"] = {
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    mmnext_state_gather_success: GATHER_SUCCESS*P_MM_STAY_READY,
                    mmnext_state_south: GATHER_FAIL*P_MM_STAY_READY,

                    next_state_gather_success: GATHER_SUCCESS*P_MM_GO_DORMANT,
                    next_state_south: GATHER_FAIL*P_MM_GO_DORMANT,
                }
            elif(mat==2):
                transition_prob[state]["GATHER"] = {
                    # next_state_MM_attack_not_hit: P_MM_GO_DORMANT,
                    mmnext_state_south: P_MM_STAY_READY,
                    next_state_south: P_MM_GO_DORMANT,
                }

    return transition_prob


transition_prob = get_transition_probabilities(0)


def get_utilities(utilities):
    new_utilities = np.zeros(
        shape=(numposState, numMaterials, numArrows, numMMstate, numhealthState))

    for pos, mat, arrow, mmstate, health in states:

        cur_state = tuple([pos, mat, arrow, mmstate, health])

        if health == 0:
            continue

        cur_max = -inf

        for action in transition_prob[cur_state].keys():	
            if action == "SHOOT":	
                if arrow == 0:	
                    1/0	
            elif action == "CRAFT":	
                if mat == 0:
                    1/0
            # elif action == "GATHER":
            #     if mat >= numMaterials-1:
            #         continue
            elif action == "NONE":
                continue

            total_reward = 0	
            cur = 0	
            for p, m, a, s, h in transition_prob[cur_state][action].keys():	
                new_state = tuple([p, m, a, s, h])	
                if h == 0:	
                    # try:	
                    total_reward += (step_cost[action] + terminal_reward) * \
                        transition_prob[cur_state][action][new_state]	
                    # except:	
                    #     print(cur_state, action, new_state)

                # <pos> : {C, E, N, S, W}   position of IJ
                elif((s==0) and (mmstate==1) and ((pos==0) or (pos==1))):
                    total_reward += (step_cost[action]) * \
                        transition_prob[cur_state][action][new_state] 
                    total_reward += (-40) * transition_prob[cur_state][action][new_state]
                else:
                    total_reward += (step_cost[action]) * \
                        transition_prob[cur_state][action][new_state]

                cur += gamma * \
                    transition_prob[cur_state][action][new_state] * \
                    utilities[p, m, a, s, h]

            cur += total_reward

            if cur_max < cur:
                cur_max = cur

        new_utilities[pos, mat, arrow, mmstate, health] = cur_max

    return new_utilities


def get_action(new_utilities):
    for pos, mat, arrow, mmstate, health in states:
        # print("Initial state: " + str(pos) + str(mat) + str(arrow) + str(mmstate) + str(health))
        cur_state = tuple([pos, mat, arrow, mmstate, health])
        cur_action = "NONE"
        if health == 0:
            to_print.append(
                f"({positions[pos]},{mat},{arrow},{statedescriptor[mmstate]},{healthdescriptor[health]}):{cur_action}=[{round(new_utilities[pos, mat, arrow, mmstate, health], 3)}]")
            continue

        cur_max = -inf
        # cur_action = ""

        for action in transition_prob[cur_state].keys():	
            # print("action: " + str(action))
            if action == "SHOOT":	
                if arrow == 0:	
                    1/0	
            elif action == "CRAFT":	
                if mat == 0:
                    1/0
            # elif action == "GATHER":
            #     if mat >= numMaterials-1:
            #         continue
            elif action == "NONE":
                continue

            total_reward = 0	
            cur = 0	
            for p, m, a, s, h in transition_prob[cur_state][action].keys():	
                # print("utility: " + str(new_utilities[p,m,a,s,h]))
                # print("final state: " + str(p) + str(m) + str(a) + str(s) + str(h))
                new_state = tuple([p, m, a, s, h])	
                if h == 0:	
                    # print("health 0")
                    # try:	
                    total_reward += (step_cost[action]+terminal_reward) * \
                        transition_prob[cur_state][action][new_state]
                    # print("added in reward terminal: " + str((terminal_reward) * \
                    #     transition_prob[cur_state][action][new_state]))	
                    # except:	
                    #     print(cur_state, action, new_state)
                        
                # <pos> : {C, E, N, S, W}   position of IJ                        
                elif((s==0) and (mmstate==1) and ((pos==0) or (pos==1))):
                    total_reward += (step_cost[action]) * \
                        transition_prob[cur_state][action][new_state] 
                    total_reward += (-40) * transition_prob[cur_state][action][new_state]
                else:
                    total_reward += (step_cost[action]) * \
                        transition_prob[cur_state][action][new_state]
                    # print("added in step cost reward" + str((step_cost[action]) * \
                    #     transition_prob[cur_state][action][new_state]))

                cur += gamma * \
                    transition_prob[cur_state][action][new_state] * \
                    new_utilities[p, m, a, s, h]
                # print("added in gamma: " + str(gamma * \
                #     transition_prob[cur_state][action][new_state] * \
                #     new_utilities[p, m, a, s, h]))
            cur += total_reward
            if cur_max <= cur:
                cur_max = cur
                cur_action = action

        to_print.append(
            f"({positions[pos]},{mat},{arrow},{statedescriptor[mmstate]},{healthdescriptor[health]}):{cur_action}=[{round(new_utilities[pos, mat, arrow, mmstate, health], 3)}]")


def hasConverged(utilities, new_utilities):
    return np.max(np.abs(new_utilities - utilities)) < delta


def value_iteration():

    utilities = np.zeros(shape=(numposState, numMaterials,
                                numArrows, numMMstate, numhealthState))

    iterations = 0

    while True:
        to_print.append(f"iteration={iterations}")

        new_utilities = get_utilities(utilities)
        
        get_action(new_utilities)

        converged = hasConverged(utilities, new_utilities)

        if converged:
            break

        utilities = new_utilities
        iterations += 1

        # to_print.append("\n")


for i in range(4):
    # if i==0:
    #     get_transition_probabilities(0)
    if i == 1:
        transition_prob = get_transition_probabilities(1)
        # print(transition_prob[(1,0,0,0,4)]["LEFT"])
    elif i == 2:
        transition_prob = get_transition_probabilities(0)
        step_cost["STAY"] = 0

    elif i == 3:
        step_cost["STAY"] = -20
        gamma = 0.25

    value_iteration()

    all_tasks.append(to_print)
    to_print = []

os.makedirs(os.path.dirname("./outputs/"), exist_ok=True)
with open('./outputs/part_2_trace.txt', 'w+') as f:
    f.writelines("%s\n" % line for line in all_tasks[0])

with open('./outputs/part_2_task_2.1_trace.txt', 'w+') as f:
    f.writelines("%s\n" % line for line in all_tasks[1])

with open('./outputs/part_2_task_2.2_trace.txt', 'w+') as f:
    f.writelines("%s\n" % line for line in all_tasks[2])

with open('./outputs/part_2_task_2.3_trace.txt', 'w+') as f:
    f.writelines("%s\n" % line for line in all_tasks[3])

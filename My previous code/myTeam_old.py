# myTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import heapq
from ast import Raise
from typing import List, Tuple

from numpy import true_divide
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys,os
from game import Grid

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
                             first = 'MixedAgent', second = 'MixedAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########                                       

class MixedAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """
    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
            "offensiveWeights":{'closest-food': -1, 
                                        'bias': 1, 
                                        '#-of-ghosts-1-step-away': -100, 
                                        'successorScore': 100, 
                                        'chance-return-food': 10,
                                        'stop': -10,
                                        'carry-risk': -100, 
                                        },
            "defensiveWeights": {
                                        # 'teamDistance': 2, 
                                        'invaderDistance': -10, 
                                        'stop': -200,  # Increase the negative weight
                                        'reverse': -20, 
                                        'chaseInvader': 5,
                                        # 'patrolDistance': 100,
            },
            "escapeWeights": {'onDefense': 40, 
                                        'enemyDistance': 30, 
                                        'stop': -100, 
                                        'carrying': 10,
                                        'distanceToHome': -20,
            },
            "avoidEnemyWeights": {'closestEnemy': -500,'#-of-enemies-nearby': -10,'stop': -200}

        }
    QLWeightsFile = BASE_FOLDER+'/QLWeightsMyTeam.txt'

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}
    LAST_ACTION = {}


    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action,pddl_state]] = None # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0 # index of action in self.highLevelPlan should be execute next

        self.startPosition = gameState.getAgentPosition(self.index) # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str,Tuple]] = []
        self.lowLevelActionIndex = 0

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = True # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.01 #default exploration prob, change to take a random step
        self.initial_alpha = 1.0
        self.alpha =  0.05  #default learning rate
        self.discountRate = 0.9 # default discount rate on successor state q value when update
        
        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index]={}
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        if os.path.exists(MixedAgent.QLWeightsFile):
            with open(MixedAgent.QLWeightsFile, "r") as file:
                MixedAgent.QLWeights = eval(file.read())
            print("Load QLWeights:",MixedAgent.QLWeights )
        
    
    def final(self, gameState : GameState):
        """
        This function write weights into files after the game is over. 
        You may want to comment (disallow) this function when submit to contest server.
        """
        print("Write QLWeights:", MixedAgent.QLWeights)
        file = open(MixedAgent.QLWeightsFile, 'w')
        file.write(str(MixedAgent.QLWeights))
        file.close()
    

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        #-------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        highLevelAction = 'attack'
        positiveGoal, negtiveGoal = self.getGoals(objects,initState)
        legalActions = gameState.getLegalActions(self.index)

        # Check if we can stick to current plan 
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:",self.index,"compute plan:")
            print("\tOBJ:"+str(objects),"\tINIT:"+str(initState), "\tPOSITIVE_GOAL:"+str(positiveGoal), "\tNEGTIVE_GOAL:"+str(negtiveGoal),sep="\n")
            self.highLevelPlan: List[Tuple[Action,pddl_state]] = self.getHighLevelPlan(objects, initState,positiveGoal, negtiveGoal) # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = [] # reset low level plan
            print("\tPLAN:",self.highLevelPlan)
        if len(self.highLevelPlan)==0:
            # raise Exception("Solver retuned empty plan, you need to think how you handle this situation or how you modify your model ")
            highLevelAction = 'defence'
        # if len(MixedAgent.LAST_ACTION) > 0:
        #     # Get the action of the other agent
        #     other_agent = [index for index in MixedAgent.LAST_ACTION.keys() if index != self.index][0]
        #     other_action = MixedAgent.LAST_ACTION[other_agent]

        #     # If the other agent is also planning to perform the same action, change action
        #     if other_action == highLevelAction:
        #         if highLevelAction == 'attack':
        #             highLevelAction = 'defence'
        #         else:
        #             highLevelAction = 'patrol'
        else:
            # Get next action from the plan
            highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.LAST_ACTION[self.index] = highLevelAction
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        print("Agent:", self.index, highLevelAction)

        #-------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState) and self.lowLevelActionIndex >= len(self.lowLevelPlan):
            self.lowLevelPlan = self.getLowLevelPlanQL(gameState, highLevelAction) #Generate low level plan with q learning
            # self.lowLevelPlan = self.getLowLevelPlanHS(gameState, highLevelAction)
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            # Debugging: print the low-level plan and action index)
            self.lowLevelActionIndex = 0
        # print("Low-level plan:", self.lowLevelPlan)
        # print("Low-level action index:", self.lowLevelActionIndex)
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        if lowLevelAction not in legalActions:
        # You can choose a random action from the list of legal actions
        # Or you can implement a more sophisticated method to choose a different action
            lowLevelAction = random.choice(legalActions)
        self.lowLevelActionIndex+=1
        print("\tAgent:", self.index,lowLevelAction)
        return lowLevelAction

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)
        
        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    def get_pddl_state(self,gameState:GameState) -> Tuple[List[Tuple],List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []


        # Collect available foods on the map
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(myPos,self.getFood(gameState), gameState.getWalls())
        if cloestFoodDist != None and cloestFoodDist <=CLOSE_DISTANCE:
            states.append(("near_food",myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 :
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap,myPos) <=CLOSE_DISTANCE:
                states.append(("near_capsule",myObj))
                break
        
        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore> 3:
                states.append(("winning_gt3",))
            if currentScore> 5:
                states.append(("winning_gt5",))
            if currentScore> 10:
                states.append(("winning_gt10",))
            if currentScore> 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))

        # Collect team agents states
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if agent_index != self.index and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agent_index)) <= CLOSE_DISTANCE:
                states.append(("near_ally",))
            
            if agent_state.scaredTimer>0:
                states.append(("is_scared",agent_object))

            if agent_state.numCarrying>0:
                states.append(("food_in_backpack",agent_object))
                if agent_state.numCarrying >=20 :
                    states.append(("20_food_in_backpack",agent_object))
                if agent_state.numCarrying >=10 :
                    states.append(("10_food_in_backpack",agent_object))
                if agent_state.numCarrying >=5 :
                    states.append(("5_food_in_backpack",agent_object))
                if agent_state.numCarrying >=3 :
                    states.append(("3_food_in_backpack",agent_object))
                
            if agent_state.isPacman:
                states.append(("is_pacman",agent_object))
            
            

        # Collect enemy agents states
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer>0:
                states.append(("is_scared",enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if self.getMazeDistance(agent_state.getPosition(), enemy_position) <= CLOSE_DISTANCE:
                        states.append(("enemy_around",enemy_object, "a{}".format(agent_index)))
            else:
                if noisyDistance[enemy_index] >=LONG_DISTANCE :
                    states.append(("enemy_long_distance",enemy_object, "a{}".format(self.index)))
                elif noisyDistance[enemy_index] >=MEDIUM_DISTANCE :
                    states.append(("enemy_medium_distance",enemy_object, "a{}".format(self.index)))
                else:
                    states.append(("enemy_short_distance",enemy_object, "a{}".format(self.index)))                                                                                                                                                                                                 


            if enemy_state.isPacman:
                states.append(("is_pacman",enemy_object))
            typeIndex += 1
            
        return objects, states
    
    def stateSatisfyCurrentPlan(self, init_state: List[Tuple],positiveGoal, negtiveGoal):
        if self.highLevelPlan is None:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False
        
        if positiveGoal != self.currentPositiveGoalStates or negtiveGoal != self.currentNegativeGoalStates:
            return False
        
        if self.currentActionIndex < len(self.highLevelPlan) and self.pddl_solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0] ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(self.highLevelPlan) -1 and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = [] # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.currentActionIndex < len(self.highLevelPlan) and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Current action precondition satisfied, continue executing current action of the plan
            return True
        
        # Current action precondition not satisfied anymore, need new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        if (("winning_gt10",) in initState):
            return self.goalDefWinning(objects, initState)
        if any("enemy_around" in state for state in initState):
            return self.enemyAround(objects, initState)
        else:
            return self.goalScoring(objects, initState)

    def enemyAround(self,objects: List[Tuple], initState: List[Tuple]):
        positiveGoal = []
        noaviodAgent = []
        negtiveGoal = [("food_available",)] 
        pacman_enemy = None

        # First loop
        for state in initState:
            if "is_pacman" in state:
                if state[1] == 'e1' or state[1] == 'e3':
                    negtiveGoal += [state]
                    noaviodAgent = state[1]

        # Second loop
        for state in initState:
            if "enemy_around" in state:
                if state[1] == noaviodAgent or state[2] == 'a2':
                    pass
                else:
                    negtiveGoal += [state]
                    
        for state in negtiveGoal:
            if state[0] == "is_pacman":
                pacman_enemy = state[1]
                break
        if pacman_enemy is not None:
            negtiveGoal = [state for state in negtiveGoal if not (state[0] == 'enemy_around' and state[1] == pacman_enemy)]
        else:
            pass

        if self.index == 2:
            negativeGoal = []
            positiveGoal = [("is_agent2")]

        return positiveGoal, negtiveGoal
        # for state in initState:
        #     if "is_pacman" in state:
        #         if state[1] == 'e1' or state[1] == 'e3':
        #             negtiveGoal += [state]
        #             noaviodAgent = state[1]
        #     if "enemy_around" in state:
        #         # negtiveGoal += [noaviodAgent]
        #         if noaviodAgent == state[1]:
        #             pass
        #         else:
        #             negtiveGoal += [state]
        # return positiveGoal, negtiveGoal

    def goalScoring(self,objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.

        positiveGoal = []
        negtiveGoal = [("food_available",)] # no food avaliable means eat all the food
        
        if self.index == 2:
            negativeGoal = []
            positiveGoal = [("is_agent2")]
        
        if (("near_capsule",) in initState):
            negativeGoal = []
            positiveGoal = [("eat_capsule", "a{}".format(self.index))]
        # for state in initState:
        #     if "enemy_around" in state:
        #         negtiveGoal += [state]

        # for state in initState:
        #     if "is_pacman" in state:
        #         if state[1] == 'a2':
        #             negativeGoal = []
        #             positiveGoal = [("defend_foods",)]


        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.
        
        return positiveGoal, negtiveGoal

    def goalDefWinning(self,objects: List[Tuple], initState: List[Tuple]):
        # If winning greater than 5 points,
        # this example want defend foods only, and let agents patrol on our ground.
        # The "win_the_game" pddl state is only reachable by the "patrol" action in pddl,
        # using it as goal, pddl will generate plan eliminate invading enemy and patrol on our ground.

        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        
        return positiveGoal, negtiveGoal

    #------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(self, gameState: GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.

        # map = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        # foods = self.getFood(gameState) # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        # capsules = self.getCapsules(gameState) # a list of capsules
        # foodNeedDefend = self.getFoodYouAreDefending(gameState) # return food will be eatan by enemy (food next to enemy)
        # capsuleNeedDefend = self.getCapsulesYouAreDefending(gameState) # return capsule will be eatan by enemy (capsule next to enemy)
        # Raise(NotImplementedError("Heuristic Search low level "))
        # return [] # You should return a list of tuple of move action and target location (exclude current location).

        # A* Search
        def aStarSearch(start, goals, gameState:GameState):
            # The open list
            open = [(start, [], 0)]
            # The closed list
            closed = []
            while open:
                # Pop the node with the smallest f value off the open list
                node, actions, g = heapq.heappop(open)
                if node in closed:
                    continue
                closed.append(node)
                for goal in goals:
                    if node == goal:
                        return actions
                for pos in gameState.getLegalNeighbors((int(node[0]), int(node[1]))):
                    if pos not in closed:
                        cost = 1  # cost per action
                        h = min(self.getMazeDistance(goal, pos) for goal in goals)  # heuristic
                        f = g + cost + h
                        # Calculate the direction to the neighbor
                        dir = Actions.vectorToDirection((pos[0] - node[0], pos[1] - node[1]))
                        heapq.heappush(open, (pos, actions + [dir], g + cost))



        # Implement the function
        myPos = gameState.getAgentState(self.index).getPosition()

# # debug
#         fd = self.getFoodYouAreDefending(gameState)
#         fdlist = fd.asList()
#         if len(fdlist) > 0:
#             distanc = min([self.getMazeDistance(myPos, food) for food in fdlist])
#             gg = [food for food in fdlist if self.getMazeDistance(myPos, food) == distanc]
#         print("foodlist:======",gg)

        if highLevelAction == 'attack':
            # Get list of food as potential goals
            goals = self.getFood(gameState).asList()
        else:  # 'defence'
            # Get list of invaders as potential goals
            invaders = [gameState.getAgentState(i).getPosition() for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() is not None]
            # enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            # invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
            if len(invaders) > 0:
                goals = invaders
                print("this is invaders>0",goals)
            else:
                # If no invaders are visible, patrol near the food closest to the center
                food = self.getFoodYouAreDefending(gameState)
                foodList = food.asList()
                if len(foodList) > 0:
                    minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
                    goals = [food for food in foodList if self.getMazeDistance(myPos, food) == minDistance]
                    # print("this is foodlist>0",goals)
                    # print("This is my loc:=======",myPos)
                    if myPos in goals:
                        goals.remove(myPos)
                        if len(foodList) > 1:
                            foodList.remove(myPos)
                            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
                            goals = [food for food in foodList if self.getMazeDistance(myPos, food) == minDistance]  
                else:
                    goals = [self.startPosition]
                    # print("this is don't have fooslist",goals)
        # if carrying food more than 5, go back to the starting position
        if gameState.getAgentState(self.index).numCarrying > 5:
            goals = [self.startPosition]
        # Use A* to get low level plan to the nearest goal
        actions = aStarSearch(myPos, goals, gameState)
        # Return a list of tuples of move action and target location
        return [(action, gameState.getAgentState(self.index).getPosition() + Actions.directionToVector(action)) for action in actions]
    
    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    #------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getLowLevelPlanQL(self, gameState:GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = self.alpha

        ##########
        # The following classification of high level actions is only a example.
        # You should think and use your own way to design low level planner.
        ##########
        if highLevelAction == "attack":
            # The q learning process for offensive actions are complete, 
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            self.alpha = self.alpha 
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            self.alpha = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        elif highLevelAction == "avoid_enemy":
            # The q learning process for avoiding enemies
            rewardFunction = self.getAvoidEnemyReward
            featureFunction = self.getAvoidEnemyFeatures
            weights = self.getAvoidEnemyWeights()
            self.alpha = self.alpha
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            self.alpha = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update 

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon) # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                        if self.trainning:
                            self.updateWeights(gameState, action, rewardFunction, featureFunction, weights,learningRate)
                        values.append((self.getQValue(featureFunction(gameState, action), weights), action))
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,action)
        print(values)
        return [(action, nextPos)]


    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, features, weights):
        return features * weights
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState:GameState, action, rewardFunction, featureFunction, weights, learningRate):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState, featureFunction, weights)) - self.getQValue(features, weights)
            weights[feature] =weights[feature] + learningRate*correction * features[feature]
        # Check if the agent is cornered
        myState = nextState.getAgentState(self.index)
        myPos = myState.getPosition()
        enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            if min(dists) <= 1:  # If an invader is adjacent
                if feature == 'stop' or feature == 'reverse':
                    weights[feature] = weights[feature] + learningRate * correction * features[feature] * -1
                else:
                    weights[feature] = weights[feature] + learningRate * correction * features[feature]
            else:
                weights[feature] = weights[feature] + learningRate * correction * features[feature]
    
    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features,weights))
            return max(qVals)
    
    
    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward. 
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)
        print("this is :",currentAgentState.getDirection())
        # invader_noisy_distance = [currentAgentState.getAgentDistances() for i in self.getOpponents(gameState) ]
        # print(invader_noisy_distance)
        

        ghosts = self.getGhostLocs(gameState)
        nextghosts = self.getGhostLocs(nextState)
        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g,gameState.getWalls()) for g in ghosts)
        print("gost gosr gost",ghosts)
        print("nextgost nextgost gost",nextghosts)
        base_reward =  -10 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 5
        if score <0:
            base_reward += score 
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned * 2
        # gameState.getLegalActions(self.index)
        # Calculate distances to the ghosts in the current and next states
        if len(ghosts) > 0:
            currentMinDistToGhost = min([self.getMazeDistance(currentAgentState.getPosition(), g) for g in ghosts])
            nextMinDistToGhost = min([self.getMazeDistance(nextAgentState.getPosition(), g) for g in nextghosts])
        # If the agent is moving away from a ghost, give a reward
            if nextMinDistToGhost > currentMinDistToGhost:
                base_reward += 2
            else:
                base_reward -= 1
        # , "  currentDS:",currentMinDistToGhost,",nextDS: ",nextMinDistToGhost

        print(ghosts)
        print(gameState.getAgentDistances())

        print("Agent ", self.index," reward ",base_reward, gameState.getLegalActions(self.index))
        return base_reward
    
    def getDefensiveReward(self,gameState: GameState, nextState: GameState):
        def aStarSearch(self, gameState, goal):
            """Returns a list of actions that leads to the `goal` position"""
            start = gameState.getAgentState(self.index).getPosition()
            actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
            frontier = util.PriorityQueue()
            frontier.push((start, []), 0)
            explored = set()

            while not frontier.isEmpty():
                node, path = frontier.pop()

                if node in explored:
                    continue

                explored.add(node)

                if node == goal:
                    return path

                for action in actions:
                    dx, dy = Actions.directionToVector(action)
                    nextx, nexty = int(node[0] + dx), int(node[1] + dy)
                    if not gameState.hasWall(nextx, nexty):
                        nextNode = (nextx, nexty)
                        cost = len(path) + 1
                        frontier.push((nextNode, path + [action]), cost)

            return []  # No path found
        # Current and next agent states
        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)

        
        # invader_index = [i for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman]
        # invader_noisy_distance = [currentAgentState.getAgentDistances(i) for i in invader_index]
        # Current and next invaders
        currentInvaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() is not None]
        nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman and nextState.getAgentState(i).getPosition() is not None]
        print("currentinvader",len(currentInvaders))
        reward = 0

        # If there are no invaders, encourage patrolling
        defensiveHalfGrid = gameState.getRedFood() # Assuming `self.red` tells if you're on the red team
        patrolPoints = [(x, y) for x in range(defensiveHalfGrid.width) for y in range(defensiveHalfGrid.height) if defensiveHalfGrid[x][y]]
        patrolPointsQueue = []
        
            
        # Refill the patrol points queue if it's empty
        if len(patrolPointsQueue) == 0:
            patrolPointsQueue = patrolPoints.copy()

        if len(currentInvaders) == 0:
            # Get the next patrol point
            patrolPoint = patrolPointsQueue[0]

            # Check if the agent has reached the patrol point
            patrolDistances = len(aStarSearch(self,nextState, patrolPoint))
            if patrolDistances < 3:
                # Remove this patrol point from the queue
                patrolPointsQueue.pop(0)

                # Get the next patrol point
                if len(patrolPointsQueue) > 0:
                    patrolPoint = patrolPointsQueue[0]
                    patrolDistances = len(aStarSearch(self,gameState, patrolPoint))
                else:
                    patrolPoint = None

            if patrolPoint is not None:
                # closestPatrolPoint = min(patrolPoints, key = lambda x: self.getMazeDistance(currentAgentState.getPosition(), x))
                # print(patrolPoint)
                # currentDistToPatrolPoint = self.getMazeDistance(currentAgentState.getPosition(), patrolPoint)
                patrolDistances = len(aStarSearch(self,gameState, patrolPoint))
                # nextDistToPatrolPoint = self.getMazeDistance(nextAgentState.getPosition(), patrolPoint)
                nextpatrodistance = len(aStarSearch(self,nextState, patrolPoint))
                print("curren agent location:",currentAgentState.getPosition(), ",patrolpoint location:",patrolPoint,",the distance between them:",nextpatrodistance)
                if nextpatrodistance < patrolDistances:
                    reward += 100
                else:
                    reward -= 10

        # If the number of invaders has decreased, then the agent has eaten an invader
        if len(nextInvaders) < len(currentInvaders):
            if len(nextInvaders) > 0:
                print("There are invaders")
                reward += 100 * (len(currentInvaders) / len(nextInvaders))
            else :  # We've caught all invaders!
                reward += 100
                print("There are no invaders")



        if len(nextInvaders) > 0:
            reward = 0
        # Distance to the closest invader in the current state
            currentDistToInvader = min([self.getMazeDistance(currentAgentState.getPosition(), invader.getPosition()) 
                                        for invader in currentInvaders])
        # Distance to the closest invader in the next state
            nextDistToInvader = min([self.getMazeDistance(nextAgentState.getPosition(), invader.getPosition()) 
                                     for invader in nextInvaders])
            # print(nextDistToInvader)

            # If the agent is moving closer to an invader, give a penalty
            if nextDistToInvader < currentDistToInvader:
                print("current distance: ",currentDistToInvader, ">next distance: ",nextDistToInvader,"-----Moving closer to invader")
                reward += 10
            # If the agent is moving away from an invader, give a reward
            if nextDistToInvader > currentDistToInvader:
                print("current distance: ",currentDistToInvader, "< next distance: ",nextDistToInvader,"-----moving far away to invader")
                reward -= 10
        # if len(currentInvaders) == 0:

        print("Agent ", self.index," reward ",reward, gameState.getLegalActions(self.index))
        return reward

    
    def getEscapeReward(self, gameState: GameState, nextState: GameState):
        oldDistToHome = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), self.startPosition)
        newDistToHome = self.getMazeDistance(nextState.getAgentState(self.index).getPosition(), self.startPosition)
        walls = gameState.getWalls()
        reward = 0
        if newDistToHome < oldDistToHome:
            reward = 10  # Reward for moving closer to home
        elif newDistToHome > oldDistToHome:
            reward = -5  # Penalty for moving away from home
        else:
            reward = -1  # Small penalty for not moving

        if newDistToHome == 0:  # Agent has reached home
            reward += 100

        if nextState.getAgentState(self.index).getPosition() == self.startPosition:  # Agent got caught
            reward -= 50

        print("Agent ", self.index," reward ",reward)
        return reward

    def getAvoidEnemyReward(self, gameState: GameState, nextState: GameState):
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)

        reward = 0
        currDist = min(self.getMazeDistance(currentAgentState.getPosition(), g) for g in ghosts)
        nextDist = min(self.getMazeDistance(nextAgentState.getPosition(), g) for g in ghosts)

        if nextDist > currDist:
            # If the agent has increased the distance to the nearest enemy, give a high reward
            reward += 8
        elif nextDist == currDist:
            # If the distance has not changed, give a small reward
            reward += 1
        else:
            # If the agent has decreased the distance to the nearest enemy, give a smaller penalty
            reward -= 5

        # If the agent is eaten by a ghost, give a large penalty
        # if nextAgentState.isPacman and len([g for g in ghosts if g == nextAgentState.getPosition()]) > 0:
        if nextState.getAgentState(self.index).getPosition() == self.startPosition:
            reward -= 100
        

        print("Agent ", self.index," reward ",reward)
        return reward

    # def getAvoidEnemyReward(self, gameState: GameState, nextState: GameState):
    #     currentAgentState:AgentState = gameState.getAgentState(self.index)
    #     nextAgentState:AgentState = nextState.getAgentState(self.index)

    #     ghosts = self.getGhostLocs(gameState)
    #     ghost_1_step = [g for g in ghosts if nextAgentState.getPosition() in Actions.getLegalNeighbors(g)]

    #     reward = 0
    #     currDist = min(self.getMazeDistance(currentAgentState.getPosition(), e) for e in ghosts)
    #     if len(ghost_1_step) == 0:
    #         # Handle the case where ghost_1_step is empty
    #         return 0  # Return a default value or handle it as desired

    #     nextDist = min(self.getMazeDistance(nextAgentState.getPosition(), e) for e in ghost_1_step)

    #     if nextDist > currDist:
    #         # If the agent has increased the distance to the nearest enemy, give a high reward
    #         return 10
    #     elif nextDist == currDist:
    #         # If the distance has not changed, give a small reward
    #         return 1
    #     else:
    #         # If the agent has decreased the distance to the nearest enemy, give a penalty
    #         return -10



    #------------------------------- Feature Related Action Functions -------------------------------


    
    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState) 
        currAgentState = gameState.getAgentState(self.index)

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)
        
        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(nextState)/(walls.width+walls.height) * 10

        # Bias
        features["bias"] = 1.0
        
        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts) 
        
        
        dist_home =  self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1
        
        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height)) # The closer to home, the larger food carried, more chance return food
        # Risk of carrying food while a ghost is nearby
        if currAgentState.numCarrying > 0 and features["#-of-ghosts-1-step-away"] > 0:
            features["carry-risk"] = 1
        else:
            features["carry-risk"] = 0

        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = dist/(walls.width+walls.height)
        else:
            features["closest-food"] = 0

        if action == Directions.STOP: features['stop'] = 1

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]
    


    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        walls = gameState.getWalls()

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)/(walls.width+walls.height)

        # Distance to home
        features["distanceToHome"] = self.getMazeDistance(myPos, self.startPosition)/(walls.width+walls.height)

        # if action == Directions.STOP: features['stop'] = 1
        # features["distanceToHome"] = self.getMazeDistance(myPos,self.startPosition)

        # Number of food pellets the agent is carrying
        features['carrying'] = gameState.getAgentState(self.index).numCarrying

        return features

 


    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]

    

    def getDefensiveFeatures(self, gameState, action):
        def aStarSearch(self, gameState, goal):
            """Returns a list of actions that leads to the `goal` position"""
            start = gameState.getAgentState(self.index).getPosition()
            actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
            frontier = util.PriorityQueue()
            frontier.push((start, []), 0)
            explored = set()

            while not frontier.isEmpty():
                node, path = frontier.pop()

                if node in explored:
                    continue

                explored.add(node)

                if node == goal:
                    return path

                for action in actions:
                    dx, dy = Actions.directionToVector(action)
                    nextx, nexty = int(node[0] + dx), int(node[1] + dy)
                    if not gameState.hasWall(nextx, nexty):
                        nextNode = (nextx, nexty)
                        cost = len(path) + 1
                        frontier.push((nextNode, path + [action]), cost)

            return []  # No path found
        features = util.Counter()
        previous_state = gameState.getAgentState(self.index)
        prePos = previous_state.getPosition()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        walls = gameState.getWalls()
        

        # team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        # team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        # features['teamDistance'] = team_dist / (walls.width + walls.height)

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        # features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [len(aStarSearch(self, gameState, a.getPosition())) for a in invaders]
            features['invaderDistance'] = min(dists) / (walls.width + walls.height)

        if len(invaders) > 0:  # invaders present
            min_invader_distance = min([len(aStarSearch(self, gameState, a.getPosition())) for a in invaders])
            next_min_invader_distance = min([len(aStarSearch(self, successor, a.getPosition())) for a in invaders])
            features['chaseInvader'] = 1 if next_min_invader_distance < min_invader_distance else 0

        
        # Check if the agent is cornered
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

                # Define the patrol points
        # defensiveHalfGrid = gameState.getRedFood()  # assuming self.red tells if you're on the red team
        # patrolPoints = [(x, y) for x in range(defensiveHalfGrid.width)
        #                 for y in range(defensiveHalfGrid.height) if defensiveHalfGrid[x][y]]
        # patrolPointsQueue = []
        
            
        # # Refill the patrol points queue if it's empty
        # if len(patrolPointsQueue) == 0:
        #     patrolPointsQueue = patrolPoints.copy()

        # if len(invaders) == 0:
        #     # Get the next patrol point
        #     patrolPoint = patrolPointsQueue[0]

        #     # Check if the agent has reached the patrol point
        #     if patrolPoint == prePos:
        #         # Remove this patrol point from the queue
        #         patrolPointsQueue.pop(0)

        #         # Get the next patrol point
        #         if len(patrolPointsQueue) > 0:
        #             patrolPoint = patrolPointsQueue[0]
        #         else:
        #             patrolPoint = None

        #     min_patrol_distance = min([len(aStarSearch(self,gameState, patrolPoint))])
        # if len(invaders) == 0: features['patrolDistance'] = min_patrol_distance/ (walls.width + walls.height) 
        # else:features['patrolDistance'] = 0
        

        return features


    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]
    
    def getAvoidEnemyFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes distance to enemies we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesPos = [a.getPosition() for a in enemies if a.getPosition() != None and not a.isPacman]
        
        if len(enemiesPos) > 0:
            closestEnemyDist = min([self.getMazeDistance(myPos, enemyPos) for enemyPos in enemiesPos])
            features['closestEnemy'] = closestEnemyDist

        features['#-of-enemies-nearby'] = len([enemyPos for enemyPos in enemiesPos if self.getMazeDistance(myPos, enemyPos) < 5])

        if action == Directions.STOP: features['stop'] = 1

        return features

    
    def getAvoidEnemyWeights(self):
        return MixedAgent.QLWeights["avoidEnemyWeights"]

    
    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def stateClosestFood(self, gameState:GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def getGhostLocs(self, gameState:GameState):
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        return ghosts
    


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
    # You should add your weights for new
    #  low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.

    QLWeights = {
        
            'offensiveWeights': {
                'closest-food': -69.23206445189587,
                'bias': -26.593570411700416,
                '#-of-ghosts-1-step-away': -8.179959069618443,
                'successorScore': 28.044486285577044,
                'stop': 0.01585392208029396,
                'chance-return-food': 45.930031644416246,
                'depositFood': 89.91953977913874,
                'eats-ghost': 0.9054075832911208,
                'enemyscared': 26.309370322774168,
                'distance-to-enemyscared': -28.819302342065594,
                'closest-capsule': -7.26705307545401
            },
            'defensiveWeights': {
                'onDefense': -20.625694922126886,
                'distance_to_middlle': -267.74079755550525,
                'numInvaders': -10.09488135165112,
                'invaderDistance': -152.510650923526605,
                'stop': -25.402700611111896,
                'reverse': -18.605275701574566,
                'distaToInvader_noisyDist': -250.29566210941928
            },
            'escapeWeights': {
                'onDefense': 44.358274346154914,
                'enemyDistance': 30.474666124856483,
                'stop': -100,
                'carrying': 10.0,
                'distanceToHome': 28.237126757645846
            }
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
        self.trainning = False # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.01 #default exploration prob, change to take a random step
        self.alpha = 0.05 #default learning rate
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


    # def final(self, gameState : GameState):
    #     """
    #     This function write weights into files after the game is over. 
    #     You may want to comment (disallow) this function when submit to contest server.
    #     """
    #     print("Write QLWeights:", MixedAgent.QLWeights)
    #     file = open(MixedAgent.QLWeightsFile, 'w')
    #     file.write(str(MixedAgent.QLWeights))
    #     file.close()


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
            highLevelAction = 'defence'
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
        if self.index == 2:
            return self.defense(objects, initState)
        else:
            return self.goalScoring(objects, initState)

    def defense(self,objects: List[Tuple], initState: List[Tuple]):
        positiveGoal = [("is_agent2",)]
        negtiveGoal = []
        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]

            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)]
        return positiveGoal, negtiveGoal

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
                    # debug
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
        if highLevelAction == "attack" or highLevelAction == "avoid_enemy":
            # The q learning process for offensive actions are complete,
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update,
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update

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
        if self.index == 2:
            print("features:",features,"weight",weights,'result:',features * weights)
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
            if feature == 'stop':
                print("weight for chance:",weights[feature],"correction:",correction,"reward:",reward, "first of correction:",reward + self.discountRate*self.getValue(nextState, featureFunction, weights),"second",self.getQValue(features, weights))
            weights[feature] =weights[feature] + learningRate*correction * features[feature]

    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def AStarSearch(self, gameState, goal):
        """Returns a list of actions that leads to the `goal` position"""
        start = gameState.getAgentState(self.index).getPosition()
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        frontier = util.PriorityQueue()
        frontier.push((start, []), 0)
        explored = set()
        boundary_x = gameState.getWalls().width // 2

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

    # This function is for half grid search
    def AStarSearch_middle(self, gameState, goal):
        """Returns a list of actions that leads to the `goal` position"""
        start = gameState.getAgentState(self.index).getPosition()
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        frontier = util.PriorityQueue()
        frontier.push((start, []), 0)
        explored = set()
        boundary_x = gameState.getWalls().width // 2

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
                if nextx > boundary_x:
                    continue
                if not gameState.hasWall(nextx, nexty):
                    nextNode = (nextx, nexty)
                    cost = len(path) + 1
                    frontier.push((nextNode, path + [action]), cost)

        return []  # No path found

    def distToFood(self, gamestate, food):
        myPos = gamestate.getAgentPosition(self.index)
        foodlist = []
        if len(food) <= 0:
            return 0
        for foodPos in food:
            foodlist.append(self.getMazeDistance(foodPos, myPos))

        return foodlist

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

        base_reward =  -10 + nextAgentState.numReturned + nextAgentState.numCarrying * 2
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)
        currentDistToFood = min(self.distToFood(gameState, self.getFood(gameState).asList()))
        nextDistTofood = min(self.distToFood(nextState, self.getFood(nextState).asList()))
        if currentDistToFood > nextDistTofood:
            # if we get closer to food
            base_reward += 2

        if ghost_1_step > 0:
            base_reward -= 5
        if score <0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned * 3
        # gameState.getLegalActions(self.index)
        # Calculate distances to the ghosts in the current and next states
        if len(ghosts) > 0:
            currentMinDistToGhost = min([self.getMazeDistance(currentAgentState.getPosition(), g) for g in ghosts])
            nextMinDistToGhost = min([self.getMazeDistance(nextAgentState.getPosition(), g) for g in nextghosts])
            # If the agent is moving away from a ghost, give a reward
            if nextMinDistToGhost > currentMinDistToGhost:
                base_reward += 3
            else:
                base_reward -= 1


        print(ghosts)
        # print(gameState.getAgentDistances())

        print("Agent ", self.index," reward ",base_reward, gameState.getLegalActions(self.index))

        return base_reward
    
    def getDefensiveReward(self,gameState: GameState, nextState: GameState):
        # Current and next agent states
        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)
        walls = gameState.getWalls()
        middle_row = walls.height // 2
        middle_with = walls.width // 2
        while walls[middle_with][middle_row]:
            # Try moving down first
            if search_y_down > 0:
                search_y_down -= 1
                if not walls[middle_with][search_y_down]:
                    middle_row = search_y_down
                    break
            # If that didn't work or isn't possible, try moving up
            if search_y_up < walls.height - 1:
                search_y_up += 1
                if not walls[middle_with][search_y_up]:
                    middle_row = search_y_up
                    break
        rightmost_point = (middle_with,middle_row)
        noisyDistance = gameState.getAgentDistances()
        nextnoisyDistance = nextState.getAgentDistances()
        # print(noisyDistance)
        # print(nextnoisyDistance)
        enemy_index = [i for i in self.getOpponents(gameState)]
        long_dist_invader = [noisyDistance[a] for a in enemy_index if (gameState.getAgentState(a).isPacman)]
        long_dist_invader_next = [nextnoisyDistance[a] for a in enemy_index if (gameState.getAgentState(a).isPacman)]


        # Current and next invaders
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        currentInvaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]
        nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman and nextState.getAgentState(i).getPosition() != None]

        # print(currentInvaders)
        reward = 0
        print("numberof invader",len(invaders))
        print("number of another method invader", len(currentInvaders))
        # if no invader and head to the middle
        if len(invaders) == 0 and len(long_dist_invader) == 0:
            CurrentDistanceToMiddle = len(self.AStarSearch(gameState, rightmost_point))
            NextistanceToMiddle = len(self.AStarSearch(nextState, rightmost_point))
            if CurrentDistanceToMiddle > NextistanceToMiddle:
                reward -= 20

        if len(invaders) == 0 and len(long_dist_invader) > 0:
            currenDistToPac = min(long_dist_invader)
            print(long_dist_invader)
            nextDistToPac = min(long_dist_invader_next)
            print(nextDistToPac)
            if nextDistToPac < currenDistToPac:
                reward -= 10
        # If the number of invaders has decreased, then the agent has eaten an invader
        if len(nextInvaders) < len(currentInvaders):
            if len(nextInvaders) > 0:
                print("There are invaders")
                reward -= 100 * (len(currentInvaders) / len(nextInvaders))
            else :  # We've caught all invaders!
                reward -= 100
                print("There are no invaders")


        if len(nextInvaders) > 0:
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
                reward += 20
            # If the agent is moving away from an invader, give a reward
            if nextDistToInvader > currentDistToInvader:
                print("current distance: ",currentDistToInvader, "< next distance: ",nextDistToInvader,"-----moving far away to invader")
                reward -= 10
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




    #------------------------------- Feature Related Action Functions -------------------------------



    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState)
        currAgentState = gameState.getAgentState(self.index)
        succerssor = gameState.generateSuccessor(self.index, action)
        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)
        # print("thisis ghost location",ghosts)


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

        # if the enemy are scared or not
        enemyscared_1 = True if gameState.getAgentState(self.getOpponents(gameState)[0]).scaredTimer > 1.0 else 0
        enemyscared_2 = True if gameState.getAgentState(self.getOpponents(gameState)[1]).scaredTimer > 1.0 else 0
        features['enemyscared'] = 1.0 if enemyscared_1 or enemyscared_2 else 0
        if features['enemyscared'] == 1.0:
            if len(ghosts) >0:
                features['distance-to-enemyscared'] = 1.0 + min(self.getMazeDistance(gameState.getAgentPosition(self.index), g) for g in ghosts) / float(walls.width + walls.height)
            else:
                features['distance-to-enemyscared'] = 0

        if features['enemyscared'] == 1.0  and (next_x, next_y) in ghosts:  # if enemy 1 scared
            features["eats-ghost"] = 1.0   # eat enemy
        else:
            features["eats-ghost"] = 0

        dist_home =  self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["depositFood"] = 1.0 if succerssor.getScore() - gameState.getScore() > 0 else 0

        # Risk of carrying food while a ghost is nearby
        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height)) # The closer to home, the larger food carried, more chance return food
        # features["carry-risk"] = (currAgentState.numCarrying) / dist_home
        # get the capsule feature
        if len(self.getCapsules(gameState)) > 0:
            min_capsule_distance = min(self.getMazeDistance(gameState.getAgentPosition(self.index), capsule)
                                    for capsule in self.getCapsules(gameState))
            features['closest-capsule'] = min_capsule_distance / float(walls.width + walls.height)
        else:
            features['closest-capsule'] = 0
        # Closest food
        print("time_left",gameState.data.timeleft)
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = dist/float(walls.width+walls.height)
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
        features['onDefense'] = 1.0
        if myState.isPacman: features['onDefense'] = 0.0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)/float(walls.width+walls.height)

        # Distance to home
        features["distanceToHome"] = self.getMazeDistance(myPos, self.startPosition)/float(walls.width+walls.height)

        # Number of food pellets the agent is carrying
        features['carrying'] = gameState.getAgentState(self.index).numCarrying

        return features




    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]


    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        walls = gameState.getWalls()
        previous_state = gameState.getAgentState(self.index)
        prePos = previous_state.getPosition()
        successor = self.getSuccessor(gameState, action)
        defensiveHalfGrid = gameState.getRedFood()
        middle_with = walls.width // 2
        middle_row = walls.height // 2
        # Start searching from the middle point
        search_y_up = middle_row
        search_y_down = middle_row
        # If the middle point is a wall, adjust the y-coordinate
        while walls[middle_with][middle_row]:
            # Try moving down first
            if search_y_down > 0:
                search_y_down -= 1
                if not walls[middle_with][search_y_down]:
                    middle_row = search_y_down
                    break
            # If that didn't work or isn't possible, try moving up
            if search_y_up < walls.height - 1:
                search_y_up += 1
                if not walls[middle_with][search_y_up]:
                    middle_row = search_y_up
                    break

        rightmost_point = (middle_with,middle_row)

        noisyDistance = successor.getAgentDistances()
        enemy_index = [i for i in self.getOpponents(gameState)]
        # rightmost_point = max((x, y) for x in range(walls.width) for y in range(walls.height) if defensiveHalfGrid[x][y] and y == middle_row)
        # print("this is righ point:",rightmost_point)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        features['onDefense'] = 0.0
        if myState.isPacman: features['onDefense'] = 1.0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        # if we face confloc
        enemies_notpac = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        dist_notpac = [self.getMazeDistance(prePos, a.getPosition()) for a in enemies_notpac]
        if len(enemies_notpac) > 0:
            dist_notpac = min(dist_notpac)
            if dist_notpac == 1 or dist_notpac == 2:
                features['reverse'] =  -1.0
                features['stop'] = -1.0

        # calculate the approximate dist
        long_dist_invader = [noisyDistance[a] for a in enemy_index if (gameState.getAgentState(a).isPacman)]
        if len(long_dist_invader) > 0 and len(invaders) == 0:
            print('long_dist_invader:',long_dist_invader)
            features['distance_to_middlle'] = 0
            features["distaToInvader_noisyDist"] = min(long_dist_invader) / float(walls.width * walls.height)

        # print("distance_test",successor.getAgentState(self.index).getPosition(),rightmost_point, gameState.getWalls())
        if len(invaders) > 0:  # invaders present
            dists = [len(self.AStarSearch(successor, a.getPosition())) for a in invaders]
            features['invaderDistance'] = min(dists) / float(walls.width * walls.height)
            features['distance_to_middlle'] = 0
        # print("distance",len(self.AStarSearch(gameState, rightmost_point)))
        if len(invaders) == 0:
            features['distance_to_middlle'] = len(self.AStarSearch_middle(successor, rightmost_point)) / float(walls.width * walls.height)
            features['invaderDistance'] = 0

        # Check if the agent is cornered
        if action == Directions.STOP: features['stop'] = 1.0/float(walls.width+walls.height)
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1.0/float(walls.width+walls.height)

        return features


    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]


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

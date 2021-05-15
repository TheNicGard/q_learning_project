#!/usr/bin/env python3

import rospy
import numpy as np
import os
import csv, time, copy
from gazebo_msgs.msg import ModelStates
from q_learning_project.msg import QMatrixRow, QMatrix, QLearningReward, RobotMoveDBToBlock

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

class QLearning(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("q_learning")

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-9 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][12] = 5
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt")

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { dumbbell: "red", block: 1}
        colors = ["red", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"dumbbell": colors[int(x[0])], "block": int(x[1])},
            self.actions
        ))


        # Fetch states. There are 64 states. Each row index corresponds to the
        # state number, and the value is a list of 3 items indicating the positions
        # of the red, green, blue dumbbells respectively.
        # e.g. [[0, 0, 0], [1, 0 , 0], [2, 0, 0], ..., [3, 3, 3]]
        # e.g. [0, 1, 2] indicates that the green dumbbell is at block 1, and blue at block 2.
        # A value of 0 corresponds to the origin. 1/2/3 corresponds to the block number.
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.txt")
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))

        self.q_matrix = None
        self.reward_update = 0

        # set learning rate and discount factor
        self.learning_rate = 1
        self.discount_factor = 0.5

        self.q_matrix_pub= rospy.Publisher("/q_learning/q_matrix", QMatrix, queue_size=10)
        self.action_pub = rospy.Publisher("/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10)
        self.reward_sub = rospy.Subscriber("/q_learning/reward", QLearningReward, self.reward_call_back)
        self.model_state_sub = rospy.Publisher("/gazebo/set_model_states", ModelStates, self.model_state_callback)
        # wait for publishers and subscribers to initialize
        time.sleep(1)


    # saves q_matrix to q_matrix.csv
    def save_q_matrix(self):
        with open(os.path.dirname(__file__) + "/q_matrix.csv", 'w') as csvfile:
            q_matrix_writer = csv.writer(csvfile)
            for row_of_states in self.q_matrix:
                csv_row = []
                for action_value in row_of_states:
                    csv_row.append(str(action_value))
                q_matrix_writer.writerow(csv_row)
        return


    # loads q_matrix from csv, not sure where we need this but
    # we will need it somewhere
    def load_q_matrix(self):
        with open(os.path.dirname(__file__)+ "/q_matrix.csv", 'r') as csvfile:
            q_matrix_writer = csv.reader(csvfile)
            self.q_matrix = []
            for row in q_matrix_writer:
                self.q_matrix.append(list(map(lambda x : int(x), row)))


    # publish commands and await rewards according to the values in
    # self.q_matrix, or loads the q_matrix from the csv
    def execute_according_to_q_matrix(self, q_location:str='self'):
        if q_location == "csv":
            self.load_q_matrix()
        if self.q_matrix is not None:
            current_state_num = 0
            self.state_update = False
            while not rospy.is_shutdown():
                # get allowed actions
                allowed_actions = []
                for state in range(len(self.states)):
                    action_num = self.action_matrix[current_state_num][state]
                    if action_num >= 0:
                        allowed_actions.append((int(action_num), int(state)))
                # pick and publish best action, update state
                if len(allowed_actions) > 0:
                    best_q_val = self.q_matrix[current_state_num][allowed_actions[0][0]]
                    best_action = allowed_actions[0][0]
                    next_state_num = allowed_actions[0][1]
                    for (action_num, next_state) in allowed_actions:
                        if self.q_matrix[current_state_num][action_num] > best_q_val:
                            best_q_val = self.q_matrix[current_state_num][action_num]
                            best_action = action_num
                            next_state_num = next_state
                    action_dict = self.actions[best_action]
                    print(action_dict)
                    action_msg = RobotMoveDBToBlock()
                    action_msg.robot_db = action_dict['dumbbell']
                    action_msg.block_id = action_dict['block']
                    self.action_pub.publish(action_msg)
                    current_state_num = next_state_num
                    # wait until state changes to move to next action
                    while self.state_update == False:
                        time.sleep(1)
                    # reset check for state callback
                    self.state_update = False
                else:
                    # if no allowed actions, wait for reset
                    current_state_num = 0
        else:
            raise Exception("'execute_according_to_q_matrix' could not find q_matrix")


    def model_state_callback(self, data):
        # get the initial locations of the three numbered blocks
        if (self.current_numbered_blocks_locations == None):
            self.current_numbered_blocks_locations = {}
            for block_id in self.numbered_block_model_names:
                block_idx = data.name.index(self.numbered_block_model_names[block_id])
                self.current_numbered_blocks_locations[block_id] = data.pose[block_idx].position

        db_block_mapping = {}
        for robot_db in self.robot_dbs:
            db_idx = data.name.index(self.db_model_names[robot_db])
            db_block_mapping[robot_db] = self.is_in_front_of_any_block(data.pose[db_idx])

        # if a dumbbell has moved in front of a numbered block, set self.state_update=true
        for robot_db in self.robot_dbs:
            if (self.current_robot_db_locations[robot_db] != db_block_mapping[robot_db]):
                self.state_update = True
        
        


    def reward_call_back(self, data):
        self.reward = data.reward
        self.reward_update = 1


    # run the algorithm to train the q_matrix
    def train_q_matrix(self):
        # initialize q_matrix
        self.q_matrix = [[0 for i in range(len(self.actions))] for j in range(len(self.states))]
        self.last_checked_q_matrix = [[0 for i in range(len(self.actions))] for j in range(len(self.states))]

        current_state_num = 0
        self.reward_update = 0
        self.state_update = 0
        q_matrix_difference = 100
        iteration = 0
        while q_matrix_difference > 10:
            # training process
            # generate a random action

            allowed_actions = []
            for state in range(len(self.states)):
                action_num = self.action_matrix[current_state_num][state]
                if action_num >= 0:
                    allowed_actions.append((action_num, state))
            if len(allowed_actions) > 0:
                action_state_pair = allowed_actions[int(np.random.randint(0, len(allowed_actions)))]
                chosen_action_num = int(action_state_pair[0])
                action_dict = self.actions[chosen_action_num]
                action_msg = RobotMoveDBToBlock()
                action_msg.robot_db = action_dict['dumbbell']
                action_msg.block_id = action_dict['block']
                self.action_pub.publish(action_msg)
                # wait until we get reward and update accordingly
                while self.reward_update == 0:
                    time.sleep(1)
                # reset check for reward callback
                self.reward_update = 0

                next_state_num = action_state_pair[1]
                best_q_value = 0
                for action_q_value in self.q_matrix[next_state_num]:
                    if action_q_value >= best_q_value:
                        best_q_value = action_q_value
                q_matrix_adjustment = self.reward + (self.discount_factor * best_q_value) - self.q_matrix[current_state_num][chosen_action_num]
                self.q_matrix[current_state_num][chosen_action_num] += int(self.learning_rate * q_matrix_adjustment)
                
                # publish new q_matrix
                self.q_matrix_msg = QMatrix()
                for state_row in self.q_matrix:
                    q_row = QMatrixRow()
                    for q_value in state_row:
                        q_row.q_matrix_row.append(q_value)
                    self.q_matrix_msg.q_matrix.append(q_row)
                self.q_matrix_pub.publish(self.q_matrix_msg)
                current_state_num = next_state_num
                
                iteration += 1
                print("Training iteration:", iteration)
                if (iteration % (len(self.states) * len(self.actions))) == 0:
                    q_matrix_difference = 0
                    for s_index, state in enumerate(self.q_matrix):
                        for a_index, value in enumerate(state):
                            q_matrix_difference += abs(value - self.last_checked_q_matrix[s_index][a_index])
                    self.last_checked_q_matrix = copy.deepcopy(self.q_matrix)
                    print("New q_matrix_difference:", q_matrix_difference)
            else: # no actions left
                current_state_num = 0



if __name__ == "__main__":
    node = QLearning()
    try:
        if rospy.get_param('~train'):
            node.train_q_matrix()
            node.save_q_matrix()
            print("QMatrix Converged and Saved to q_matrix.csv")
        else:
            node.execute_according_to_q_matrix("csv")
    except:
        print("No such paramters")
        node.execute_according_to_q_matrix("csv")

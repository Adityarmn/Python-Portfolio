import tensorflow as tf
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import os
import random

class DQNAgentBoostrapped:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.num_heads = 3
        self.eval_network = {}
        self.target_network = {}
        self.selected_head = 0

        self.gamma = 0.95
        self.learning_rate = 0.001
        self.learning_step1 = 0
        self.learning_step2 = 0
        self.learning_step3 = 0
        self.temperature_schedule = 0.5

        self.epsilon = 1.0 #exploration rate, perlu exploration karena DRL bisa evolve over time dan bisa ada informasi yang enggak ke cover
        self.epsilon_decay = 0.995 # decay our epsilon so we slowly shift our agent from exploring at random to exploiting the knowledge that it is learned
        self.epsilon_min = 0.01 #Kalo ke decay terus dibuat bates minimum biar setidaknya bisa eksplorasi
        self.replace_target_freq = 1000
        self.eval_network1 = None
        self.target_network1 = None

        for k in range (self.num_heads):
            self.eval_network["headeval{0}".format(k)] = self.build_model()
        for j in range (self.num_heads):
            self.target_network["headtarget{0}".format(j)] = self.build_model()

    def huber_loss(self, y_true, y_predict, delta=1.0):
        err = y_true - y_predict
        cond = K.abs(err) <= delta

        L1 = 0.5 * K.square(err)
        L2 = delta*K.abs(err) - 0.5*delta**2
        return K.mean(tf.where(cond, L1, L2))

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss= 'mse', optimizer=Adam(lr=self.learning_rate)) # Bisa ganti huber loss buat ganti mse

        return model

    def update_target_weights1(self):
        self.target_network["headtarget0"].set_weights(self.eval_network["headeval0"].get_weights())

    def update_target_weights2(self):
        self.target_network["headtarget1"].set_weights(self.eval_network["headeval1"].get_weights())

    def update_target_weights3(self):
        self.target_network["headtarget2"].set_weights(self.eval_network["headeval2"].get_weights())

    def choose_model(self):
        self.selected_head = 1 #np.random.randint(self.num_heads)
        self.eval_network1 = self.eval_network["headeval"+str(self.selected_head)]
        self.target_network1 = self.target_network["headtarget"+str(self.selected_head)]

    def act(self, state):
        # Untuk boltzman exploration
        action_values = self.eval_network1.predict(state[np.newaxis, :])
        exp_probabilities = np.exp(action_values / self.temperature_schedule)
        probabilities = exp_probabilities / np.sum(exp_probabilities)
        probabilities = np.nan_to_num(probabilities)
        #print(probabilities[0])
        # choose actions according to the probabilities
        action = np.random.choice(range(self.action_size), p=probabilities[0])
        #print(probabilities[0], action)
        return action
        '''
        #Untuk epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) #eksplorasi
        else:
            action_probs = self.eval_network1.predict(state[np.newaxis, :])
            return np.argmax(action_probs[0])
        '''

    def replay1(self, states, actions, rewards, states_next, done):

        if self.learning_step1 % self.replace_target_freq == 0:
            self.update_target_weights1()

        rows = np.arange(done.shape[0])
        not_done = np.logical_not(done)

        eval_next = self.eval_network["headeval0"].predict(states_next)
        target_next = self.target_network["headtarget0"].predict(states_next)
        discounted_rewards = self.gamma * target_next[rows, np.argmax(eval_next, axis=1)]

        y = self.eval_network["headeval0"].predict(states)

        if not any(action is None for action in actions):
            y[rows, actions] = rewards
            y[not_done, actions[not_done]] += discounted_rewards[not_done]

        history = self.eval_network["headeval0"].fit(states, y, epochs=1, verbose=0)
        self.learning_step1 += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history

    def replay2(self, states, actions, rewards, states_next, done):

        if self.learning_step2 % self.replace_target_freq == 0:
            self.update_target_weights2()

        rows = np.arange(done.shape[0])
        not_done = np.logical_not(done)

        eval_next = self.eval_network["headeval1"].predict(states_next)
        target_next = self.target_network["headtarget1"].predict(states_next)
        discounted_rewards = self.gamma * target_next[rows, np.argmax(eval_next, axis=1)]

        y = self.eval_network["headeval1"].predict(states)

        if not any(action is None for action in actions):
            y[rows, actions] = rewards
            y[not_done, actions[not_done]] += discounted_rewards[not_done]

        history = self.eval_network["headeval1"].fit(states, y, epochs=1, verbose=0)
        self.learning_step2 += 1

        return history

    def replay3(self, states, actions, rewards, states_next, done):

        if self.learning_step3 % self.replace_target_freq == 0:
            self.update_target_weights3()

        rows = np.arange(done.shape[0])
        not_done = np.logical_not(done)

        eval_next = self.eval_network["headeval2"].predict(states_next)
        target_next = self.target_network["headtarget2"].predict(states_next)
        discounted_rewards = self.gamma * target_next[rows, np.argmax(eval_next, axis=1)]

        y = self.eval_network["headeval2"].predict(states)

        if not any(action is None for action in actions):
            y[rows, actions] = rewards
            y[not_done, actions[not_done]] += discounted_rewards[not_done]

        history = self.eval_network["headeval2"].fit(states, y, epochs=1, verbose=0)
        self.learning_step3 += 1

        return history

    def load(self, name1, name2, name3):
        self.eval_network["headeval0"].load_weights(name1)
        self.eval_network["headeval1"].load_weights(name2)
        self.eval_network["headeval2"].load_weights(name3)
        self.update_target_weights1()
        self.update_target_weights2()
        self.update_target_weights3()

    def save(self, name1, name2, name3):
        self.eval_network["headeval0"].save_weights(name1)
        self.eval_network["headeval1"].save_weights(name2)
        self.eval_network["headeval2"].save_weights(name3)


class Memory:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
        self.pointer = 0

    def remember(self, state, action, reward, state_next, done, mask, num_agent):
        experience = (state, action, reward, state_next, done, mask, num_agent)
        self.data.append(experience)
        if self.pointer < len(self.data):
            self.pointer += 1

    def sample(self, batch, agents=1):
        """
        If 1 agent, assumes no data about other agents.
        If 2+ agents, assumes data contains all agent data.
        """
        if agents == 1:
            states = np.array([self.data[i][0] for i in batch])
            actions = np.array([self.data[i][1] for i in batch])
            states_next = np.array([self.data[i][3] for i in batch])

        rewards = np.array([self.data[i][2] for i in batch])
        dones = np.array([self.data[i][4] for i in batch])

        return states, actions, rewards, states_next, dones

class MemoryHeads:
    def __init__(self, capacity):
        self.datahead1 = deque(maxlen=capacity)
        self.datahead2 = deque(maxlen=capacity)
        self.datahead3 = deque(maxlen=capacity)
        self.pointer1 = 0
        self.pointer2 = 0
        self.pointer3 = 0

    def sample1(self, batch, agents=1):
        if not self.datahead1 :
            pass
        else:
            states = np.array([self.datahead1[i][0] for i in batch])
            actions = np.array([self.datahead1[i][1] for i in batch])
            states_next = np.array([self.datahead1[i][3] for i in batch])

            rewards = np.array([self.datahead1[i][2] for i in batch])
            dones = np.array([self.datahead1[i][4] for i in batch])

            return states, actions, rewards, states_next, dones

    def sample2(self, batch, agents=1):
        if not self.datahead2:
            pass
        else:
            states = np.array([self.datahead2[i][0] for i in batch])
            actions = np.array([self.datahead2[i][1] for i in batch])
            states_next = np.array([self.datahead2[i][3] for i in batch])

            rewards = np.array([self.datahead2[i][2] for i in batch])
            dones = np.array([self.datahead2[i][4] for i in batch])

            return states, actions, rewards, states_next, dones

    def sample3(self, batch, agents=1):
        if not self.datahead3:
            pass
        else:
            states = np.array([self.datahead3[i][0] for i in batch])
            actions = np.array([self.datahead3[i][1] for i in batch])
            states_next = np.array([self.datahead3[i][3] for i in batch])

            rewards = np.array([self.datahead3[i][2] for i in batch])
            dones = np.array([self.datahead3[i][4] for i in batch])

            return states, actions, rewards, states_next, dones

import tensorflow as tf
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import os
import random


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.learning_rate = 0.001
        self.learning_step = 0
        self.temperature_schedule = 0.5

        self.epsilon = 1.0  # exploration rate, perlu exploration karena DRL bisa evolve over time dan bisa ada informasi yang enggak ke cover
        self.epsilon_decay = 0.995 # decay our epsilon so we slowly shift our agent from exploring at random to exploiting the knowledge that it is learned
        self.epsilon_min = 0.01 # Kalo ke decay terus dibuat bates minimum biar setidaknya bisa eksplorasi
        self.replace_target_freq = 2000
        self.eval_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_weights()

    def huber_loss(self, y_true, y_predict, delta=1.0): #y_true = Ground Truth Values (batch_size, d0, dN) y_pred = Value yang diprediksi, bentuknya sama kayak y_true
        err = y_true - y_predict
        cond = K.abs(err) <= delta #Delta adalah poin di mana fungsi huber loss berubah dari kuadratik ke linear

        L1 = 0.5 * K.square(err) #L1 untuk nilai error kecil
        L2 = delta*K.abs(err) - 0.5*delta**2 #L2 untuk nilai error absolut
        return K.mean(tf.where(cond, L1, L2))

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  # Bisa ganti huber loss buat ganti mse
        return model

    def update_target_weights(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def act(self, state):
        action_values = self.eval_network.predict(state[np.newaxis, :]) # Untuk boltzman exploration
        exp_probabilities = np.exp(action_values / self.temperature_schedule)
        probabilities = exp_probabilities / np.sum(exp_probabilities)
        action = np.random.choice(range(self.action_size), p=probabilities[0]) # choose actions according to the probabilities
        return action #print(probabilities[0], action)
        '''
        # Untuk epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) #eksplorasi
        else:
            action_probs = self.eval_network.predict(state[np.newaxis, :])
            #print(action_probs, np.argmax(action_probs[0])) hasil: [[-3.6568227  -2.6656322  -0.97574604 -2.9777777 ]] 2
            return np.argmax(action_probs[0])
        '''

    def replay(self, states, actions, rewards, states_next, done):

        if self.learning_step % self.replace_target_freq == 0:
            self.update_target_weights()

        rows = np.arange(done.shape[0])
        not_done = np.logical_not(done)

        eval_next = self.eval_network.predict(states_next)
        target_next = self.target_network.predict(states_next)
        # ada 256 data (ngikut batch) tiap array/agent
        discounted_rewards = self.gamma * \
            target_next[rows, np.argmax(eval_next, axis=1)]

        y = self.eval_network.predict(states)  # target

        if not any(action is None for action in actions):
            y[rows, actions] = rewards
            y[not_done, actions[not_done]] += discounted_rewards[not_done]

        history = self.eval_network.fit(states, y, epochs=1, verbose=0)
        self.learning_step += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history

    def load(self, name):
        self.eval_network.load_weights(name)
        self.update_target_weights()

    def save(self, name):
        self.eval_network.save_weights(name)


class Memory:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
        self.pointer = 0

    def remember(self, state, action, reward, state_next, done):
        experience = (state, action, reward, state_next, done)
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
        else:
            states = []
            actions = []
            states_next = []
            for a in range(agents):
                states.append(np.array([self.data[i][0][a] for i in batch]))
                actions.append(np.array([self.data[i][1][a] for i in batch]))
                states_next.append(np.array([self.data[i][3][a]
                                             for i in batch]))

        rewards = np.array([self.data[i][2] for i in batch])
        dones = np.array([self.data[i][4] for i in batch])

        return states, actions, rewards, states_next, dones

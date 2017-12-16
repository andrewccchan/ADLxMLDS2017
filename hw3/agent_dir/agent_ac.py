from agent_dir.agent import Agent
from network import ValueNetwork
from utility import SerialNumberManager, History
import numpy as np
import tensorflow as tf
import scipy.misc
from collections import deque
import os
import random
import pdb
import json
import network_ac
import keras.backend as K


class Agent_AC(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_AC,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        K.set_session(self.sess)

        network_params = dict(
            learning_rate = 0.001,
            action_size = [env.get_action_space().n, 1],
            observ_size = [6400]
        )

        self.ac = network_ac.ACNetwork(network_params, self.sess)

        self.max_tirals = 10000
        self.trial_len = 500
        self.epsi = 1.0
        self.epsi_decay = 0.995
        self.tau = 0.125
        self.discount = 0.99
        self.batch_size = 1
        self.action_size = env.get_action_space().n

        self.history = deque(maxlen=2000)

        self.model_name = 'ac'

        self.reward_accu = 0
        self.rewards = []
        self.episode = 0
        if args.train_ac:
            self.log_file = open('log/{}'.format(self.model_name), 'w')


    def init_game_setting(self):
        pass


    def store_history(self, cur_state, action, reward, new_state, done):
        self.history.append([cur_state, action, reward, new_state, done])

    def train(self):
        cur_state = self.preprocess_frame(self.env.reset())
        action  = self.env.get_random_action()
        while True:
            action, action_prob = self.make_action(cur_state, test=False)
            new_state, reward, terminal, _ = self.env.step(action)
            new_state = self.preprocess_frame(new_state)

            self.store_history(cur_state, action_prob, reward, new_state, terminal)

            self.update_network()

            cur_state = new_state

            self.reward_accu += reward

            if terminal:
                cur_state = self.preprocess_frame(self.env.reset())
                action  = self.env.get_random_action()
                self.rewards.append(self.reward_accu)
                self.reward_accu = 0
                self.episode += 1
                if self.episode % 10 == 9:
                    msg = 'Episode {}, mean reward = {}'.format(self.episode, np.mean(self.rewards))
                    print(msg)
                    self.log_file.write(msg + '\n')
                    self.rewards = []



    def make_action(self, observation, test=True):
        self.epsi *= self.epsi_decay
        if np.random.random() < self.epsi and not test:
            action =  self.env.get_random_action()
            y = np.zeros((self.action_size))
            y[action] = 1
            return action, y
        else:
            observation = np.expand_dims(observation, 0)
            action_prob = self.ac.actor_model.predict(observation)
            action_prob /= np.sum(action_prob)
            action = np.random.choice(self.action_size, 1, p=action_prob[0])[0]
            return action, action_prob[0]
    
    def preprocess_frame(self, I):
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        I = I.astype(np.float).ravel()
        return I

    def update_network(self):
        if len(self.history) < self.batch_size:
            return

        rewards = []
        samples = random.sample(self.history, self.batch_size)

        # train critic
        for cur_state, action, reward, new_state, done in samples:

            new_state = np.expand_dims(new_state, 0)
            cur_state = np.expand_dims(cur_state, 0)
            if not done:
                target_action = self.ac.target_actor_model.predict(new_state)
                target_action = np.expand_dims(np.squeeze(target_action), 0)
                future_reward = self.ac.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.discount * future_reward
            action = np.expand_dims(np.squeeze(action), 0)
            reward = np.expand_dims(np.asarray(reward), 1)
            self.ac.critic_model.fit([cur_state, action], reward, verbose=0)
        
        # train actor
        for cur_state, action, reward, new_state, _ in samples:
            new_state = np.expand_dims(new_state, 0)
            cur_state = np.expand_dims(cur_state, 0)           
            predicted_action = self.ac.actor_model.predict(cur_state)
            grads = self.sess.run(self.ac.critic_grads, feed_dict={
                self.ac.critic_state_input:  cur_state,
                self.ac.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.ac.optimizer, feed_dict={
                self.ac.actor_state_input: cur_state,
                self.ac.actor_critic_grad: grads
            })
        

   


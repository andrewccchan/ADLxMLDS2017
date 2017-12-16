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

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
        random.seed(0)
        self.action_size = 3
        self.discount = 0.99
        self.learning_rate = 0.001
        self.model_name = 'pong'

        self.history = History(self.action_size)

        self.epi = 0
        self.prev_observ = None

        self.step_cnt = 0
        self.reward_accu = 0.0
        self.game_len = []
        self.rewards = []
        self.batch_size = 10

        if args.test_pg:
            self.network_params = dict(
                action_size = self.action_size,
                learning_rate = self.learning_rate
            )
            self.val_network = ValueNetwork(self.network_params)
            self.load_model()
            self.val_network.model.summary()
        elif args.train_pg:
            self.network_params = dict(
                action_size = self.action_size,
                learning_rate = self.learning_rate
            )
            self.val_network = ValueNetwork(self.network_params)
            print('Network parameters: {}'.format(self.network_params))
            self.val_network.model.summary()
            with open('{}.json'.format(self.model_name), 'w') as f:
                json.dump(self.val_network.model.to_json(), f)
            self.log_file = open('{}'.format(self.model_name), 'w')




    def load_model(self):
        from keras.models import load_model,model_from_json

        self.val_network.model = model_from_json(json.load(open("{}.json".format(self.model_name))))
        self.val_network.model.load_weights("{}_model.hdf5".format(self.model_name))

    def init_game_setting(self):
        pass


    def train(self):
        self.epi = 0
        observ = self.env.reset()
        self.prev_observ = None

        while True:
            action, action_prob, x = self.make_action(observ, test=False)
            observ, reward, terminal, _ = self.env.step(action + 1)

            self.history.store(x, reward, action, action_prob)

            self.step_cnt += 1
            self.reward_accu += reward

            if terminal:
                self.epi += 1

                observ = self.env.reset()
                self.prev_observ = None

                self.update_network()

                self.history.reset()

                self.game_len.append(self.step_cnt)
                self.rewards.append(self.reward_accu)
                self.step_cnt = 0
                self.reward_accu = 0.0

                if self.epi % self.batch_size == self.batch_size - 1:
                    # save model
                    self.val_network.model.save_weights(os.path.join('models_pg',"{}_model_weight.hdf5".format(self.model_name)))
                    # reporting
                    msg = 'Episode {}, game length = {}, mean reward = {}'.format(self.epi, np.mean(self.game_len), np.mean(self.rewards))
                    self.log_file.write(msg + '\n')
                    print(msg)

                    self.game_len = []
                    self.rewards = []



    def make_action(self, observation, test=True):
        observ_proc = self.preprocess_frame(observation)
        x = observ_proc - self.prev_observ  if self.prev_observ is not None else np.zeros(observ_proc.shape)
        self.prev_observ = observ_proc

        x = np.expand_dims(x, axis=0)
        action_prob = self.val_network.model.predict(x, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=action_prob)[0]
        if test:
            action = np.argmax(action_prob)
            return action + 1
        else:
            return action, action_prob, x

    def preprocess_frame(self, I):
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        I = I.astype(np.float).ravel()
        return I.reshape((80,80,1))

    def __cal_discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def update_network(self):
        gradients = np.vstack(self.history.grad_buffer)
        rewards = np.vstack(self.history.reward_buffer)
        discount_rewards = self.__cal_discount_rewards(rewards)
        discount_rewards = (discount_rewards - np.mean(discount_rewards)) / np.std(discount_rewards)
        gradients *= discount_rewards

        observ_input = np.squeeze(np.vstack([self.history.observ_buffer]), axis=1)
        labels = self.history.prob_buffer + self.learning_rate * np.squeeze(np.vstack([gradients]))

        self.val_network.model.train_on_batch(observ_input, labels)

import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras
import random
from collections import deque
import pdb

class ACNetwork:
    def __init__(self, params, sess):
        self.action_size = params['action_size']
        self.learning_rate = params['learning_rate']
        self.observ_size = params['observ_size']
        self.build_actor_model()
        self.build_critic_model()

        # Initialize variables
        self.sess = sess
        self.sess.run(tf.initialize_all_variables())


    def __build_actor(self):
        state_input = Input(shape=self.observ_size)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.action_size[0], activation='relu')(h3)
        model = Model(input=state_input, output=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return state_input, model

    def __build_critic(self):
        state_input = Input(shape=self.observ_size)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=[self.action_size[0]])
        action_h1    = Dense(48)(action_input)

        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)

        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return state_input, action_input, model

    def build_actor_model(self):
        self.actor_state_input, self.actor_model = self.__build_actor()
        _, self.target_actor_model = self.__build_actor()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size[0]])
        actor_model_weights = self.actor_model.trainable_weights
        # pdb.set_trace()
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
    
    def build_critic_model(self):
        self.critic_state_input, self.critic_action_input, self.critic_model = self.__build_critic()
        _, _, self.target_critic_model = self.__build_critic()

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) 
    
    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)
    
    def update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_actor_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_actor_model.set_weights(critic_target_weights) 







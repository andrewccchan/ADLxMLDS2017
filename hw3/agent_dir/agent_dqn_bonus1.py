from agent_dir.agent import Agent
from network import QNetwork
from utility import ReplayMemory, SerialNumberManager
import numpy as np
import tensorflow as tf
import pdb
import os
import random


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        
        # tf. session 
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))

        network_params = dict(
            action_size = env.get_action_space().n,
            learning_rate_minimum = args.learning_rate_min,
            learning_rate = args.learning_rate,
            learning_rate_decay_step = args.learning_rate_decay_step,
            learning_rate_decay = args.learning_rate_decay,
            do_duel_q = args.duel_q
        )

        memory_params = dict(
            memory_size = args.memory_size,
            action_size = env.get_action_space().n,
            observ_size = (84, 84, 4),
            batch_size = args.batch_size
        )

        self.q_network = QNetwork(network_params, self.sess)
        self.memory = ReplayMemory(memory_params)
        self.max_steps = args.max_steps

        # number of steps before learning starts
        self.learning_start = args.learning_start

        # random action
        self.rand_epsi = args.rand_epsi
        self.epsi_begin = 1
        self.epsi_final = args.epsi_final
        self.epsi_end_time = 8000000
        self.step = 0

        self.discount = args.discount
        self.target_update_frequency = args.target_update_frequency
        self.train_frequncy = args.train_frequency
        self.testing_frequency = args.testing_frequency


        # self.sess.run(tf.global_variables_initializer())
        
        self.total_loss = 0.0
        self.total_q = 0.0
        self.total_reward = 0.0
        self.update_cnt = 0

        self.ep_reward = 0.0
        self.ep_reward_list = []
        self.ep_cnt = 0

        # double q and duel q option
        self.do_double_q = args.double_q
        self.do_duel_q = args.duel_q

        self.model_saver = tf.train.Saver()

        self.serial_man = SerialNumberManager('./model', './log', './params', './out')
        self.model_path, self.params_path, self.out_path, self.log_path = self.serial_man.get_paths('14-duel_q')
        print('[INFO]:\n model_path={}\n params_path={}\n'.format(self.model_path, self.params_path))
        self.log_file = open(self.log_path, 'w')
        
        self.test_mode = args.test_dqn


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        pass
        
        
    def train(self):
        """
        Implement your training algorithm here
        """
        # Initial observation
        observ = self.env.reset()
            
        for self.step in range(self.max_steps):
            if self.step == self.learning_start:
                self.total_reward = 0.0
                self.total_q = 0.0
                self.total_loss = 0.0
                self.update_cnt = 0
                self.ep_reward = 0.0
                self.ep_reward_list = []
                #self.ep_cnt = 0
            
            # Decide an action
            action = self.make_action(observ)
            
            # Act
            observ, reward, terminal, _ = self.env.step(action)

            # Update replay memory
            self.memory.store(observ[:,:,-1], reward, action, terminal)

            # Update netowrks
            if self.step > self.learning_start:
                if self.step % self.train_frequncy == 0:
                    # Gradient descent
                    #print('updating network')
                    self.update_network()
                if self.step % self.target_update_frequency == self.target_update_frequency - 1:
                    #print('updating target network')
                    self.q_network.update_target_network()
            
            # model logging
            self.total_reward += reward
            
            # check terminal
            if terminal:
                observ = self.env.reset()
                self.ep_reward_list.append(self.ep_reward)
                self.ep_reward = 0.0
                self.ep_cnt += 1
            else:
                self.ep_reward += reward

            if self.step > self.learning_start and self.step % self.testing_frequency == 0:
                avg_loss = self.total_loss / self.update_cnt
                avg_q = self.total_q / self.update_cnt
                avg_reward = self.total_reward / self.testing_frequency
                msg_str = 'step = {}, avg. loss={}, avg. q={}, avg. reward={}'.format(self.step, avg_loss, avg_q, avg_reward)
                print(msg_str)
                self.log_file.write(msg_str + '\n')

                self.total_reward = 0.0
                self.total_q = 0.0
                self.total_loss = 0.0
                self.update_cnt = 0

                self.model_saver.save(self.sess, self.model_path, global_step=self.step)

            if self.step > self.learning_start and self.ep_cnt % 100 == 99 and terminal:
                try:
                    avg_ep_reward = np.mean(self.ep_reward_list)
                except:
                    avg_ep_reward = 0.0
                msg_str = 'ep={}, avg. reward = {}'.format(self.ep_cnt, avg_ep_reward)
                print(msg_str)
                self.log_file.write(msg_str + '\n')
                self.ep_reward = 0.0
                self.ep_reward_list = []

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        # if test:
        #     epsi = self.rand_epsi
        # elif self.step < self.learning_start:
        #     epsi = self.epsi_begin
        # elif self.step > self.epsi_end_time:
        #     epsi = self.epsi_final
        # else:
        #     slope = (self.epsi_final - self.epsi_begin) / (self.epsi_end_time - self.learning_start)
        #     epsi = self.epsi_begin + (self.step - self.learning_start) * slope
        if self.test_mode:
            epsi = self.rand_epsi
        else:
            epsi = (self.epsi_final + max(0., (self.epsi_begin - self.epsi_final) \
                    * (self.epsi_end_time - max(0., self.step - self.learning_start)) / self.epsi_end_time))
        do_random_action = random.random() < epsi
        if do_random_action:
            return self.env.get_random_action()
        else:
            return np.asscalar(self.q_network.q_action.eval({self.q_network.input: [observation]}, session=self.sess))
        

    
    def update_network(self):
        # sample training data from replay memory
        prev_state, actions, rewards, next_state, terminals = self.memory.sample()
        
        # If using double DQN
        if self.do_double_q:
            # Decide action by action Q network
            q_actions = self.q_network.q_action.eval({self.q_network.input: next_state}, session=self.sess)
            # Determine expected future reward by value Q network
            feed_dict = {
                self.q_network.target_input: next_state,
                self.q_network.selection_idx: [[idx, val] for idx, val in enumerate(q_actions)]
            }
            future_qval = self.q_network.selected_q_val.eval(feed_dict, session=self.sess)
            target_q_val = (1. - terminals) * self.discount * future_qval + rewards

        else:
            future_qval = self.q_network.target_qval.eval({self.q_network.target_input: next_state}, session=self.sess)
            max_qval = np.max(future_qval, axis=1)
            target_q_val = (1. - terminals) * self.discount * max_qval + rewards
        
        feed_dict = {
            self.q_network.target_q_t: target_q_val,
            self.q_network.action: actions,
            self.q_network.input: prev_state,
            self.q_network.learning_rate_step: self.step
        }
        _, train_q, train_loss = self.sess.run([self.q_network.train_op, self.q_network.qval, self.q_network.loss], feed_dict=feed_dict)

        self.total_loss += train_loss
        self.total_q += np.mean(train_q)
        self.update_cnt += 1





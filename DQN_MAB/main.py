from produce_dataset import generateSet
from RL_brain import DeepQNetwork
import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

from run_me import DQN_test
import matplotlib.pyplot as plt
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

train_set_size = 10000
tf.reset_default_graph()

def run_recommend(iter_batch,env,RL):
    step = 0
    for episode in range(20):
        # initial observation
        chosen_person = np.random.randint(low=1,high=train_set_size,size=1)
        # print(chosen_person)
        # print(env.preference[chosen_person,:])
        s = env.preference[chosen_person,:]
        s = np.array(s).flatten()
        for i in range(300):
            # fresh env
            # to do

            # RL choose action based on observation
            action = RL.choose_action(s)

            # RL take action and get next observation and reward
            reward = env.step(s=chosen_person,action=action)

            RL.store_transition(s, action, reward, s_ = np.array(env.preference[chosen_person,:]).flatten() )

            if (step > 0) and (step % 5 == 0):
                RL.learn()

            # swap observation
            #observation = s_

            # break while loop when end of this episode

            step += 1
        # ccr[iter_batch*20+episode] = start_testing()
        # print('ccr', iter_batch*20+episode, ' = ', ccr[iter_batch*20+episode])

        # end of iter
        #print('iter over')




if __name__ == '__main__':
    env = generateSet()
    #test_agent = DQN_test()
    #tf.reset_default_graph()
    RL = DeepQNetwork(env.MAB, env.MAB,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=300,
                      memory_size=2000,
                      # output_graph=True
                      )
    ccr = np.zeros(800)
    ccr40 = np.zeros(800)
    for iter_batch in range(40):
        print('num of iter: ',iter_batch)
        run_recommend(iter_batch,env,RL)

        #test_agent.start_testing()

        test_s = random.randint(0, train_set_size)
        print('%%%%%%%%%%%%%%%%%%%%%%%train')
        print(test_s)
        print('the preference of the chosen person for ten news types: ', env.preference[test_s, :])
        s=env.preference[test_s, :]
        s = np.array(s).flatten()
        print('highest preference: ',RL.choose_action(s))
        print('%%%%%%%%%%%%%%%%%%%%%%%test')
        ss = np.array([.1,.2,.3,.4,.4,.3,.2,.1,.9,.1])
        test = ss.flatten()


        #RL.savenet()
        # ccr[iter_batch] = start_testing(RL)
        # print('!!!!!!!!!!!!!!!!!ccr800', iter_batch, ' = ', ccr[iter_batch])
        #print(test)
    x = 1
    RL.savenet()
    print('model saved')
    ccr_x = range(800)
    plt.plot(ccr_x,ccr)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title('testing ccr')



    #np.save('save/w1', )

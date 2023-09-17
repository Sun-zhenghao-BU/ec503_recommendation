from environmnet import step
# from environmnet import state_trans
import numpy as np
import tensorflow as tf2
import os
import warnings
from RL_tar import DeepQNetwork
from dataset import get_data
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf = tf2.compat.v1
tf.disable_v2_behavior()
warnings.filterwarnings("ignore")



num_people_onegroup=1000
num_movie=10
num_both_like=5
type_people=5
shuffle_rate = 0.6
iter_num = 100
total_people = num_people_onegroup*type_people
recommend_time = 1000

def state_trans(action, omit,s,ts):
    if action == omit:
        s_ = copy.deepcopy(s)
        s_[omit]=1
        # s_omit = s_[np.where(ts == num_both_like-1)]

        s_[np.where(ts== num_both_like-1)]=0
        for i in range(num_movie):
            if ts[i] !=0:
                ts[i] = ts[i]+1
        ts[omit] = 1
        omit = np.where(ts == num_both_like)
        ts[omit] = 0
        return s_,omit,ts
    else:
        s_ = s
        # s_[action] =  -1
        return s_,omit,ts


if __name__ == '__main__':
    pref, timestamp = get_data(num_people_onegroup=num_people_onegroup,num_movie=num_movie,
                               num_both_like=num_both_like,type_people=type_people,shuffle_rate = shuffle_rate)
    RL = DeepQNetwork(n_actions = num_movie,
                      n_features = num_movie,
                      learning_rate=0.001,#0.005,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=4,   # todo
                      memory_size=1,#2000
                      batch_size = 1#10
                      # output_graph=True
                      )
    for iter in range(iter_num):
        print('iter',iter)
        count = 0
        # print('##################')
        chosen_person = np.random.randint(low=1, high=total_people, size=1)
        ts = timestamp[chosen_person,:]
        ts = np.array(ts).flatten()

        choose_omit = np.where(ts == num_both_like)
        choose_omit = list(choose_omit)
        s = pref[chosen_person, :]
        s = np.array(s).flatten()
        s[choose_omit]=0
        ts[choose_omit] = 0
        # print('choose_omit',choose_omit)
        good_result = 0
        if iter == 10:
            x = 1
        for time in range(recommend_time):
            # print('s[choose_omit]33333', s[choose_omit])
            action,actions_value = RL.choose_action(s)

            reward,correct = step(action=action,omit = choose_omit,actions_value=actions_value,s=s)

            if correct > 0 :
                good_result+=1
            s_,choose_omit,ts =  state_trans(action= action,omit = choose_omit,s = s,ts = ts)
            transition = RL.store_transition(s = s, a = action, r = reward, s_=s_)
            # print('#################')
            actions_value_next = RL.Q_s_(s_)
            target = actions_value_next.copy()
            target[0,action] = reward + np.max(actions_value_next)
            print('#############################')
            print('s',s)
            print('s_',s_)
            print('actions_value',actions_value)
            print('target',target)
            print('omit',choose_omit)
            print('action',action)
            with tf.Session() as sess:
                print('loss1',sess.run(tf.squared_difference(target, actions_value)))
                print('loss',sess.run(tf.reduce_mean(tf.squared_difference(target, actions_value))))
            # if (count > 0) and (count % 5 == 0):
            #     RL.learn()
            #     print('!!!!!!!!!!!!learn!!!!!!!!!!!!!!!!!!!!')
            RL.learn(transition)
            count += 1
            s = s_
            num_zero = 0
            for i in range(len(s)):
                if s[i] == 0:
                    num_zero +=1
            if num_zero>1:
                continue
            print('game over')
            break
        print('good_result',good_result)
        print('total recommend',count)


# tensorboard --logdir=logs

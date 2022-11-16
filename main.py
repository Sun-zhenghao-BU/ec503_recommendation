from produce_dataset import generateSet
from RL_brain import DeepQNetwork
import numpy as np


def run_recommend():
    step = 0
    for episode in range(5):
        # initial observation

        chosen_person = np.random.randint(low=1,high=10,size=1)
        # print(chosen_person)
        # print(env.preference[chosen_person,:])
        s = env.preference[chosen_person,:]
        s = np.array(s).flatten()
        for i in range(100):
            # fresh env
            # to do

            # RL choose action based on observation
            action = RL.choose_action(s)

            # RL take action and get next observation and reward
            reward = env.step(s=chosen_person,action=action)

            RL.store_transition(s, action, reward, s_ = np.array(env.preference[chosen_person,:]).flatten() )

            if (step > 50) and (step % 5 == 0):
                RL.learn()

            # swap observation
            #observation = s_

            # break while loop when end of this episode

            step += 1

    # end of game
    print('game over')

if __name__ == '__main__':
    env = generateSet()
    RL = DeepQNetwork(env.MAB, env.MAB,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    for i in range(100):
        run_recommend()
        test_s = np.random.randint(low=1, high=10, size=1)
        print('%%%%%%%%%%%%%%%%%%%%%%%train')
        print(test_s)
        print('the preference of the chosen person for ten news types: ', env.preference[test_s, :])
        s=env.preference[test_s, :]
        s = np.array(s).flatten()
        print('highest preference: ',RL.choose_action(s))
        print('%%%%%%%%%%%%%%%%%%%%%%%test')
        ss = np.array([.1,.2,.3,.4,.4,.3,.2,.1,.9,.1])
        test = ss.flatten()
        #print(test)


    #np.save('save/w1', )

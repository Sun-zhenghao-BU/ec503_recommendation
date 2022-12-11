import numpy as np

# def step(action,omit,actions_value,s):
#     if action == omit:
#         correct = 10
#     else:
#          correct = -1
#     cosine_similarity = np.dot(actions_value, s) / (np.linalg.norm(actions_value, 2) * np.linalg.norm(s, 2))
#     s_ave = 0
#     for i in range(len(s)):
#         s_ave+=s[i]
#     s_ave = s_ave/len(s)
#     # print('actions_value####',actions_value)
#     reward = (actions_value[omit] - s_ave)
#     if reward >0:
#         reward = reward
#     print('reward####', reward)
#     print('s',s)
#     print('actions_value',actions_value)
#     print('omit',omit)
#     # reward = correct
#     return reward,correct

def step(action,omit,actions_value,s):
    if action == omit:
        correct = 10
    else:
         correct = -1
    cosine_similarity = np.dot(actions_value, s) / (np.linalg.norm(actions_value, 2) * np.linalg.norm(s, 2))
    reward = 0
    for i in range(len(s)):
        if actions_value[omit]>actions_value[i]:
            reward += 1
        elif actions_value[omit]<actions_value[i]:
            reward-=1
    reward = reward/40
    if reward >0:
        reward = reward

    reward = correct

    # print('reward####', reward)
    # print('s',s)
    # print('actions_value',actions_value)
    # print('omit',omit)
    # reward = correct
    return reward,correct

# def step(action,omit,actions_value,s):
#     if action == omit:
#         correct = 5
#     else:
#          correct = -1
#     # cosine_similarity = np.dot(actions_value, s) / (np.linalg.norm(actions_value, 2) * np.linalg.norm(s, 2))
#     # reward = 0
#     # for i in range(len(s)):
#     #     if actions_value[omit]>actions_value[i]:
#     #         reward += 1
#     #     elif actions_value[omit]<actions_value[i]:
#     #         reward-=1
#     #
#     # if reward >0:
#     #     reward = reward
#     # q_target = actions_value.copy()
#     # q_target[action] = reward+0.9 * np.max(q_target, axis=1)
#     reward = correct
#     # print('reward####', reward)
#     # print('s',s)
#     # print('actions_value',actions_value)
#     # print('omit',omit)
#     # reward = correct
#     return reward,correct

def state_trans(action,omit):
    if action == omit:
        satisfy = 1
    else:
        satisfy = 0

import itertools
import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout

def write_csv(filename, histo_to_write, delimiter=';', action_ratio=0.8, max_samp_by_user=5, max_state=100,
              max_action=50, nb_states=[], nb_actions=[]):
    '''
    From  a given historic, create a csv file with the format:
    columns : state;action_reward;n_state
    rows    : itemid&rating1 | itemid&rating2 | ... ; itemid&rating3 | ... | itemid&rating4; itemid&rating1 | itemid&rating2 | itemid&rating3 | ... | item&rating4
    at filename location.

    Parameters
    ----------
    filename :        string
                      path to the file to be produced
    histo_to_write :  List(DataFrame)
                      List of the historic for each user
    delimiter :       string, optional
                      delimiter for the csv
    action_ratio :    float, optional
                      ratio form which movies in history will be selected
    max_samp_by_user: int, optional
                      Nulber max of sample to make by user
    max_state :       int, optional
                      Number max of movies to take for the 'state' column
    max_action :      int, optional
                      Number max of movies to take for the 'action' action
    nb_states :       array(int), optional
                      Numbers of movies to be taken for each sample made on user's historic
    nb_actions :      array(int), optional
                      Numbers of rating to be taken for each sample made on user's historic

    Notes
    -----
    if given, size of nb_states is the numbller of sample by user
    sizes of nb_states and nb_actions must be equals

    '''
#     with open(filename, mode='w') as file:
#         f_writer = csv.writer(file, delimiter=delimiter)
#         f_writer.writerow(['state', 'action_reward', 'n_state'])
#         for user_histo in histo_to_write:
#             states, actions = sample_histo(user_histo, action_ratio, max_samp_by_user, max_state, max_action,
#                                                 nb_states, nb_actions)
#             for i in range(len(states)):
#                 # FORMAT STATE
#                 state_str = '|'.join(states[i])
#                 # FORMAT ACTION
#                 action_str = '|'.join(actions[i])
#                 # FORMAT N_STATE
#                 n_state_str = state_str + '|' + action_str
#                 f_writer.writerow([state_str, action_str, n_state_str])
#
# write_csv()

# with open(filename, mode='w') as file:
#     f_writer = csv.writer(file, delimiter=delimiter)
#     f_writer.writerow(['state', 'action_reward', 'n_state'])
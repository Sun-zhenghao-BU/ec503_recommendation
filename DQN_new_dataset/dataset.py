# In this dataset, we set the number of people is 1000, and we have 100 movies
import numpy as np
import random
def get_data(num_people_onegroup=1000, num_movie=40, num_both_like=20,  type_people=3 ,shuffle_rate = 0):
    preference = np.zeros((type_people*num_people_onegroup, num_movie))
    timestamp = np.zeros((type_people*num_people_onegroup,num_movie))
    for k in range(type_people):
        for j in range(num_both_like):
            while True:
                index = np.random.randint(1, num_movie)
                if preference[k*num_people_onegroup,index] == 0:
                    for i in range(num_people_onegroup):
                        preference[k*num_people_onegroup+i, index] =1
                        timestamp [k*num_people_onegroup+i, index] = j+1
                    break
    # shuffle
    for i in range(type_people*num_people_onegroup):
        if np.random.randint(1,100)/100< shuffle_rate:
            shuffle_index_off = -1
            timestamp_off = -1
            while True:
                shuffle_index_off = np.random.randint(1, num_movie)
                if preference[i, shuffle_index_off] == 1:
                    preference[i, shuffle_index_off] = 0
                    timestamp_off = timestamp[i, shuffle_index_off]
                    timestamp[i, shuffle_index_off] = 0

                    break
            while True:
                shuffle_index_in = np.random.randint(1, num_movie)
                if shuffle_index_in !=  shuffle_index_off and preference[i,shuffle_index_in] == 0:
                    preference[i, shuffle_index_in] = 1
                    timestamp[i, shuffle_index_in] = timestamp_off
                    break


    return preference,timestamp



pref,timestamp = get_data()
x = 1
print(pref)



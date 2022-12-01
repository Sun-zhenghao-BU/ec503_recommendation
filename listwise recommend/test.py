import numpy as np
import pandas as pd
import csv
print('run test.py')
# # a = np.array([i for i in range(10)])
# # print(a)
# csv_file = 'Preprocess_MIND/embed_net_input'
# with open(csv_file, 'w+', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile, dialect='excel')
#     datapath = 'Preprocess_MIND/csv_trans'
#     itempath = 'Preprocess_MIND/item_csv'
#     data = pd.read_csv(datapath, sep=' ',
#                        names=['userId', 'itemId', 'rating', 'timestamp'])
#     movie_titles = pd.read_csv(itempath, sep=',', names=['itemId', 'itemName'],
#                                usecols=range(2), encoding='latin-1')
#     data.merge(movie_titles, on='itemId', how='left')
#     print(type(data))
#     writer.writerows(data)

# print(np.random.choice(10))


datapath = 'Preprocess_MIND/csv_trans'
itempath = 'Preprocess_MIND/item_csv'
data = pd.read_csv(datapath, sep=' ',
                   names=['userId', 'itemId', 'rating', 'timestamp'])
movie_titles = pd.read_csv(itempath, sep=',', names=['itemId', 'itemName'],
                           usecols=range(2), encoding='latin-1')
data = data.merge(movie_titles, on='itemId', how='left')

# preprocess
data = data.sort_values(by=['timestamp'])
# make them start at 0
data['userId'] = data['userId'] - 1
data['itemId'] = data['itemId'] - 1
user_count = data['userId'].max() + 1
movie_count = data['itemId'].max() + 1
user_movies = {}  # list of rated movies by each user
for userId in range(user_count):
    user_movies[userId] = data[data.userId == userId]['itemId'].tolist()
user_movies_writer = open('test_data/user_movies')
print(type(user_movies))
user_movies_writer.writelines([user_movies[i] for i in range(user_count)])
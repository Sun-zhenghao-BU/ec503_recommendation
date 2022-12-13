 # EC503_Recommendation System

 ## Zhenghao Sun, Shu Yang, Shenyou Fan, Quan Pham

 ### 1.Introduction
With the rapid development of computer technology and the rise of data learning, all aspects of our daily lives are becoming inseparable from all kinds of
data. Personalized recommendation systems make use of millions of data, allowing computers to learn the relationships between seemingly unrelated data and
gain insight into our preferences and choices. In this paper, we will investigate and implement our own recommendation system using reinforcement learning
methods such as Q-learing, Deep Q Network (DQN), and Actor Critic. We will also write reward functions for different algorithms with their characteristics
and compare the impact of different reward functions on different methods.

 ### 2.Qucik Start
 - ***DQN_MAB***:  
We implement the DQN to solve the recommendation based on the genre of the movie
 - ***DQN_new***:   
We implement the DQN to slove the recommendation based on the rating history of the cutomer
 - ***Q-learning_MAB***:  
We implement Q-learning to slove the multiple-armed bandit problem with e-greedy algoritm
 - ***list_wise_new***:  
We implement DQN with actor-critic structure and embedding preprocess to solve the list-wise recommendation task

 ### 3.Environment Setup
python 3.9
After running all the file before we have to setup our enivironment first. They can be activated with:

```
pip install -r requirements.txt
```
Github link: 
https://github.com/ShuYangConlany/ec503_recommendation

MIND dataset link:
https://msnews.github.io/

ML-100k dataset link:
https://github.com/SudeshGowda/ml-100k-dataset

Data process file: 
ec503_recommendation/list_wise_new/preprocess.py

supporting code:
https://github.com/luozachary/drl-rec

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:11:38 2019

@author: krithika
"""

import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
g_steps = 500
score_req = 50
initg=10000

def init_population():
    training_data=[]
    scores=[]
    accepted_scores=[]
    for _ in range(initg):
        score=0
        memory=[]
        prev_obs=[]
        for _ in range(g_steps):
            action=random.randrange(0,2)
            obs,reward,done,info=env.step(action)
            if len(prev_obs)>0:
                memory.append([prev_obs,action])
            prev_obs=obs
            score+=reward
            if done:
                break
        if(score>=score_req):
            accepted_scores.append(score)
            for data in memory:
                if data[1]==1:
                    output=[0,1]
                elif data[1]==0:
                    output=[1,0]
                training_data.append([data[0],output])
                
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data
t_d=init_population()

X = np.array([i[0] for i in t_d]).reshape(-1,len(t_d[0][0]),1)
Y = np.array([i[1] for i in t_d])

model=Sequential()
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense((256)))
model.add(Activation("relu"))
model.add(Dropout(0.6))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.7))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.6))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.hdf5', 
                               verbose=1, save_best_only=True)
hist = model.fit(X,Y, epochs=5,callbacks=[checkpointer],
          verbose=1, shuffle=True)


scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(g_steps):
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_req)
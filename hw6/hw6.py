import numpy as np
import keras
import sys
from keras.layers import Input, Embedding, Flatten, Add, Dot, Concatenate, Dense
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import pickle
def get_model(n_users, n_items, n_occ, n_zip, latent_dim = 10):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_occ = Input(shape=[1])
    user_zip = Input(shape=[1])
    user_Info = Input(shape=[2])
    item_Info = Input(shape=[18])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    #user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    #user_bias = Flatten()(user_bias)
    #item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    #item_bias = Flatten()(item_bias)
    
    user_occ_vec = Embedding(n_occ, 20, embeddings_initializer='random_normal')(user_occ)
    user_occ_vec = Flatten()(user_occ_vec)
    user_zip_vec = Embedding(n_zip, 20, embeddings_initializer='random_normal')(user_zip)
    user_zip_vec = Flatten()(user_zip_vec)
    
    merge_vec = Concatenate()([user_vec, item_vec, user_occ_vec, user_zip_vec, user_Info, item_Info])
    hidden = Dense(500, activation='relu')(merge_vec)
    hidden = Dropout(0.1)(hidden)
    hidden = Dense(250, activation='relu')(hidden)
    hidden = Dropout(0.1)(hidden)
    hidden = Dense(125, activation='relu')(hidden)
    output = Dense(1)(hidden)
    model = keras.models.Model([user_input, item_input, user_occ, user_zip, user_Info, item_Info], output)
    model.compile(loss='mse', optimizer='adam')
    return model

user_len = 6040
item_len = 3688
latent_dim = 10
n_occ = 21
n_zip = 3439
model = get_model(user_len, item_len,  n_occ, n_zip, latent_dim)

model.load_weights('strongbest.h5')


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
	        return pickle.load(f)

userIdMapIdx = load_obj('userIdMapIdx')
userInfo_gender = load_obj('userInfo_gender') 
userInfo_age = load_obj('userInfo_age') 
userInfo_occupation = load_obj('userInfo_occupation') 
zipcode_index = load_obj('zipcode_index') 
Genres_vec = load_obj('Genres_vec') 
movieIdMapIdx = load_obj('movieIdMapIdx') 

testf = open(sys.argv[1]+'test.csv','r')
testf.readline()
x_test = []
for line in testf:
    x_test.append(line.split(','))
x_test = np.array(x_test)
x_test_user = x_test[:, 1].astype(int)
x_test_item = x_test[:, 2].astype(int)

numT = 100336
x_test_userInfo = []
for i in range(numT):
    temp = []
    ti = userIdMapIdx[x_test_user[i]]
    temp.append(userInfo_gender[ti])
    temp.append(userInfo_age[ti])
    temp.append(userInfo_occupation[ti])
    temp.append(zipcode_index[ti])
    x_test_userInfo.append(temp)
x_test_userInfo = np.array(x_test_userInfo)
x_test_userInfo = x_test_userInfo.astype(float)
age = x_test_userInfo[:,1]
nor_age = (age - np.mean(age))/np.std(age)
x_test_userInfo[:,1] = nor_age 


numT = 100336
Genres_len = 18
x_test_movie = np.zeros((numT, Genres_len))
for i in range(numT):
    x_test_movie[i] = Genres_vec[  movieIdMapIdx[  x_test_item[i]  ]  ]
    
res = model.predict([x_test_user, x_test_item, x_test_userInfo[:,2], 
                    x_test_userInfo[:,3], x_test_userInfo[:,:2],x_test_movie])
with open(sys.argv[2], 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, v in  enumerate(res):
        f.write('%d,%f\n' %(i+1, v))

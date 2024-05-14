import pandas as pd
from keras import backend as K
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def root_mean_square_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

  
filelocation = "C:\\Daten\\Uni\\Bachelorarbeit\\03 - Programmierung\\restaurants_ratings\\restaurants_ratings.csv"
dataset = pd.read_csv(filelocation, usecols = ['idYelp', 'rating', 'idUser'], engine ='python')

dataset.idUser = dataset.idUser.astype('category').cat.codes.values
dataset.idYelp = dataset.idYelp.astype('category').cat.codes.values


train, test = train_test_split(dataset, test_size=0.2, random_state=42)

n_users = dataset.idUser.unique()
n_restaurants = dataset.idYelp.unique()

# Number of Embeddings -> 1, 10 or 50
n_embeddings = 1

# Number of Neurons -> 1, 10 or 50
n_units =  1

# Restaurant Input Layer
restaurant_input = Input(shape=[1], name="restaurant-Input", dtype='int64')
# Restaurant Embedding Layer
restaurant_embedding = Embedding(len(n_restaurants)+1, n_embeddings, name="restaurant-Embedding")(restaurant_input)
restaurant_vec = Flatten(name="Flatten-restaurants")(restaurant_embedding)
# Hidden Layer / Restaurant Dense Layer 1 and 2 with n_units Neurons
restaurant_dense_1 = Dense(n_units, activation="sigmoid")(restaurant_vec)
restaurant_dense_2 = Dense(n_units, activation="sigmoid")(restaurant_dense_1)

# User Input Layer
user_input = Input(shape=[1], name="User-Input", dtype='int64')
# User Embedding Layer
user_embedding = Embedding(len(n_users)+1, n_embeddings, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)
# Hidden Layer / User Dense Layer 1 and 2 with n_units Neurons
user_dense_1 = Dense(n_units, activation="sigmoid")(user_vec)
user_dense_2 = Dense(n_units, activation="sigmoid")(user_dense_1)

# Variant 1: Without Dense Layer
output = Dot(name="Dot-Product", axes=1)([user_vec, restaurant_vec])

# Variant 2: 1 Dense Layer
#output = Dot(name="Dot-Product", axes=1)([user_dense_1, restaurant_dense_1])

# Variant 3: 2 Dense Layers
#output = Dot(name="Dot-Product", axes=1)([user_dense_2, restaurant_dense_2])

model = Model([user_input, restaurant_input], output)
model.compile('adam', loss = root_mean_square_error)
model.summary()
history = model.fit([train.idUser, train.idYelp], train.rating, epochs=10,
                    validation_data=([test.idUser, test.idYelp], test.rating))
result = model.predict([test.idUser, test.idYelp]).view()
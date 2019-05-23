#%%
import keras
import numpy as np 
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Reshape, LSTM, GRU, Dropout
from keras.layers import BatchNormalization, Activation, ReLU  
from keras.activations import relu, sigmoid

from utils import *

#%%
# load dataset
filecsv = 'Data/Wednesday-workingHours.pcap_ISCX.csv'
dataset, target = readTrainCSV(filecsv, fea_sel=0)

#%% split data
X_train, X_test, y_train, y_test = train_test_split(
                dataset, target, test_size=0.2, random_state=0)

X_train = np.array(X_train).astype(np.float)
X_test = np.asarray(X_test).astype(np.float)
y_train = np.asarray(y_train).astype(np.float)
y_test = np.asarray(y_test).astype(np.float)


#%%
NUM_CLASSES = len(set(y_train))
NUM_FEATURES = X_train.shape[1]

#%% drop data to testt

# SIZE = 1000

# X_train = X_train[:SIZE]
# y_train = y_train[:SIZE]

# TEST_SIZE = 100

# X_test = X_test[:TEST_SIZE]
# y_test = y_test[:TEST_SIZE]

#%% convet label to one hot 
def one_hot_converter(target, num_class):
    one_hot_vect = np.zeros((target.shape[0], num_class))
    one_hot_vect[np.arange(target.shape[0]), target] = 1
    return one_hot_vect
y_train_ = np.asarray(y_train).astype(np.int)
y_train_ = one_hot_converter(y_train_, 6)

y_test_ = np.asarray(y_test).astype(np.int)
y_test_ = one_hot_converter(y_test_, 6)


#%% RNN model 

model = Sequential()
model.add(Reshape((NUM_FEATURES,1), input_shape=(NUM_FEATURES,)))
model.add(Dense(64, input_shape=(1,NUM_FEATURES)))
model.add(BatchNormalization())
model.add(ReLU(max_value=10))
model.add(LSTM(32, return_sequences=True))
model.add(Dense(32, activation='relu'))
# mdoel.add(ReLU(max_value=10))
model.add(LSTM(16))
model.add(Dense(6, activation = 'softmax'))

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-5)
model.compile(optimizer=opt,
                loss='mean_squared_error', metrics=['accuracy'])


#%%
model.summary()
batch_size = 128
epochs = 1000

model.fit(X_train, y_train_,
              batch_size=batch_size,
              epochs=epochs,
                validation_data=(X_test, y_test_),
              shuffle=True)

#%% save model
model_path = './models/'
model_name = 'rnn_model_test.h5'
model.save(model_path + model_name)

#%% testing
y_pred = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_pred, y_test))
# print("Accuracy2: ", accuracy_score(y_pred_, y_test_))

#%% load model
from keras.models import load_model
model = load_model(model_path + model_name)
scores = model.evaluate(X_test, y_test_, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
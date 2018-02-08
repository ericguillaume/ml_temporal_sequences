import json
import math

import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K



def get_relative_accuracy_pred_func(relative_accuracy_limit):
    def relative_accuracy_pred(y_true, y_pred): return K.mean(K.clip(K.sign(relative_accuracy_limit - (K.abs(y_pred - y_true) / y_true)), 0.0, 1.0)) # ERROR lambda function mais qui a besoin d'etre nomme par keras/tensorflow
    return relative_accuracy_pred

def load_data():
    with open("bitcoin.json") as f:
        content = f.readlines()
        for line in content:
            data = json.loads(line)
            # print(type(data))
            # print(data.keys())
            #
            # price_usd = data["price_usd"]
            # print(type(price_usd))
            # print(len(price_usd)) # elements 0 are timestamp of days one after the other
            # for i in range(1, len(price_usd)):
            #     diff = price_usd[i][0] - price_usd[i-1][0]
            #     if diff <= 82260000 or diff >= 90000000:
            #         print(diff)

            volume_usd = data["volume_usd"]
            # print(type(volume_usd))
            # print(len(volume_usd))  # elements 0 are timestamp of days one after the other

            price_usd = np.array([x[1] for x in data["price_usd"]], dtype="float32")
            volume_usd = np.array([x[1] for x in data["volume_usd"]], dtype="float32")
            return np.column_stack((price_usd, volume_usd))


n_features = 2


np_data = load_data()
scaler = StandardScaler()
scaler = scaler.fit(np_data)
X_unscaled = np_data[0:-1]
y_unscaled = np_data[:-1,0]
np_data = scaler.transform(np_data)

X = np_data[0:-1]
y = np_data[:-1,0]
X = X.reshape(len(X), 1, n_features)
y = y.reshape(len(X), 1) # FACT: can be removed, the y output accept numpy arrays of size (1725) or of shape (1725, 1)

# configure network
n_batch = 1725
n_epoch = 20
n_neurons = 10

relative_accuracy_limit = 0.05


# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', get_relative_accuracy_pred_func(relative_accuracy_limit)])
print(model.summary())
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False, validation_data=(X, y))
	model.reset_states()

scores = model.evaluate(X, y, verbose=0, batch_size=n_batch) # ERROR evalutes set a batch_size default to 32, in my case I need 64
print(model.metrics_names)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f" % (scores[0]*100))
print("Accuracy prediction: %.2f" % (scores[2]))


#Xhat = model.predict(X, batch_size=n_batch)
#Xhat_unscaled = scaler.inverse_transform(Xhat)
#accuracy = np.mean(np.clip(np.sign(1 - (np.absolute(Xhat_unscaled - X_unscaled)/ accuracy_limit)), 0.0, 1.0))
#print("accuracy = {}".format(accuracy))



online_mode = False
if online_mode:
    # re-define the batch size
    n_batch = 1
    # re-define model
    new_model = Sequential()
    new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    new_model.add(Dense(1))


    # copy weights
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)

    # compile model
    new_model.compile(loss='mean_squared_error', optimizer='adam')


    # online forecast
    # for i in range(len(X)):
    # 	testX, testy = X[i], y[i]
    # 	testX = testX.reshape(1, 1, n_features)
    # 	yhat = new_model.predict(testX, batch_size=n_batch)
    # 	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))


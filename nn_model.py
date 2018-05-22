from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras import backend as K
import numpy as np
import json

## External variables
JSON_FILE = "nn_1t.json"
INPUT_LEN = 15

## Training variables
variables = {
    "train_epochs" : 5,
    "batch_size" : 32,
    "lstm_units" : [200, 200],
##  "dropout_rate" : [0.2, 0.2],
    "activation_final" : "softmax",
    "loss_function" : "binary_crossentropy",
    "learning_rate" : 0.1
}

class Network():
    def __init__(self):
        self.model = Sequential()
	## Expects an input matrix of length 5 with 4 elements each. Output length is 100 
        ## TODO Normalize input
        self.model.add(LSTM(variables["lstm_units"][0], input_shape=(INPUT_LEN,4), return_sequences=True))
        #self.model.add(Dropout(variables["dropout_rate"][0]))
        self.model.add(LSTM(variables["lstm_units"][1]))
        #self.model.add(Dropout(variables["dropout_rate"][1]))
        self.model.add(Dense(4, activation=variables["activation_final"]))

        sgd = SGD(lr= variables["learning_rate"], decay=1e-6, momentum=0.9, nesterov=True)
        ##rms = RMSprop(lr=variables["learning_rate"], decay=1e-6)
	## Review different types of optimizers, loss functions
        self.model.compile(optimizer=sgd,
		      loss=variables["loss_function"],
                      metrics=["accuracy"])

    def train(self, in_data, out_data):
        self.model.fit(in_data, out_data, epochs=variables["train_epochs"], batch_size=variables["batch_size"])

    def predict(self, in_data, out_data):
        self.output = self.model.predict(in_data, batch_size=variables["batch_size"]) 
        success = 0
        failure = 0
        for i in range(len(out_data)):
            if (self.output[i][3] > in_data[i][-1][3]) and (out_data[i][3] > in_data[i][-1][3]):
                success+=1
            if (self.output[i][3] < in_data[i][-1][3]) and (out_data[i][3] < in_data[i][-1][3]):
                success+=1
            else: 
                failure+=1
        rate = round((success/(success+failure))*100, 2)
        print("Rate: {}% Success: {} Failure: {} Transactions: {}".format(rate, success, failure, success+failure))
        return rate
        
class Evaluation():
    def __init__(self):
        pass    

if __name__ == "__main__":
    net = Network()

    ## Training data
    ## Load JSON and convert Numpy arrays
    with open(JSON_FILE) as f:
        data = json.loads(f.read())
    in_data = np.asarray([x[0] for x in data])
    out_data = np.asarray([x[1] for x in data])
    net.train(in_data, out_data)
    
    ## Predict sample
    net.predict(in_data, out_data)

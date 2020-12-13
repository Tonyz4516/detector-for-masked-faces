import os
import json
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle

# from util import get_normalized_data, y2indicator

def get_normalized_data(mode, Xtrain, sc = None, Ytrain = False,Xtest = False,Ytest = False):
    
    if mode == 'Classifiy':
#         sc = pickle.load(open('code/sc.pkl','rb'))
#         trainX = Xtrain.values
        trainX = sc.transform(Xtrain)
        return trainX
    else:
        scaler = StandardScaler()
        
        trainX = Xtrain.values
        y_train = Ytrain["label"].values
        testX = Xtest.values
        y_test = Ytest["label"].values
    

        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
        pickle.dump(scaler, open('code/sc.pkl','wb'))

 
        
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y_train.reshape(-1,1))
        y_nn = enc.transform(y_train.reshape(-1,1)).toarray()
        y_nn = np.float32(y_nn)

        y_nn_test = enc.transform(y_test.reshape(-1,1)).toarray()
        y_nn_test = np.float32(y_nn_test)

        return trainX,y_nn,testX,y_nn_test
    

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y



class TFNN:
    def __init__(self, savefile, D=None, K=None):
        self.savefile = savefile
        if D and K:

      # we can define some parts in the model to be able to make predictions
            self.build(D, K)

    def build(self,D,K):



    # define variables and expressions
        self.inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        self.targets = tf.placeholder(tf.float32, shape=(None,K), name='targets')
        self.W1 = tf.Variable(tf.random_normal([512, 300], stddev=0.03), name='W1')
        self.b1 = tf.Variable(tf.random_normal([300]), name='b1')
        self.W2 = tf.Variable(tf.random_normal([300, K], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([K]), name='b2')
    # variables must exist when calling this
    # try putting this line in the constructor and see what happens
        self.saver = tf.train.Saver({'W1': self.W1, 'b1': self.b1,"W2":self.W2,'b2':self.b2})

        hidden_out = tf.matmul(self.inputs, self.W1) + self.b1
        hidden_out = tf.nn.relu(hidden_out)

    # output layer
        y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, self.W2), self.b2))

        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)


        cost = -tf.reduce_mean(tf.reduce_sum(self.targets * tf.log(y_clipped)
                         + (1 - self.targets) * tf.log(1 - y_clipped), axis=1))

        self.predict_prop = tf.reduce_max(y_clipped, reduction_indices=[1])
        self.predict_lb = tf.argmax(y_clipped, 1)

        correct_prediction = tf.equal(tf.argmax(y_clipped, 1), tf.argmax(self.targets, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return cost


    def fit(self, X, Y, Xtest, Ytest):
        N, D = X.shape
        K = Y.shape[1]


    # hyperparams
        max_iter = 50
        lr = 0.05
        batch_sz = 1000
        n_batches = N // batch_sz

        cost = self.build(D,K)

        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)


        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            for i in range(max_iter):
                X, Y = randomize(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

                    session.run(train_op, feed_dict={self.inputs: Xbatch, self.targets: Ybatch})
                if i % 10 == 0:
                    acc_train = session.run(self.accuracy, feed_dict={self.inputs: Xbatch, self.targets: Ybatch})
                    print("Training Accuracy= %f" % acc_train)

            acc_test = session.run(self.accuracy, feed_dict={self.inputs: Xtest, self.targets: Ytest})
            print("Test Accuracy= %f" % acc_test)

            # save the model
            self.saver.save(session, self.savefile)


        # save dimensions for later
        self.D = D
        self.K = K




    def predict_label(self, X):


        with tf.Session() as session:
      # restore the model
            self.saver.restore(session, self.savefile)
            P = session.run(self.predict_prop, feed_dict={self.inputs: X})
            L = session.run(self.predict_lb, feed_dict={self.inputs: X})
        return P,L

    def predict_acc(self, X,Y):


        with tf.Session() as session:
      # restore the model
            self.saver.restore(session, self.savefile)

            acc_test = session.run(self.accuracy, feed_dict={self.inputs: X, self.targets: Y})
        return acc_test



    def save(self, filename):
        j = {
          'D': self.D,
          'K': self.K,
          'model': self.savefile
        }
        with open(filename, 'w') as f:
            json.dump(j, f)

    @staticmethod
    def load(filename):
        with open(filename) as f:
            j = json.load(f)
        return TFNN(j['model'], j['D'], j['K'])




# def main():
#     Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
#
#     model = TFLogistic("./tf.model")
#     model.fit(Xtrain, Ytrain, Xtest, Ytest)
#
#     # test out restoring the model via the predict function
#     # print("final train accuracy:", model.score(Xtrain, Ytrain))
#     # print("final test accuracy:", model.score(Xtest, Ytest))
#
#     # save the model
#     model.save("NN_model.json")
#
#     # load and score again
#     model = TFLogistic.load("NN_model.json")
#     print("final train accuracy (after reload):", model.predict(Xtrain, Ytrain)[1])
#     print("final test accuracy (after reload):", model.predict(Xtest, Ytest)[1])
#
#
# if __name__ == '__main__':
#     main()

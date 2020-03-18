import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st


class MLParser:
    def __init__(self, wordList, window_size):
        self.WINDOW_SIZE = window_size
        self.wordList = wordList
        self.ONE_HOT_DIM = len(self.wordList)
        self.train_op = None
        self.loss = None
        self.x = None
        self.X_train = None
        self.y_label = None
        self.Y_train = None
        self.W1 = None
        self.b1 = None


    def calculateNeighbors(self, sentences):
        data = []
        for sentence in sentences:
            for idx, word in enumerate(sentence):
                for neighbor in sentence[max(idx - self.WINDOW_SIZE, 0) : min(idx + self.WINDOW_SIZE, len(sentence)) + 1] : 
                    if neighbor != word:
                        data.append([word, neighbor])

        neighborDf = pd.DataFrame(data, columns = ["input", "label"])
        return neighborDf

    def createWordVector(self):
        word2int = {}
        for index, row in self.wordList.iterrows():
            word2int[row[0]] = index
        return word2int

    # function to convert numbers to one hot vectors
    def to_one_hot_encoding(self, data_point_index):
        one_hot_encoding = np.zeros(self.ONE_HOT_DIM)
        one_hot_encoding[data_point_index] = 1
        return one_hot_encoding

    def declareLoss(self, neighborDf, word2int):
        X = [] # input word
        Y = [] # target word

        for x, y in zip(neighborDf['input'], neighborDf['label']):
            X.append(self.to_one_hot_encoding(word2int[ x ]))
            Y.append(self.to_one_hot_encoding(word2int[ y ]))

        # convert them to numpy arrays
        self.X_train = np.asarray(X)
        self.Y_train = np.asarray(Y)

        # making placeholders for X_train and Y_train
        self.x = tf.placeholder(tf.float32, shape=(None, self.ONE_HOT_DIM))
        self.y_label = tf.placeholder(tf.float32, shape=(None, self.ONE_HOT_DIM))

        # word embedding will be 2 dimension for 2d visualization
        EMBEDDING_DIM = 2 

        # hidden layer: which represents word vector eventually
        self.W1 = tf.Variable(tf.random_normal([self.ONE_HOT_DIM, EMBEDDING_DIM]))
        self.b1 = tf.Variable(tf.random_normal([1])) #bias
        hidden_layer = tf.add(tf.matmul(self.x,self.W1), self.b1)

        # output layer
        W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, self.ONE_HOT_DIM]))
        b2 = tf.Variable(tf.random_normal([1]))
        prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

        # loss function: cross entropy
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_label * tf.log(prediction), axis=[1]))
        #return {"loss":loss,"x":x,"X_train":X_train,"y_label":y_label,"Y_train":Y_train}

    def train(self):
        # training operation
        self.train_op = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)

    def startSession(self, iteration):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init) 
        bar = st.progress(0)
        progtext = st.empty()
        prog = 0
        for i in range(iteration):
            # input is X_train which is one hot encoded word
            # label is Y_train which is one hot encoded neighbor word
            sess.run(self.train_op, feed_dict={self.x: self.X_train, self.y_label: self.Y_train})
            prog = i/iteration
            
            bar.progress(prog)
            if i % 1000 == 0:
                progtext.markdown('Iteration: '+'  \t'+str(i)+'  \n'+' Loss: '+'  \t'
                                + str(sess.run(self.loss, feed_dict={self.x: self.X_train, self.y_label: self.Y_train})))
        bar.empty()
        loss = sess.run(self.loss, feed_dict={self.x: self.X_train, self.y_label: self.Y_train})
        progtext.markdown("Loss: " +str(loss))
        with open("Data/loss.txt", "w") as lossfile:
            lossfile.write(str(loss))
        return sess

    def calculateVectors(self, sess):
        # Now the hidden layer (W1 + b1) is actually the word look up table
        vectors = sess.run(self.W1 + self.b1)
        return vectors

    def vectorToDf(self, vectors, wordList):
        w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
        w2v_df['word'] = wordList
        w2v_df = w2v_df[['word', 'x1', 'x2']]
        return w2v_df
    
    def reloadVectorToDf(self, vectorDf, wordList):
        vectorDf['word'] = wordList
        vectorDf = vectorDf[['word', 'x1', 'x2']]
        return vectorDf

    def plot(self, w2v_df, vectors):
        fig, ax = plt.subplots()

        for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
            ax.annotate(word, (x1,x2 ))

        PADDING = 1.0
        x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
        y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
        x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
        y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
        
        plt.xlim(x_axis_min,x_axis_max)
        plt.ylim(y_axis_min,y_axis_max)
        plt.rcParams["figure.figsize"] = (10,10)

        return plt
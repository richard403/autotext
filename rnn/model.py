#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/5 下午5:52

# @author: whx

import tensorflow as tf
import numpy as np
from config import SEQ_LEN





class RNN(tf.keras.Model):
    def __init__(self, numChars):
        super().__init__()
        self.numChars = numChars
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.numChars)


    @tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int64), tf.TensorSpec(shape=None, dtype=tf.bool)])
    def call(self, inputs, from_logits=False):
        batchSize = tf.shape(inputs)[0]
        inputs = tf.one_hot(inputs, depth=self.numChars)       # [batch_size, seq_length, num_chars]
        state = self.cell.get_initial_state(batch_size=batchSize, dtype=tf.float32)
        for t in range(SEQ_LEN):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)

    def predictPy(self, worlds, temperature=1.):
        batch_size, _ = tf.shape(worlds)
        logits = self.call(worlds, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()

        out = np.array([np.random.choice(self.numChars, p=prob[i, :])
                         for i in range(batch_size.numpy())])
        # out = np.array([tf.argmax(prob[i, :]) for i in range(batch_size.numpy())])

        return out

    @tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int64), tf.TensorSpec(shape=None, dtype=tf.float32)])
    def predictPyFunc(self, worlds, temperature=1.):
        batchSize = tf.shape(worlds)[0]
        logits = self.call(worlds, True)

        prob = tf.nn.softmax(logits / temperature)

        arr = tf.TensorArray(dtype=tf.int64, size=batchSize, dynamic_size=False)
        for i in tf.range(batchSize):
            pp = tf.slice(prob, [i, 0], [1, -1])
            pp = tf.squeeze(pp, axis=0)
            xx = tf.numpy_function(lambda x: np.random.choice(self.numChars, p=x), [pp], tf.int64)
            arr = arr.write(i, xx)

        out = arr.stack()
        # out = tf.argmax(prob, axis=1)
        # tf.print(out)
        return out

    @tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int64), tf.TensorSpec(shape=None, dtype=tf.float32)])
    def predict(self, worlds, temperature=1.):
        logits = self.call(worlds, True)
        prob = tf.nn.softmax(logits / temperature)
        cat = tf.random.categorical(tf.math.log(prob), 1)
        out = tf.squeeze(cat, axis=0)
        return out












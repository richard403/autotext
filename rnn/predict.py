#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/8 下午4:18

# @author: whx

import tensorflow as tf
import numpy as np
from config import SEQ_LEN
import argparse
from utils import configParse


def predict(world, keyDict, model, temperature=1.0, num=100):
    baseWorlds = set(keyDict.keys())
    diff = list(set(world) - baseWorlds)
    diff.sort()
    baseLen = len(baseWorlds)
    for index, _w in enumerate(diff):
        keyDict.update({_w: baseLen + index})
    worldIndexList = [keyDict[i] for i in world][len(world) - SEQ_LEN:]
    worldIndexList = np.expand_dims(np.asarray(worldIndexList), axis=0)

    idxDict = {}
    for k, v in keyDict.items():
        idxDict[v] = k
    print(worldIndexList)

    def modelPredict(inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = model.call(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()
        # out = np.array([np.random.choice(baseLen, p=prob[i, :])
        #                  for i in range(batch_size.numpy())])
        out = np.array([tf.argmax(prob[i, :]) for i in range(batch_size.numpy())])

        return out

    print(world)
    resultList = [world, ]
    for i in range(num):
        y_pred = modelPredict(worldIndexList, temperature)
        print(idxDict[y_pred[0]], end='', flush=True)
        resultList.append(idxDict[y_pred[0]])
        worldIndexList = np.concatenate([worldIndexList[:, 1:], np.expand_dims(y_pred, axis=1)], axis=1)

    return resultList

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=3)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--load_flag')
    args = parser.parse_args()

    confDict = configParse.ConfigParser.getConf2Dict('RNN')



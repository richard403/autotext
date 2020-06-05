#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/4 下午6:08

# @author: whx

import tensorflow as tf
import numpy as np
import os
import json


path = '/home/whx/workspace/work/python_code/autotext/data/train/1'
saveFile = '/home/whx/workspace/work/python_code/autotext/data/train.tfrecord'

rawTxt = open(path, encoding='utf-8').read().lower()
print(rawTxt)
char = ''.join(set(rawTxt))

with tf.io.TFRecordWriter(saveFile) as writer:
    feature = {
                'txt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rawTxt.encode('utf-8')])),
                'char': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char.encode('utf-8')])),
            }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


rawDataSet = tf.data.TFRecordDataset(saveFile)
featureDescription = {
            'txt': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'char': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
def _parseExample(exampleString):
    featureDict = tf.io.parse_single_example(exampleString, featureDescription)
    return featureDict['char']

for i in rawDataSet.map(_parseExample):
    print(i.numpy().decode('utf-8'))








#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/4 下午1:43

# @author: whx

import tensorflow as tf
import numpy as np
import os
import json

class DataLoader():
    def __init__(self, dataDir, saveFile, keyFile):
        self.dataDir = dataDir
        self.saveFile = saveFile
        self.fileList = ['%s/%s' % (dataDir, fileName) for fileName in os.listdir(self.dataDir)]
        self.keyDict = {}
        self.keys = []

        if keyFile is not None and os.path.isfile(keyFile):
            with open(keyFile, encoding='utf-8') as f1:
                string = f1.read()
                self.keyDict = json.loads(string)
                self.keys = list(set(self.keyDict.keys()))
                self.keys.sort()

    def writeFeature(self, keyFile=None):
        with tf.io.TFRecordWriter(self.saveFile) as writer:
            for _file in self.fileList:
                rawTxt = open(_file, encoding='utf-8').read().lower()

                rawSet = set(rawTxt)
                diffList = list(rawSet - set(self.keys))
                diffList.sort()
                keyLen = len(self.keyDict )
                for index, _key in enumerate(diffList):
                    self.keyDict .update({_key: keyLen + index})

                feature = {
                    'txt': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.keyDict [_w] for _w in rawTxt])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        with open(keyFile, 'w') as f2:
            f2.write(json.dumps(self.keyDict , ensure_ascii=False).encode('utf-8').decode('utf-8'))


    @staticmethod
    def readFeature(saveFile, seqLen):
        rawDataSet = tf.data.TFRecordDataset(saveFile)
        featureDescription = {
            'txt': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=-1),
        }

        def _parseExample(exampleString):
            featureDict = tf.io.parse_single_example(exampleString, featureDescription)
            txt = featureDict['txt']
            return txt

        trainList = []
        labelList = []
        for txt in rawDataSet.map(_parseExample):
            txtLen = txt.shape[0]
            for i in range(txtLen - seqLen):
                trainList.append(tf.slice(txt, [i], [seqLen]))
                labelList.append(tf.gather(txt, i + seqLen))

        return tf.data.Dataset.from_tensor_slices((trainList, labelList))




if __name__ == '__main__':
    dataDir = '/home/whx/workspace/work/python_code/autotext/data/train'
    saveFile = '/home/whx/workspace/work/python_code/autotext/data/train.tfrecord'
    keyFile = '/home/whx/workspace/work/python_code/autotext/data/key.json'
    # dataLoader = DataLoader(dataDir, saveFile)
    # dataLoader.writeFeature(keyFile)
    DataLoader.readFeature(saveFile, 3)
    # [14  2  9  0 11  5  3 10  7  4 13 12  6  1  8  6]





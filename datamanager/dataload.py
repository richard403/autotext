#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/4 下午1:43

# @author: whx

import tensorflow as tf
from utils import configParse
import os
import json

class DataLoader():
    def __init__(self, dataDir, saveFile, keyFile=None):
        self.dataDir = dataDir
        self.saveFile = saveFile
        self.keyFile = keyFile
        self.fileList = ['%s/%s' % (dataDir, fileName) for fileName in os.listdir(self.dataDir)]
        self.keyDict = {}
        self.keys = []

        if self.keyFile is not None and os.path.isfile(self.keyFile):
            with open(self.keyFile, encoding='utf-8') as f1:
                string = f1.read()
                self.keyDict = json.loads(string)
                self.keys = list(set(self.keyDict.keys()))
                self.keys.sort()

    def writeFeature(self):
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

        with open(self.keyFile, 'w') as f2:
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
    d_confDict = configParse.ConfigParser.getConf2Dict('RNN')
    d_trainDir = d_confDict['train_data']
    d_tfRecordFile = d_confDict['tfrecord_file']
    d_svModelFile = d_confDict['sv_model_dir']
    d_keyFile = d_confDict['key_file']
    dataLoader = DataLoader(d_trainDir, d_tfRecordFile, d_keyFile)
    dataLoader.writeFeature()
    # DataLoader.readFeature(saveFile, 3)
    # [14  2  9  0 11  5  3 10  7  4 13 12  6  1  8  6]





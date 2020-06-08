#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/8 下午3:05

# @author: whx

# export PYTHONPATH="/home/whx/workspace/work/python_code/autotext:{$PYTHONPATH}"
import tensorflow as tf
import numpy as np
from datamanager import dataload
from rnn.model import RNN
from config import SEQ_LEN
import argparse
import os
from utils import dateUtil
from utils import fileUtil
from utils import logger
from utils import configParse
from rnn import predict


_LOG = logger.getLogger('rnn_train')

def saveModel(model, saveModelDir):
    if saveModelDir is None:
        return
    if not os.path.isdir(saveModelDir):
        _LOG.error("保存模型目录：%s 不存在" % saveModelDir)
        return

    tagDir = os.path.join(saveModelDir, dateUtil.getNow(format='%Y%m%d_%H'))
    tf.saved_model.save(model, fileUtil.getDir(tagDir, create=True))


def loadModel(loadModelDir, flag):
    tagDir = os.path.join(loadModelDir, flag)
    if not os.path.isdir(tagDir):
        _LOG.error("载入目录：%s 不存在" % tagDir)
        return None
    model = tf.saved_model.load(tagDir)
    _LOG.info('载入模型path: %s' % tagDir)
    return model


def restoredFromCK(numChars):
    model_to_be_restored = RNN(numChars)
    checkpoint = tf.train.Checkpoint(myModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint(d_ckDir))
    return model_to_be_restored


def train(dataLoader, model, numBatches, seqLen, batchSize, saveModelFile=None):
    learningRate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
    ds = dataLoader.readFeature(d_tfRecordFile, seqLen).repeat(numBatches).shuffle(buffer_size=batchSize * 11).\
        batch(batchSize).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    checkpoint = tf.train.Checkpoint(myModel=model)
    ckName = dateUtil.getNow(format='%Y%m%d_%H') + '.ckpt'
    manager = tf.train.CheckpointManager(checkpoint, directory=d_ckDir, checkpoint_name=ckName, max_to_keep=3)

    @tf.function
    def train_one_step(x, y, batchIndex):
        with tf.GradientTape() as tape:
            y_pred = model.call(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            # tf.print('y=', y, 'y_pred=', y_pred)
            loss = tf.reduce_mean(loss)
            tf.print("batch_index:", batchIndex, "loss:", loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if d_tensorboard:
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=batchIndex)

    # tf.summary.trace_on(graph=True, profiler=True)
    batchIndex = tf.Variable(0, dtype=tf.int64)
    for ws, w in ds:
        train_one_step(ws, w, batchIndex)
        batchIndex.assign(batchIndex + 1)
        if batchIndex % d_ckNum == 0:  # 每隔100个Batch保存一次
            path = manager.save(checkpoint_number=batchIndex)
            _LOG.info("model saved to %s" % path)

    path = manager.save(checkpoint_number=batchIndex)
    _LOG.info("model saved to %s" % path)
    saveModel(model, saveModelFile)
    # with summary_writer.as_default():
    #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=d_confDict['tensorboard_path'])  # 保存Trace信息到文件（可选）


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument('--ck_num', default=100)
    parser.add_argument('--test_num', default=100)
    parser.add_argument('--restore', help="是否从ck中恢复", action="store_true")
    parser.add_argument('--tensorboard', action="store_true")
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=3)
    parser.add_argument('--temperature', default=1.0)
    parser.add_argument('--load_flag')
    args = parser.parse_args()

    d_confDict = configParse.ConfigParser.getConf2Dict('RNN')
    d_trainDir = d_confDict['train_data']
    d_tfRecordFile = d_confDict['tfrecord_file']
    d_svModelFile = d_confDict['sv_model_dir']
    d_keyFile = d_confDict['key_file']
    d_result = d_confDict['result_txt']
    d_ckNum = int(args.ck_num)
    d_ckDir = os.path.join(d_confDict['ck_dir'], 'rnn')
    d_tensorboard = args.tensorboard

    dataLoader = dataload.DataLoader(d_trainDir, d_tfRecordFile, d_keyFile)

    if d_tensorboard:
        summary_writer = tf.summary.create_file_writer(d_confDict['tensorboard_path'])

    if args.load_flag is not None:
        loadModel(d_svModelFile, args.load_flag)
    elif args.restore:
        model = restoredFromCK(len(dataLoader.keyDict))
    else:
        model = RNN(numChars=len(dataLoader.keyDict))

    if args.mode == 'train':
        train(dataLoader, model, int(args.num_epochs), SEQ_LEN, int(args.batch_size), d_svModelFile)

    if args.mode == 'test':
        resultList = predict.predict("wiki百科", dataLoader.keyDict, model, float(args.temperature), int(args.test_num))
        with open(d_result, 'w') as f:
            f.write(''.join(resultList))

# python ./datamanager/dataload.py
# python ./rnn/train.py --num_epochs=600 --restore --batch_size=50  --mode=train --tensorboard
# python ./rnn/train.py --test_num 200  --mode=test



#!/usr/bin/env python 
# -*-coding:utf-8 -*-

# Date    :   19-9-20 上午10:49

# Author  :   whx 

import logging
from logging import handlers
import sys
import os
import traceback
from utils import configParse
import config


APP_CONF = configParse.ConfigParser.getConf()
if APP_CONF.has_option('LOG', 'log_dir'):
    LOG_DIR = APP_CONF.get('LOG', 'log_dir')
    if len(LOG_DIR) <= 5:
        LOG_DIR = os.path.join(config.ROOT_PATH, 'data', 'logs')
else:
    LOG_DIR = os.path.join(config.ROOT_PATH, 'logs')


# print(LOG_DIR)

class Singleton(type):
    _instance = {}
    def __init__(cls, name, bases, dct):
        super(Singleton, cls).__init__(name, bases, dct)

    def __call__(cls, *args, **kwargs):
        logName = kwargs.get('logName', 'logName')
        outPath = r'%s' % kwargs.get('outPath', 'outPath')
        if (logName, outPath, cls) not in cls._instance:
            cls._instance[(logName, outPath, cls)] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance[(logName, outPath, cls)]


class Log():
    __metaclass__ = Singleton

    def __init__(self, logName, outPath=LOG_DIR, level=logging.DEBUG):
        self.logger = logging.getLogger(logName)

        self.mainDir = os.path.join(outPath, "main")
        self.errDir = os.path.join(outPath, "error")

        self.filePath = os.path.join(self.mainDir, r'%s.log' % logName)
        self.errPath = os.path.join(self.errDir, r'%s.log' % logName)

        self.formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s [%(name)s] : %(message)s')

        if not os.path.isdir(self.mainDir):
            os.makedirs(self.mainDir)

        if not os.path.isdir(self.errDir):
            os.makedirs(self.errDir)

        self.mainHandler = handlers.RotatingFileHandler(self.filePath, mode='a', maxBytes=10 << 20, backupCount=2, encoding='utf-8')
        self.errHandler = handlers.RotatingFileHandler(self.errPath, mode='a', maxBytes=10 << 20, backupCount=2, encoding='utf-8')

        self.mainHandler.setLevel(logging.INFO)
        self.mainHandler.setFormatter(self.formatter)

        self.errHandler.setLevel(logging.ERROR)
        self.errHandler.setFormatter(self.formatter)

        self.streamHandler = logging.StreamHandler(sys.stderr)
        self.streamHandler.setLevel(logging.DEBUG)
        self.streamHandler.setFormatter(self.formatter)

        self.logger.addHandler(self.mainHandler)
        self.logger.addHandler(self.errHandler)
        self.logger.addHandler(self.streamHandler)
        self.logger.setLevel(level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(traceback.format_exc())
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


def getLogger(logName="logger", outPath=LOG_DIR):
    return Log(logName=logName, outPath=outPath)


if __name__ == '__main__':
    log = getLogger("test")
    log.info("testlog")
    log.error("test")
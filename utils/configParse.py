#!/usr/bin/env python
# -*-coding:utf-8 -*-

# @Date    :   2019-09-20 10:33:02

# @Author  :   whx

import sys
import configparser
import os
from config import CONF_PATH

DEFAULT='default'


def getConfPathByName(confName):
    return os.path.join(CONF_PATH, '%s.ini' % confName)


class ConfigParser(object):
    confDict = {}

    @classmethod
    def getConf(cls, confName=DEFAULT):
        if confName in cls.confDict:
            return cls.confDict[confName]

        confFile = getConfPathByName(confName)
        if not os.path.exists(confFile):
            raise Exception("找不到ini配置地址path%s", confFile)

        confObj = configparser.ConfigParser()
        confObj.read(confFile, encoding="utf-8")
        cls.confDict[confName] = confObj
        return confObj


    @staticmethod
    def getConf2Dict(section, conf=DEFAULT):
        if type(conf) == configparser.ConfigParser:
            return dict(conf.items(section))

        if type(conf) == str:
            return dict(ConfigParser.getConf(conf).items(section))

    @staticmethod
    def getConfValue(section, attr, conf=DEFAULT):
        if type(conf) == configparser.ConfigParser:
            return conf.get(section, attr)

        if type(conf) == str:
            return ConfigParser.getConf(conf).get(section, attr)




if __name__ == '__main__':
    # configparser.ConfigParser()
    # config = configparser.ConfigParser()
    # config.read(getConfPathByName('es'), encoding="utf-8")
    # secs = config.sections()  # 获取所有的节点名称
    # options = config.options('HD_7_ES')
    # print(options)
    # print(dict(config.items('HD_7_ES')))
    #
    print(eval(ConfigParser.getConf2Dict('es', 'HD_7_ES').get('sniffer', 'False')) == True)
    # print(ConfigParser.confDict.items())
    # print(ConfigParser.getConfValue('app', 'LOG', 'log_dir'))
    # print(ConfigParser.getSection('app', 'LOG'))


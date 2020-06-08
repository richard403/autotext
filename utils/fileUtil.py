#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/8 上午11:20

# @author: whx

import os

def getDir(tagDir, create=False):
    if os.path.isdir(tagDir):
        return tagDir
    os.makedirs(tagDir)
    return tagDir
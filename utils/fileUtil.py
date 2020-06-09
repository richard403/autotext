#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created on 2020/6/8 上午11:20

# @author: whx

import os
import shutil

# op: 0 不创建，1 没有则创建，2 先删除再创建
def getDir(tagDir, op=0):
    if op == 1:
        os.makedirs(tagDir)
        return tagDir
    elif op == 2:
        if os.path.isdir(tagDir):
            shutil.rmtree(tagDir)
        os.makedirs(tagDir)
        return tagDir
    else:
        return tagDir if os.path.isdir(tagDir) else None


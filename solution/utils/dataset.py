# -*-coding: utf-8 -*-
# sys packages
import numpy as np
import pandas as pd


def loadData(input_dir):
    df = pd.read_csv('{}/rgbd.txt'.format(input_dir), sep=' ', names=['t1', 'rgb', 't2', 'depth'])
    return df.t1.values, df.rgb.values, df.depth.values

# -*-coding: utf-8 -*-
# sys packages
import numpy as np
import pandas as pd

# self define --- begin
from utils.ImageUtils import downscale
from utils.ImageUtils import deriveResidualsNumeric
from utils.ImageUtils import deriveResidualsAnalytic


def load_data(input_dir):
    df = pd.read_csv('{}/rgbd.txt'.format(input_dir), sep=' ', names=['t1', 'rgb', 't2', 'depth'])
    return df.t1.values, df.rgb.values, df.depth.values

# -*-coding: utf-8 -*-
# sys packages
import numpy as np
import pandas as pd


def loadData(input_dir):
    df = pd.read_csv('{}/rgbd.txt'.format(input_dir), sep=' ', names=['t1', 'rgb', 't2', 'depth'])
    return df.t1.values, df.rgb.values, df.t2.values, df.depth.values


def saveData(np_array, outdir, fn='estimate.txt', column=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']):
    # write the head of the estimate.txt
    with open('{}/{}'.format(outdir, fn), "w") as f:
        f.write('# timestamp tx ty tz qx qy qz qw\n')
    f.close()

    csv = pd.DataFrame(np_array, columns=column)
    csv.to_csv('{}/{}'.format(outdir, fn),
               encoding='utf-8', index_label=False, index=False, sep=' ',
               mode='a', header=False)

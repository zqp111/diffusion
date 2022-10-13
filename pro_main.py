# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import numpy as np
import argparse
from processor.base_method import import_class
import sys
import os
import torch.multiprocessing as mp

def main(rank, arg, train):
    trainer = train(arg, rank=rank)
    trainer.start()

if __name__ == '__main__':

    processors = dict()
    print(sys.argv[1])
    processors[sys.argv[1]] = import_class('processor.'+sys.argv[1]+'.main') 
    
    Processor = processors[sys.argv[1]] 
    # p = Processor(sys.argv[2:])   #sys.argv[0]指.py程序本身,argv[2:]指从命令行获取的第二个参数

    # print('start')
    # p.start()
    nprocs = int(sys.argv[2])
    arg = sys.argv[3:]
    mp.spawn(main, nprocs=nprocs, args=(arg, Processor))
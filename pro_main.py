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
    parser = argparse.ArgumentParser(description="AE network")

    processors = dict()
    print(sys.argv[1])
    processors[sys.argv[1]] = import_class('processor.'+sys.argv[1]+'.main') 

    subparsers = parser.add_subparsers(dest='processor')  # 启动命名空间为processor的添加子命令，dest=‘’将存储子命令名称的属性的名称为processor
    for k, p in processors.items():  # 将字典中的每一对变成元组的形式（[name,zhang],[age,20]）
        subparsers.add_parser(k, parents=[p.get_parser()]) # 添加子命令K,  这个子命令K继承了p.get_paeser()中定义的所有的命令参数
    # read arguments
    arg = parser.parse_args() #开始读取命令行的数值并保存
    # start
    
    Processor = processors[sys.argv[1]] 
    # p = Processor(sys.argv[2:])   #sys.argv[0]指.py程序本身,argv[2:]指从命令行获取的第二个参数

    # print('start')
    # p.start()
    arg = sys.argv[2:]
    mp.spawn(main, nprocs=2, args=(arg, Processor))
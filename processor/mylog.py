# -*- encoding: utf-8 -*-
'''
@File    :   Mylog.py
@Time    :   2020/09/11 15:29:58
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''
import os
import time
import logging
import inspect
from logging.handlers import RotatingFileHandler


class MyLog(object):

    def __init__(self, dir):
        self.init_handlers(dir)
        self.create_handlers()
        self.__loggers = {}
        logLevels = self.handlers.keys()
        for level in logLevels:
            logger = logging.getLogger(str(level))
            # 如果不指定level，获得的handler似乎是同一个handler?
            logger.addHandler(self.handlers[level])
            logger.setLevel(level)
            self.__loggers.update({level: logger})

    def printfNow(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def init_handlers(self, dir):
        dir_time = time.strftime('%Y-%m-%d', time.localtime())
        self.handlers = {
            logging.DEBUG: os.path.join(dir, 'eval_%s.log'%dir_time),
            logging.INFO: os.path.join(dir, 'train_%s.log'%dir_time),
            logging.WARNING: os.path.join(dir, 'all_%s.log'%dir_time)
            }

    def create_handlers(self):
        logLevels = self.handlers.keys()

        for level in logLevels:
            path = os.path.abspath(self.handlers[level])
            self.handlers[level] = RotatingFileHandler(path, maxBytes=10000000, backupCount=2, encoding='utf-8')

    def get_log_message(self, level, message):
        _, filename, lineNo, functionName, _, _ = inspect.stack()[2]
        '''日志格式：[时间] [类型] [记录代码] 信息'''
        return "[%s] [%s - %s - %s] %s" % (self.printfNow(), filename.split('/')[-1], lineNo, functionName, message)

    def train_log(self, message):
        message = self.get_log_message("info", message)

        self.__loggers[logging.INFO].info(message)

    def eval_log(self, message):
        message = self.get_log_message("debug", message)

        self.__loggers[logging.DEBUG].debug(message)
    
    def log(self, message):
        message = self.get_log_message("warning", message)

        self.__loggers[logging.WARNING].warning(message)    



if __name__ == "__main__":

    dir = "./log"
    logger = MyLog(dir)

    logger.train_log("train")
    logger.eval_log("eval")
    logger.log("all")


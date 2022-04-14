# -*- encoding: utf-8 -*-
'''
@File    :   base_method.py
@Time    :   2020/09/11 15:29:36
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''


def import_class(name):
    try:
        nameList = name.split(".")
        mod = __import__(nameList[0])
        for nam in nameList[1:]:
            mod = getattr(mod, nam)
        return mod
    except ImportError as err:
        print("cant find the module", err.args)


def str2bool(s):
    if s.lower() in ["true", "t", "yes", "y"]:
        return True
    elif s.lower() in ["false", "f", "no", "n"]:
        return False
    else:
        raise ValueError("Invalid str, cant conv to bool")


if __name__ == "__main__":
    import_class("model.model") 
    str2bool('a') #


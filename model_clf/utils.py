import os
import pickle

def save_pickle(obj, path):
    """保存对象为 pickle 文件"""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    """读取 pickle 文件为对象"""
    with open(path, 'rb') as f:
        return pickle.load( f)

def is_chinese(string):
    if type(string) is not str:
        string = str(string)
    """判断字符串是否包含中文"""
    return any('\u4e00' <= char <= '\u9fff' for char in string)

def ensure_dir(path,exist_ok=True):
    """确保目录存在，不存在就创建"""
    os.makedirs(path, exist_ok=exist_ok)
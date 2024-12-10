import os
import os.path as path
import json
import numpy as np

def load_file_from_cwd(file_name,mode='r',encoding='utf-8',**kwargs):
    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(path.join(cwd,file_name),mode=mode,encoding=encoding,**kwargs) as f:
        return f.read()

def save_file_to_cwd(contents,file_name,mode='w',encoding = 'utf-8',**kwargs):
    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(path.join(cwd,file_name),mode=mode,encoding=encoding,**kwargs) as f:
        return f.write(contents)
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np

'''------------ 选择喜欢的模型 -------------------'''

def svr_model():
    """linear svr model"""
    svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
                                       "gamma": np.logspace(-3, 3, 7)})
    return svr
import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib
from sklearn.model_selection import train_test_split


'''--------------- 预处理，读取数据的帮助方法 ----------------'''

def deal_with_rough_data(rough_data,output_location):
    """deal with rough data to fit my requirements ,output a new csv file and return the data"""
    data = rough_data[rough_data['MW'] < 254] # c18h38 分子量在 86~254间
    data = data[data['MW'] > 86]  # c6h14
    data = data[data['Hy'] > 0]   # Hy 在 0~1
    data = data[data['Hy'] < 1]
    #data = data[data['BLTF96'] < 0]
    data = data[data['BLTD48'] < 0]
    #data = data[data['BLTA96'] < 0]
    data = data[data['MLOGP'] > 0]
    data = data[data['ALOGP'] != 'n.a.']

    data.to_csv(output_location,index=False)
    return data


def output_columns_values(data, column_names):
    """output cols values"""
    return data[column_names].values


def read_and_output_data(data_path, feat_cols, target_col):
    """read the data path and excited feat cols, and target col
       output the x_column and y_column
    """
    data = pd.read_csv(data_path)
    x_col = output_columns_values(data,feat_cols)
    y_col = output_columns_values(data,target_col)
    return x_col, y_col


'''------------ 训练与输出、读取模型 -------------------'''

def train_model(x_read, y_read, model):
    """train model relay on x_read, y_read, and model """
    x = preprocessing.scale(x_read)
    # y = preprocessing.scale(y_read)
    y = y_read * 100
    return model().fit(x, y)


def output_model(input_data_path,feat_cols,target_col,model_type,output_model_name):
    """output model"""
    x_read, y_read = helper_method.read_and_output_data(data_path=input_data_path, feat_cols=feat_cols, target_col=target_col)
    model = helper_method.train_model(x_read, y_read, model_type)
    joblib.dump(model, output_model_name)


def load_model(model_name):
    return joblib.load(model_name)


'''------------------ 测试模型 -------------------------------'''

def test_model(model,x_read,y_read):
    """print the model score and print none"""
    x = preprocessing.scale(x_read)
    y = y_read * 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=10)
    model.fit(x_train, y_train)
    r2_score = model.score(x_train, y_train)
    print('model_R2 = ', r2_score)
    y_mean = y_read.mean(axis=0)
    y_std = y_read.std(axis=0)
    print('y_mean = ',y_mean)
    print('_std = ',y_std)
    print('(y_read - y_mean) / y_std = ',(y_read - y_mean) / y_std)


'''------------ 利用模型预测、输出结果 -------------------'''

def predict_y(model, x_read):
    """return predict y according to x_read and model"""
    x = preprocessing.scale(x_read)
    x_reshape = np.array(x)
    y_predict = model.predict(x_reshape)  # 对结果进行可视化：
    return y_predict / 100


def output_y_predict(model_name, data_path, feat_cols,output_location):
    """output_y_predict"""
    model = load_model(model_name)
    data = helper_method.deal_with_rough_data(pd.read_csv(data_path),"./data/selected_full_test_dataset.csv")
    feat_cols_values = data[feat_cols].values
    predict_y_nums = predict_y(model, feat_cols_values)
    output = data[['Name','id','symbol']].copy()    # 取三列
    output['predict_y_nums'] = predict_y_nums       # 加一列
    output.to_csv(output_location,index=False)



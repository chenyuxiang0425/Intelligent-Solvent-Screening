from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd
import helper_method

'''------ 根据特征向量的重要度选取特征向量的方法 --------------'''


def random_forest_regressor_model(x_read, y_read, names):
    """ 随机森林 """
    rf = RandomForestRegressor(n_estimators=20, max_features=2)
    rf.fit(x_read, y_read)
    list = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
    return list


def extra_trees_classifier(x_read, y_read, names):
    """ 极度随机树 比随机森立更随机 """
    model = ExtraTreesClassifier()
    model.fit(x_read, (y_read*3).astype('int'))
    list = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), reverse=True)
    return list


def get_important_column_name(item):
    """ 上面 return 的 list 中每个元素是个二元元组，获取第二个元素，也就是column name"""
    return item[1]

def get_important_factors(item):
    """ 上面 return 的 list 中每个元素是个二元元组，获取第二个元素，也就是重要度 """
    return item[0]


def list_to_dict(my_list, header = None):
    """ 把list结构改成字典 """
    to_dict = {}
    length = len(my_list)
    if header is not None:   # 选取前header个元素
        length = header

    for i in range(length):
        to_dict[get_important_column_name(my_list[i])] = get_important_factors(my_list[i])

    return to_dict

def print_dict(my_dict):
    """打印字典"""
    for key,value in my_dict.items():
        print(key + ": ",value)

if __name__ == '__main__':
    MeSH_data_path = './data/MeSH_data_within_names.csv'

    data = pd.read_csv(MeSH_data_path)
    all_feat_cols_names = data.columns.values.tolist()  # 所有的列都被认为是 feature column
    all_feat_cols_names.remove('MeSHSolubility')       # 去除 target column
    all_feat_cols_names.remove('Name')              # 去除 name column
    x_read, y_read = helper_method.read_and_output_data(data_path=MeSH_data_path, feat_cols=all_feat_cols_names,
                                                        target_col='MeSHSolubility')
    random_forest_regressor_results = random_forest_regressor_model(x_read, y_read,all_feat_cols_names)
    extra_trees_classifier_results = extra_trees_classifier(x_read, y_read, all_feat_cols_names)

    print("random_forest_regressor_model:")
    print_dict(list_to_dict(random_forest_regressor_results,header=30))
    print("--------------------------------------------")
    print("extra_trees_classifier:")
    print_dict(list_to_dict(extra_trees_classifier_results,header=30))

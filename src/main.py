import helper_method
import models

'''----------------------------------------- 主函数 -----------------------------------------------'''

if __name__ == '__main__':

    ''' ------------------ 筛选特征值(这一步暂未完成，暂时用 spss 完成) --------------------'''
    FEAT_COLS = dict()
    FEAT_COLS['MeSH_FEAT_COLS'] =  ["Ms","nDB","nTB","nX","X3A","X2Av","X3Av","nRCONH2","nRCONHR","nRCONR2","nC=O(O)2",
                 "nRNH2","nArOH","nOHp","C-042","O-059","Ui","MLOGP2"]
    FEAT_COLS['COS_FEAT_COLS'] = ["Ms","Hy","HATS2v","BLTA96","ALOGP","GATS5e","H-047","H4p","RTe+","HATS2p","HyDp",
                 "IC0","Vm","DISPm","X3Av","MATS1e","Au","ISH"]


    ''' -------------------------------- 想做哪些工作 ----------------------------------'''
    train_model = False
    test_model = True
    output_data = False
    selected_model = models.svr_model

    if train_model:
        helper_method.output_model(input_data_path='../data/MeSH_data_within_names.csv',feat_cols=FEAT_COLS.get('MeSH_FEAT_COLS'), target_col = 'MeSHSolubility', model_type= selected_model ,output_model_name="../output/MeSH_train_model.m")
        helper_method.output_model(input_data_path='../data/COS_data_within_names.csv',feat_cols=FEAT_COLS.get('COS_FEAT_COLS'), target_col = 'COSSolubility', model_type= selected_model, output_model_name="../output/COS_train_model.m")
    if test_model:
        MeSH_x_read, MeSH_y_read = helper_method.read_and_output_data('../data/MeSH_data_within_names.csv', FEAT_COLS.get('MeSH_FEAT_COLS'), 'MeSHSolubility')
        helper_method.test_model(helper_method.load_model("../output/MeSH_train_model.m"), MeSH_x_read, MeSH_y_read)
        COS_x_read, COS_y_read = helper_method.read_and_output_data('../data/COS_data_within_names.csv', FEAT_COLS.get('COS_FEAT_COLS'), 'COSSolubility')
        helper_method.test_model(helper_method.load_model("../output/COS_train_model.m"), COS_x_read, COS_y_read)
    if output_data:
        helper_method.output_y_predict("../output/MeSH_train_model.m",'../data/selected_full_test_dataset.csv',FEAT_COLS.get('MeSH_FEAT_COLS'),'../output/MeSH_data_res.csv')
        helper_method.output_y_predict("../output/COS_train_model.m",'../data/selected_full_test_dataset.csv',FEAT_COLS.get('COS_FEAT_COLS'),'../output/COS_data_res.csv')


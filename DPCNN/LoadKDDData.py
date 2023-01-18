from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn
import numpy as np
import pickle as pkl
import pandas as pd
import os
import matplotlib.pyplot as plt

act_feat = ['0#num', '1#num', '2#num', '3#num', '4#num', '5#num', '6#num']
label_feat = ['truth', 'total_activity_day']
user_cat_feat = []
user_num_feat = ['user_feature_1', 'user_feature_2']


class KDDDataSet(Dataset):
    def __init__(self, ui, uv, ai, av, y,time):
        super(KDDDataSet, self).__init__()
        self.ui = ui
        self.uv = uv
        self.ai = ai
        self.av = av
        self.y = y
        self.time=time
        self.len = ai.shape[0]

    def __getitem__(self, item):
        return self.ui[item], self.uv[item], self.ai[item], self.av[item], self.y[item],self.time[item]

    def __len__(self):
        return self.len


def dataparse(df_u):
    # Feature dimension，for example, the total number of features of 'gender' and 'cluster_label' in user_cat_feat
    feat_dim = 0
    feat_dict = dict()
    # For example, pack all possible items of each feature belonging to user_cat_feat in all_data
    for f in user_cat_feat:
        cat_val = df_u[f].unique()
        # Packed into a dictionary, such as {0:0, 1:1, 2:2}, feat_dam is cumulative
        feat_dict[f] = dict(zip(cat_val, range(feat_dim, len(cat_val) + feat_dim)))
        feat_dim += len(cat_val)

    for f in user_num_feat:
        feat_dict[f] = feat_dim
        feat_dim += 1

    data_indice = df_u.copy()
    data_value = df_u.copy()
    for f in df_u.columns:
        if f in user_num_feat:
            data_indice[f] = feat_dict[f]
        elif f in user_cat_feat:
            data_indice[f] = data_indice[f].map(feat_dict[f])
            data_value[f] = 1.
        else:
            data_indice.drop(f, axis=1, inplace=True)
            data_value.drop(f, axis=1, inplace=True)

    # Feature dimension   data index   data value
    return feat_dim, data_indice, data_value


def load_data(past_day,future_day=23):
    # Load feature enroll_id as primary key
    # train
    df_u = pd.DataFrame()
    df_a = pd.DataFrame()
    for i in range(1, past_day + 1):
        file_name = "train_day_" + str(i) + "_activity_feature"
        df = pd.read_csv('./data/KDD/feature/' + file_name + '.csv')
        if i == 1:
            df_u = df[['enrollment_id', 'user_feature_1', 'user_feature_2']]
        df_cur_a = df[['enrollment_id'] + act_feat]
        df_a = pd.concat([df_a, df_cur_a])
    # read label
    label = pd.read_csv('./data/KDD/info/kdd_train_user_info.csv')
    label = label[['enrollment_id'] + label_feat]
    # sort
    df_a.sort_values("enrollment_id", inplace=True)
    df_u.sort_values("enrollment_id", inplace=True)
    label.sort_values("enrollment_id", inplace=True)
    # test
    test_df_u = pd.DataFrame()
    test_df_a = pd.DataFrame()
    for i in range(1, past_day + 1):
        file_name = "test_day_" + str(i) + "_activity_feature"
        df = pd.read_csv('./data/KDD/feature/' + file_name + '.csv')
        if i == 1:
            test_df_u = df[['enrollment_id', 'user_feature_1', 'user_feature_2']]
        df_cur_a = df[['enrollment_id'] + act_feat]
        test_df_a = pd.concat([test_df_a, df_cur_a])
    # read label
    test_label = pd.read_csv('./data/KDD/info/kdd_test_user_info.csv')
    test_label = test_label[['enrollment_id'] + label_feat]
    # print('test_label:' + str(len(test_label)))
    # sort
    test_df_a.sort_values("enrollment_id", inplace=True)
    test_df_u.sort_values("enrollment_id", inplace=True)
    test_label.sort_values("enrollment_id", inplace=True)
    # print('test_label:' + str(len(test_label)))
    #read time
    df_train_time = pd.read_csv('./data/KDD/train_time.csv')
    df_test_time = pd.read_csv('./data/KDD/test_time.csv')
    df_train_time.sort_values("enrollment_id",inplace=True)
    df_test_time.sort_values("enrollment_id", inplace=True)
    del df_train_time['enrollment_id']
    del df_test_time['enrollment_id']
    df_train_time_split=pd.DataFrame()
    df_test_time_split=pd.DataFrame()
    for i in range(1,past_day+future_day+1):
        #train
        df_train_time_split['year'+str(i)]=pd.to_datetime(df_train_time["day"+str(i)]).dt.year
        df_train_time_split['month' + str(i)] = pd.to_datetime(df_train_time["day" + str(i)]).dt.month
        df_train_time_split['day' + str(i)] = pd.to_datetime(df_train_time["day" + str(i)]).dt.day
        df_train_time_split['week' + str(i)] = df_train_time["week" + str(i)]
        #test
        df_test_time_split['year' + str(i)] = pd.to_datetime(df_test_time["day" + str(i)]).dt.year
        df_test_time_split['month' + str(i)] = pd.to_datetime(df_test_time["day" + str(i)]).dt.month
        df_test_time_split['day' + str(i)] = pd.to_datetime(df_test_time["day" + str(i)]).dt.day
        df_test_time_split['week' + str(i)] = df_test_time["week" + str(i)]
    #print(df_train_time_split.shape)
    return df_u, df_a, label, test_df_u, test_df_a, test_label,df_train_time_split,df_test_time_split


def getDataLoader(batch_size=32, params={}):
    # add column: day1, day2...., dayN
    # Judge whether the corresponding day is active
    past_day = params['day']
    future_day = params['future_day']
    sd = 'day'
    for i in range(1, future_day + 1):
        curday = sd + str(i)
        label_feat.append(curday)
    # load data
    df_u, df_a, label, test_df_u, test_df_a, test_label,df_train_time,df_test_time = load_data(past_day,future_day)

    # train -> train, valid
    # Feature dimension   data index   data value(student)
    u_feat_dim, u_data_indice, u_data_value = dataparse(df_u)

    # Turn u_data_indice and u_data_value into array
    ui, uv = np.asarray(u_data_indice.loc[df_u.index], dtype=int), np.asarray(
        u_data_value.loc[df_u.index], dtype=np.float32)
    params["u_feat_size"] = u_feat_dim
    params["u_field_size"] = len(ui[0])
    # action
    av = np.asarray(df_a[act_feat], dtype=np.float32)
    ai = np.asarray([range(len(act_feat)) for x in range(len(df_a))], dtype=int)
    params["a_feat_size"] = len(av[0])
    params["a_field_size"] = len(ai[0])
    # user_id, day, action_type_num
    av = av.reshape((-1, params['day'], len(act_feat)))
    ai = ai.reshape((-1, params['day'], len(act_feat)))
    params['input_size'] = len(act_feat)
    #time
    #train_time:[enroll_num,(past_day+future_day)*4]
    train_time=np.asarray(df_train_time,dtype=np.float32)
    #train_time:[enroll_num,past_day+future_day,4]
    train_time=train_time.reshape((-1,past_day+future_day,4))
    # result
    y = np.asarray(label[label_feat], dtype=np.float32)

    # Turn all data into tensor
    # train
    ui = torch.tensor(ui)
    uv = torch.tensor(uv)
    train_time = torch.tensor(train_time)
    y = torch.tensor(y)

    # Divide the training set into the validation set
    data_num = len(y)
    indices = np.arange(data_num)
    np.random.seed(params['seed'])
    np.random.shuffle(indices)
    split_1 = int(0.75 * data_num)
    ui_train, ui_valid = ui[indices[:split_1]], ui[indices[split_1:]]
    uv_train, uv_valid = uv[indices[:split_1]], uv[indices[split_1:]]
    ai_train, ai_valid = ai[indices[:split_1]], ai[indices[split_1:]]
    av_train, av_valid = av[indices[:split_1]], av[indices[split_1:]]
    time_train,time_valid=train_time[indices[:split_1]],train_time[indices[split_1:]]
    y_train, y_valid = y[indices[:split_1]], y[indices[split_1:]]
    # user_activate_day_count
    label = label.iloc[indices[:split_1]]
    day_numpy = user_activate_day_count(params, label)

    # test
    # Feature dimension   data index   data value(student)
    test_u_feat_dim, test_u_data_indice, test_u_data_value = dataparse(test_df_u)
    # Turn u_data_indice and u_data_value into array
    test_ui, test_uv = np.asarray(test_u_data_indice.loc[test_df_u.index], dtype=int), np.asarray(
        test_u_data_value.loc[test_df_u.index], dtype=np.float32)
    # action
    test_av = np.asarray(test_df_a[act_feat], dtype=np.float32)
    test_ai = np.asarray([range(len(act_feat)) for x in range(len(test_df_a))], dtype=int)
    # user_id, day, action_type_num
    test_av = test_av.reshape((-1, params['day'], len(act_feat)))
    test_ai = test_ai.reshape((-1, params['day'], len(act_feat)))
    #time
    #train_time:[enroll_num,(past_day+future_day)*4]
    test_time=np.asarray(df_test_time,dtype=np.float32)
    #train_time:[enroll_num,past_day+future_day,4]
    test_time=test_time.reshape((-1,past_day+future_day,4))
    # result
    test_y = np.asarray(test_label[label_feat], dtype=np.float32)
    # Turn all data into tensor
    test_ui = torch.tensor(test_ui)
    test_uv = torch.tensor(test_uv)
    test_time=torch.tensor(test_time)
    test_y = torch.tensor(test_y)
    # rename
    if not params['miniData']:
        ui_test = test_ui
        uv_test = test_uv
        ai_test = test_ai
        av_test = test_av
        time_test=test_time
        y_test = test_y
    else:
        data_num = len(test_y)
        indices = np.arange(data_num)
        np.random.seed(params['seed'])
        np.random.shuffle(indices)
        split_1 = int(0.6 * data_num)
        split_2 = int(0.8 * data_num)
        ui_train, ui_valid, ui_test = test_ui[indices[:split_1]], test_ui[indices[split_1:split_2]], test_ui[
            indices[split_2:]]
        uv_train, uv_valid, uv_test = test_uv[indices[:split_1]], test_uv[indices[split_1:split_2]], test_uv[
            indices[split_2:]]
        ai_train, ai_valid, ai_test = test_ai[indices[:split_1]], test_ai[indices[split_1:split_2]], test_ai[
            indices[split_2:]]
        av_train, av_valid, av_test = test_av[indices[:split_1]], test_av[indices[split_1:split_2]], test_av[
            indices[split_2:]]
        time_train, time_valid, time_test = test_time[indices[:split_1]], test_time[indices[split_1:split_2]], \
                                            test_time[indices[split_2:]]
        y_train, y_valid, y_test = test_y[indices[:split_1]], test_y[indices[split_1:split_2]], test_y[
            indices[split_2:]]
        # user_activate_day_count
        test_label = test_label.iloc[indices[:split_1]]
        day_numpy = user_activate_day_count(params, test_label)

    # ui_train: user_num , user_info
    # ai_train_: user_num , day , action_type
    # t_train:user_num,future_day
    # y_label:user_num , [truth   activate_day  activate_num]
    train_dataset = KDDDataSet(ui_train, uv_train, ai_train, av_train, y_train,time_train)
    valid_dataset = KDDDataSet(ui_valid, uv_valid, ai_valid, av_valid, y_valid,time_valid)
    test_dataset = KDDDataSet(ui_test, uv_test, ai_test, av_test, y_test,time_test)
    # packaged dataset
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_set = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_set, valid_set, test_set, day_numpy, params


def user_activate_day_count(param, label):
    day_list = []
    for i in range(0, param['future_day'] + 1):
        cur_day_count = (label['total_activity_day'] == i).sum()
        day_list.append(cur_day_count)
    print(day_list)
    day_numpy = np.array(day_list)
    # x = np.arange(0, param['future_day'] + 1)
    # plt.plot(x, day_numpy)
    # plt.show()
    return day_numpy

if __name__ == '__main__':
    load_data(7)

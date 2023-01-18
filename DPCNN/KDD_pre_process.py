import numpy as np
import pandas as pd
from sklearn import preprocessing


def extraction_data(enroll_data_file_path, log_data_file_path, date_file_path, k, source):
    enroll_data = pd.read_csv(enroll_data_file_path)
    log_data = pd.read_csv(log_data_file_path)
    date = pd.read_csv(date_file_path)
    course_id = enroll_data['course_id'].unique()
    index = np.arange(len(course_id))
    np.random.shuffle(index)
    part_index = int(k * len(course_id))
    part_course_id = course_id[index[:part_index]]
    part_enroll_data = enroll_data[enroll_data.course_id.isin(part_course_id)]
    data_temp = pd.merge(part_enroll_data, log_data, on='enrollment_id')
    data = pd.merge(data_temp, date, on='course_id')
    path = './data/KDD_Log/' + source + '/part_' + source + '_data.csv'
    data.to_csv(path, index=False)


def split_by_days(file_path, split_day, source):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['from'] = pd.to_datetime(df['from'])
    df['to'] = pd.to_datetime(df['to'])
    df['day'] = (df['time'] - df['from']).dt.days + 1
    df1 = df[df["day"] <= split_day]
    df2 = df[df["day"] > split_day]
    df2['day'] = df2['day'] - split_day
    path1 = './data/KDD_Log/' + source + '_train_data.csv'
    path2 = './data/KDD_Log/' + source + '_test_data.csv'
    df1.to_csv(path1)
    df2.to_csv(path2)


def preprocess_data(train_file_path, test_file_path, day, future_day, source):
    df_train = pd.read_csv(train_file_path, engine='python')
    df_test = pd.read_csv(test_file_path, engine='python')
    # train：Group by 'day', connect by 'enroll', and count the number of each action by 'enroll'
    # test：Group by 'day', connect by 'enroll'; group by enroll, if > 0, label = 1; otherwise = 0, label = 0
    action = ['problem', 'video', 'access', 'wiki', 'discussion', 'navigate', 'page_close']
    enroll_id = df_train[['enrollment_id']].drop_duplicates()
    information = df_train[['enrollment_id', 'username', 'course_id']].drop_duplicates()
    information = information.set_index(information['enrollment_id'])
    label = enroll_id.set_index(enroll_id['enrollment_id'])
    temp = label.copy()
    for key in range(1, day + 1):
        day_data = df_train[df_train['day'] == key]
        day_data = pd.merge(enroll_id, day_data, on=['enrollment_id'], how='left')
        all_num = information.copy()
        for a in action:
            action_ = (day_data['event'] == a).astype(int)
            day_data[a + '#num'] = action_
            action_num = day_data.groupby('enrollment_id').sum()[[a + '#num']]
            all_num = pd.merge(all_num, action_num, left_index=True, right_index=True)
        all_num['all#num'] = all_num['problem#num'] + all_num['video#num'] + all_num['access#num'] + all_num[
            'wiki#num'] + all_num['discussion#num'] + all_num['navigate#num'] + all_num['page_close#num']
        all_num.to_csv('./data/KDD/log/' + source + '_day_' + str(int(key)) + '_activity_log.csv', index=False)
    for key in range(1, future_day + 1):
        day_data = df_test[df_test['day'] == key]
        day_data = pd.merge(enroll_id, day_data, on=['enrollment_id'], how='left')
        all_num = information.copy()
        i = 1
        for a in action:
            action_ = (day_data['event'] == a).astype(int)
            day_data[a + '#num'] = action_
            action_num = day_data.groupby('enrollment_id').sum()[[a + '#num']]
            all_num = pd.merge(all_num, action_num, left_index=True, right_index=True)
            temp_num = action_num.copy()
            temp_num.columns = ['day' + str(int(key)) + '_' + str(i)]
            label = pd.merge(label, temp_num, left_index=True, right_index=True)
            i = i + 1
        all_num['all#num'] = all_num['problem#num'] + all_num['video#num'] + all_num['access#num'] + all_num[
            'wiki#num'] + all_num['discussion#num'] + all_num['navigate#num'] + all_num['page_close#num']
        num = (all_num[['all#num']] > 0).astype(int)
        num.columns = ['day' + str(int(key))]
        temp = pd.merge(temp, num, left_index=True, right_index=True)
    temp["total_activity_day"] = sum(temp['day{}'.format(i)] for i in range(1, future_day + 1))
    temp = temp[["total_activity_day"]]
    label = pd.merge(label, temp, left_index=True, right_index=True)
    label["truth"] = label["total_activity_day"] / future_day
    path = './data/KDD/info/kdd_' + source + '_user_info.csv'
    label.to_csv(path, index=False)


def standerscaler(file_path, days, source):
    scaler = preprocessing.StandardScaler()
    for i in range(1, days + 1):
        path = file_path + 'log/' + source + '_day_' + str(i) + '_activity_log.csv'
        df = pd.read_csv(path, engine='python')
        df.iloc[:, 3:11] = scaler.fit_transform(df.iloc[:, 3:11])
        new_name = {'problem#num': '0#num', 'video#num': '1#num', 'access#num': '2#num', 'wiki#num': '3#num',
                    'discussion#num': '4#num', 'navigate#num': '5#num', 'page_close#num': '6#num'}
        df.rename(columns=new_name, inplace=True)
        df['user_feature_1'] = 0
        df['user_feature_2'] = 0
        path_temp = file_path + 'feature/' + source + '_day_' + str(i) + '_activity_feature.csv'
        df.to_csv(path_temp, index=False)


def create_file_by_data(day, future_day, dilution_ratio=1.0):
    train_enroll_data_file_path = './data/KDD_Log/train/enrollment_train.csv'
    test_enroll_data_file_path = './data/KDD_Log/test/enrollment_test.csv'
    train_log_data_file_path = './data/KDD_Log/train/log_train.csv'
    test_log_data_file_path = './data/KDD_Log/test/log_test.csv'
    date_file_path = './data/KDD_Log/date.csv'

    part_train_data_file_path = './data/KDD_Log/train/part_train_data.csv'
    part_test_data_file_path = './data/KDD_Log/test/part_test_data.csv'

    train_train_data_file_path = './data/KDD_Log/train_train_data.csv'
    train_test_data_file_path = './data/KDD_Log/train_test_data.csv'
    test_train_data_file_path = './data/KDD_Log/test_train_data.csv'
    test_test_data_file_path = './data/KDD_Log/test_test_data.csv'

    train_file_path = './data/KDD/'

    # processing train data
    extraction_data(train_enroll_data_file_path, train_log_data_file_path, date_file_path, dilution_ratio, 'train')
    split_by_days(part_train_data_file_path, day, 'train')
    # 'day' before: day_k_activity_log.csv(k=1...day); 'future day' after: test_truth.csv
    preprocess_data(train_train_data_file_path, train_test_data_file_path, day, future_day, 'train')
    # Before processing: day_k_activity_log.csv(k=1……20); After processing: day_k_activity_feature.csv(k=1^20)
    standerscaler(train_file_path, day, 'train')

    # processing test data
    extraction_data(test_enroll_data_file_path, test_log_data_file_path, date_file_path, dilution_ratio, 'test')
    split_by_days(part_test_data_file_path, day, 'test')
    preprocess_data(test_train_data_file_path, test_test_data_file_path, day, future_day, 'test')
    standerscaler(train_file_path, day, 'test')


if __name__ == '__main__':
    # test
    create_file_by_data(7, 23)

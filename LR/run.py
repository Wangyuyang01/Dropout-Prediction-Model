'''
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import mean_squared_error

esp = 1e-5


# Result evaluation
def calEvalResult(loss_value, y_preds, y_trues, test_type, write_file=None):
    y_preds_bool = np.copy(y_preds)
    y_preds_bool[y_preds >= 0.5] = 1.0
    y_preds_bool[y_preds < 0.5] = 0.0

    eps = 1e-5
    y_trues_bool = np.copy(y_trues)
    y_trues_bool[y_trues >= eps] = 1.0
    y_trues_bool[y_trues < eps] = 0.0
    # auc
    #print(y_trues_bool)
    #print(y_preds)
    #print(y_trues_bool.shape)
    if y_trues_bool.sum()==y_trues_bool.shape[0]:
        y_trues_bool = np.append(y_trues_bool, 0)
        y_preds = np.append(y_preds, 0)
        y_trues = np.append(y_trues, 0)
    if y_trues_bool.sum()==0:
        y_trues_bool = np.append(y_trues_bool, 1)
        y_preds = np.append(y_preds, 1)
        y_trues = np.append(y_trues, 1)
    auc = metrics.roc_auc_score(y_trues_bool, y_preds)
    # rmse
    error = y_trues - y_preds
    rmse = (error ** 2).mean() ** 0.5
    # df
    df = abs(y_trues.mean() - y_preds.mean()) / y_trues.mean()

    print_str = '%20s loss %3.6f  auc  %.4f  rmse  %.4f  df(ActivateDay.Avg) %.4f' % (
        test_type, loss_value, auc, rmse, df)
    print(print_str)
    if (write_file != None):
        write_file.write(print_str + "\n")
    return auc, rmse, df


def calEvalResult_FLTADP(loss_value, predict_results, test_type, write_file=None,isMask=False):
    # y_preds_1, y_trues_1 whether active every day (category) : [data_num, label_num]
    # y_preds_2, y_trues_2 active days percentage (category) : [data_num,] \in [0,1]
    y_preds_1, y_trues_1, y_preds_2, y_trues_2 = predict_results

    y_auc_daylevel_activate = metrics.roc_auc_score(y_trues_1.reshape(-1), y_preds_1.reshape(-1))
    if isMask:
        y_trues_2=np.append(y_trues_2,0)
        y_preds_2=np.append(y_preds_2,0)
        #print(y_trues_2)
        #print(y_preds_2)
    y_auc_01_activate = metrics.roc_auc_score(y_trues_2 > 1e-5, y_preds_2)
    #print(y_auc_01_activate)
    # rmse
    error = (y_trues_2 - y_preds_2)
    rmse = (error ** 2).mean() ** 0.5

    # df: average number of active days in the future
    df = abs(y_preds_2.mean() - y_trues_2.mean()) / y_trues_2.mean()
    # print("df:", y_predicts_num2.mean(), y_trues_num2.mean())
    isMask_Str='  isMask'+str(isMask)
    print_str = '%20s loss %6.6f auc_day_activate %.4f auc_01_activate %.4f  rmse  %.4f  df(ActivateDay.Avg) %.4f' % (
        test_type, loss_value, y_auc_daylevel_activate, y_auc_01_activate, rmse, df)
    if isMask:
        print_str= print_str + isMask_Str
    print(print_str)
    if (write_file != None):
        write_file.write(print_str + "\n")
    return y_auc_01_activate, rmse, df


def run(epoch, datas, model, optimizer, device, model_name="None", run_type="train", lossFun='BCE', write_file=None):
    y_trues = np.array([])
    y_preds = np.array([])
    y_trues_2 = np.array([])
    y_preds_2 = np.array([])
    #mask
    y_trues_filtered = np.array([])
    y_preds_filtered = np.array([])
    y_trues_2_filtered = np.array([])
    y_preds_2_filtered = np.array([])

    training_features = None
    all_loss = 0
    if (run_type == "train"):
        model.train()
    else:
        model.eval()
    data_id = 0
    for data in datas:
        ui, uv, ai, av, y, time = data
        #print('ui_shape:',ui.shape)
        #print(time.shape)
        ui = ui.to(device)
        uv = uv.to(device)
        ai = ai.to(device)
        av = av.to(device)
        y = y.to(device)
        if (run_type == "train"):
            av_uv = torch.concat((av.reshape(-1, av.shape[1] * av.shape[2]), uv.reshape(-1, uv.shape[1])), dim=1)
            # training_feature [data_num, day * a_field_dim + u_field_dim]
            if (training_features == None):
                training_features = av_uv
            else:
                training_features = torch.concat((training_features, av_uv), dim=0)
            optimizer.zero_grad()
        if (model_name != "FLTADP"):
            # y:(Proportion of active days):batch_size * 1
            y = y[:, 0].reshape(-1, 1)
            loss, y_pred = model.forward(ui, uv, ai, av, y, lossFun)
            #print('y_pred:')
            #print(y_pred.shape)
            # y_trues is the list of proportion of active days
            y_trues = np.concatenate((y_trues, y.detach().cpu().numpy().reshape(-1)), axis=0)
            # y_pred is the activate or not in future days
            y_preds = np.concatenate((y_preds, y_pred.reshape(-1).detach().cpu().numpy()), axis=0)
        else:
            y_1 = y[:, 2:].detach().to(device)
            y_2 = y[:, 1].detach().long().to(device)
            y_2_input = F.one_hot(y_2, num_classes=model.future_day + 1).float()
            y_2 = y_2 / (model.future_day + 1)
            time = time.to(device)
            loss, y_pred_1, y_pred_2,filtered_y_1,filtered_y_2,filtered_pred1,filtered_pred2 = model.forward(ui, uv, ai, av, y_1, y_2_input, epoch,time)
            if (y_trues.shape[0] < 2):
                y_trues = y_1.detach().cpu().numpy()
                y_preds = y_pred_1.detach().cpu().numpy()
                y_trues_2 = y_2.detach().cpu().numpy()
                y_preds_2 = y_pred_2.detach().cpu().numpy()
                #mask data
                y_trues_filtered = filtered_y_1.detach().cpu().numpy()
                y_preds_filtered = filtered_pred1.detach().cpu().numpy()
                y_trues_2_filtered = filtered_y_2.detach().cpu().numpy()
                y_preds_2_filtered = filtered_pred2.detach().cpu().numpy()
            else:
                y_trues = np.concatenate((y_trues, y_1.detach().cpu().numpy()), axis=0)
                y_preds = np.concatenate((y_preds, y_pred_1.detach().cpu().numpy()), axis=0)
                y_trues_2 = np.concatenate((y_trues_2, y_2.detach().cpu().numpy()), axis=0)
                y_preds_2 = np.concatenate((y_preds_2, y_pred_2.detach().cpu().numpy()), axis=0)

                #mask_data
                # mask data
                y_trues_filtered = np.concatenate((y_trues_filtered, filtered_y_1.detach().cpu().numpy()), axis=0)
                y_preds_filtered = np.concatenate((y_preds_filtered, filtered_pred1.detach().cpu().numpy()), axis=0)
                y_trues_2_filtered = np.concatenate((y_trues_2_filtered, filtered_y_2.detach().cpu().numpy()), axis=0)
                y_preds_2_filtered = np.concatenate((y_preds_2_filtered, filtered_pred2.detach().cpu().numpy()), axis=0)
        if (run_type == "train"):
            loss.backward()
            optimizer.step()
        all_loss += loss.item() / y.shape[0]

    if (epoch != -1):
        run_type = "train: epoch " + str(epoch)

    if (model_name != "FLTADP"):
        return calEvalResult(all_loss, y_preds, y_trues, run_type, write_file)
    else:
        # y_preds_1, y_trues_1 whether active every day (category) : [data_num, label_num]
        # y_preds_2, y_trues_2 active days percentage (category) : [data_num, label_num]
        predict_results = y_preds, y_trues, y_preds_2, y_trues_2
        predict_results_filtered=y_preds_filtered,y_trues_filtered,y_preds_2_filtered,y_trues_2_filtered
        if (run_type == "train" and model.imbalance_stratage_enable != 0):
            model.FDS.update_last_epoch_stats(epoch+1)
            y_trues_2 = torch.tensor(y_trues_2)
            model.FDS.update_running_stats(training_features, y_trues_2, epoch)
        #calEvalResult_FLTADP(all_loss, predict_results_filtered, run_type, write_file,isMask=True)
        return calEvalResult_FLTADP(all_loss, predict_results, run_type, write_file)
'''
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import mean_squared_error

esp = 1e-5


# Result evaluation
def calEvalResult(loss_value, y_preds, y_trues, test_type, write_file=None):
    y_preds_bool = np.copy(y_preds)
    y_preds_bool[y_preds >= 0.5] = 1.0
    y_preds_bool[y_preds < 0.5] = 0.0

    eps = 1e-5
    y_trues_bool = np.copy(y_trues)
    y_trues_bool[y_trues >= eps] = 1.0
    y_trues_bool[y_trues < eps] = 0.0
    # auc
    #print(y_trues_bool)
    #print(y_preds)
    #print(y_trues_bool.shape)
    if y_trues_bool.sum()==y_trues_bool.shape[0]:
        y_trues_bool = np.append(y_trues_bool, 0)
        y_preds = np.append(y_preds, 0)
        y_trues = np.append(y_trues, 0)
    if y_trues_bool.sum()==0:
        y_trues_bool = np.append(y_trues_bool, 1)
        y_preds = np.append(y_preds, 1)
        y_trues = np.append(y_trues, 1)
    auc = metrics.roc_auc_score(y_trues_bool, y_preds)
    # rmse
    error = y_trues - y_preds
    rmse = (error ** 2).mean() ** 0.5
    # df
    df = abs(y_trues.mean() - y_preds.mean()) / y_trues.mean()

    print_str = '%20s loss %3.6f  auc  %.4f  rmse  %.4f  df(ActivateDay.Avg) %.4f' % (
        test_type, loss_value, auc, rmse, df)
    print(print_str)
    if (write_file != None):
        write_file.write(print_str + "\n")
    return auc, rmse, df


def calEvalResult_FLTADP(loss_value, predict_results, test_type, write_file=None,isMask=False):
    # y_preds_1, y_trues_1 whether active every day (category) : [data_num, label_num]
    # y_preds_2, y_trues_2 active days percentage (category) : [data_num,] \in [0,1]
    y_preds_1, y_trues_1, y_preds_2, y_trues_2 = predict_results

    y_auc_daylevel_activate = metrics.roc_auc_score(y_trues_1.reshape(-1), y_preds_1.reshape(-1))
    if isMask:
        y_trues_2=np.append(y_trues_2,0)
        y_preds_2=np.append(y_preds_2,0)
        #print(y_trues_2)
        #print(y_preds_2)
    y_auc_01_activate = metrics.roc_auc_score(y_trues_2 > 1e-5, y_preds_2)
    #print(y_auc_01_activate)
    # rmse
    error = (y_trues_2 - y_preds_2)
    rmse = (error ** 2).mean() ** 0.5

    # df: average number of active days in the future
    df = abs(y_preds_2.mean() - y_trues_2.mean()) / y_trues_2.mean()
    # print("df:", y_predicts_num2.mean(), y_trues_num2.mean())
    isMask_Str='  isMask'+str(isMask)
    print_str = '%20s loss %6.6f auc_day_activate %.4f auc_01_activate %.4f  rmse  %.4f  df(ActivateDay.Avg) %.4f' % (
        test_type, loss_value, y_auc_daylevel_activate, y_auc_01_activate, rmse, df)
    if isMask:
        print_str= print_str + isMask_Str
    print(print_str)
    if (write_file != None):
        write_file.write(print_str + "\n")
    return y_auc_01_activate, rmse, df


def run(epoch, datas, model, optimizer, device, model_name="None", run_type="train", lossFun='BCE', write_file=None):
    y_trues = np.array([])
    y_preds = np.array([])
    y_trues_2 = np.array([])
    y_preds_2 = np.array([])
    #mask
    y_trues_filtered = np.array([])
    y_preds_filtered = np.array([])
    y_trues_2_filtered = np.array([])
    y_preds_2_filtered = np.array([])

    training_features = None
    all_loss = 0
    if (run_type == "train"):
        model.train()
    else:
        model.eval()
    data_id = 0
    for data in datas:
        ui, uv, ai, av, y, time = data
        #print('ui_shape:',ui.shape)
        #print(time.shape)
        ui = ui.to(device)
        uv = uv.to(device)
        ai = ai.to(device)
        av = av.to(device)
        y = y.to(device)
        if (run_type == "train"):
            av_uv = torch.concat((av.reshape(-1, av.shape[1] * av.shape[2]), uv.reshape(-1, uv.shape[1])), dim=1)
            # training_feature [data_num, day * a_field_dim + u_field_dim]
            if (training_features == None):
                training_features = av_uv
            else:
                training_features = torch.concat((training_features, av_uv), dim=0)
            optimizer.zero_grad()
        if (model_name != "FLTADP"):
            # y:(Proportion of active days):batch_size * 1
            y = y[:, 0].reshape(-1, 1)
            loss, y_pred = model.forward(ui, uv, ai, av, y, lossFun)
            #print('y_pred:')
            #print(y_pred.shape)
            # y_trues is the list of proportion of active days
            y_trues = np.concatenate((y_trues, y.detach().cpu().numpy().reshape(-1)), axis=0)
            # y_pred is the activate or not in future days
            y_preds = np.concatenate((y_preds, y_pred.reshape(-1).detach().cpu().numpy()), axis=0)
        else:
            y_1 = y[:, 2:].detach().to(device)
            y_2 = y[:, 1].detach().long().to(device)
            y_2_input = F.one_hot(y_2, num_classes=model.future_day + 1).float()
            y_2 = y_2 / (model.future_day + 1)
            time = time.to(device)
            loss, y_pred_1, y_pred_2,filtered_y_1,filtered_y_2,filtered_pred1,filtered_pred2 = model.forward(ui, uv, ai, av, y_1, y_2_input, epoch,time)
            if (y_trues.shape[0] < 2):
                y_trues = y_1.detach().cpu().numpy()
                y_preds = y_pred_1.detach().cpu().numpy()
                y_trues_2 = y_2.detach().cpu().numpy()
                y_preds_2 = y_pred_2.detach().cpu().numpy()
                #mask data
                y_trues_filtered = filtered_y_1.detach().cpu().numpy()
                y_preds_filtered = filtered_pred1.detach().cpu().numpy()
                y_trues_2_filtered = filtered_y_2.detach().cpu().numpy()
                y_preds_2_filtered = filtered_pred2.detach().cpu().numpy()
            else:
                y_trues = np.concatenate((y_trues, y_1.detach().cpu().numpy()), axis=0)
                y_preds = np.concatenate((y_preds, y_pred_1.detach().cpu().numpy()), axis=0)
                y_trues_2 = np.concatenate((y_trues_2, y_2.detach().cpu().numpy()), axis=0)
                y_preds_2 = np.concatenate((y_preds_2, y_pred_2.detach().cpu().numpy()), axis=0)

                #mask_data
                # mask data
                y_trues_filtered = np.concatenate((y_trues_filtered, filtered_y_1.detach().cpu().numpy()), axis=0)
                y_preds_filtered = np.concatenate((y_preds_filtered, filtered_pred1.detach().cpu().numpy()), axis=0)
                y_trues_2_filtered = np.concatenate((y_trues_2_filtered, filtered_y_2.detach().cpu().numpy()), axis=0)
                y_preds_2_filtered = np.concatenate((y_preds_2_filtered, filtered_pred2.detach().cpu().numpy()), axis=0)
        if (run_type == "train"):
            loss.backward()
            optimizer.step()
        all_loss += loss.item() / y.shape[0]

    if (epoch != -1):
        run_type = "train: epoch " + str(epoch)

    if (model_name != "FLTADP"):
        return calEvalResult(all_loss, y_preds, y_trues, run_type, write_file)
    else:
        # y_preds_1, y_trues_1 whether active every day (category) : [data_num, label_num]
        # y_preds_2, y_trues_2 active days percentage (category) : [data_num, label_num]
        predict_results = y_preds, y_trues, y_preds_2, y_trues_2
        predict_results_filtered=y_preds_filtered,y_trues_filtered,y_preds_2_filtered,y_trues_2_filtered
        if (run_type == "train" and model.imbalance_stratage_enable != 0):
            model.FDS.update_last_epoch_stats(epoch+1)
            y_trues_2 = torch.tensor(y_trues_2)
            model.FDS.update_running_stats(training_features, y_trues_2, epoch)
        #calEvalResult_FLTADP(all_loss, predict_results_filtered, run_type, write_file,isMask=True)
        return calEvalResult_FLTADP(all_loss, predict_results, run_type, write_file)
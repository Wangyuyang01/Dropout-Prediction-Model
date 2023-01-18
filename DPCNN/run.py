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

def run(epoch, datas, model, optimizer, device, model_name="None", run_type="train", lossFun='BCE', write_file=None):
    y_trues = np.array([])
    y_preds = np.array([])
    y_trues_2 = np.array([])
    y_preds_2 = np.array([])
    
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
            optimizer.zero_grad()
        # y:(Proportion of active days):batch_size * 1
        y = y[:, 0].reshape(-1, 1)
        loss, y_pred = model.forward(ui, uv, ai, av, y, lossFun)
        #print('y_pred:')
        #print(y_pred.shape)
        # y_trues is the list of proportion of active days
        y_trues = np.concatenate((y_trues, y.detach().cpu().numpy().reshape(-1)), axis=0)
        # y_pred is the activate or not in future days
        y_preds = np.concatenate((y_preds, y_pred.reshape(-1).detach().cpu().numpy()), axis=0)
        
        if (run_type == "train"):
            loss.backward()
            optimizer.step()
        all_loss += loss.item() / y.shape[0]

    if (epoch != -1):
        run_type = "train: epoch " + str(epoch)
    return calEvalResult(all_loss, y_preds, y_trues, run_type, write_file)

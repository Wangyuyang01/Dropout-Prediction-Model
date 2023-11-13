# Dropout Prediction model project
The collection of the following Dropout Prediction algorithms:
* Student dropout prediction in massive open online courses by convolutional neural networks (DPCNN)
* Logistic Regression (LR)
## Setups
The requiring environment is as bellow:
* Ubantu
* Python 3+
* PyTorch
* Scikit-learn
* Numpy
* Pandas
```
conda create -n DPCNN python=3.6 -y
conda activate DPCNN
pip install torch==1.10.2
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```
You can also install the Python packages in env_create.sh.
```
$ sh env_create.sh
```
## Dataset
KDDCUP 2015：http://moocdata.cn/data/user-activity#User%20Activity
## Training and Running
Execute training process by main.py.
* batch_size: The batch size of the training process. Default: 64
* max_iter: The number of iterations. Default: 100
* learning_rate: The learning of the optimizer for the training process. Default: 0.005
* weight_decay: The weight decay of optimizer. Default: 1e-5
* seed: The random seed of dataset. Default: 1

The specific execution methods of each model are listed below:
1. DPCNN
  Here is an example for using DPCNN model:
  ```
  $ python main.py --model_name 'DPCNN' --weight_decay 1.00E-03 --dropout_p 0.2 --learning_rate 0.001 --seed 1 --LossFun 'MSE'  --DataSet  'KDD'
  ```
2. LR
  Here is an example for using LR model:
  ```
  $ python main.py --model_name 'LR' --weight_decay 1.00E-05  --learning_rate 0.001 --seed 1 --LossFun 'BCE'  --DataSet  'KDD'
  ```
The following bash command will help you:
```
$ python main.py -h
```
## Trainning Results
### KDD Data
use 7 predict 23

BCE

| Model |best_auc   | rmse  |  Hyperparameters  |
|:------|:-------------:|:-------------:|:-------------:|
| DPCNN | 0.7884 | 0.4480 |batch_size: 64 ,lr: 0.01 ,wd: 1e-4, dropout_p: 0.5|
| LR    | 0.7739 | 0.4371 | batch_size: 64 ,lr: 0.001 ,wd: 1e-5  |

MSE

| Model |best_auc   | rmse  |  Hyperparameters  |
|:------|:-------------:|:-------------:|:-------------:|
| DPCNN | 0.7709 | 0.1137 | batch_size: 64 ,lr: 0.001 ,wd: 1e-3, dropout_p: 0.2|
| LR    | 0.7255 | 0.4510 | batch_size: 64 ,lr: 0.001 ,wd: 1e-5                |



## 开题绘制的实验结果表（无用可删）
 ### kddcup15   
| MODEL | AUC  | RMSE  | MAE |
|:------|:-------------:|:-------------:|:-------------:|
| CAL	  | 0.8656±0.0412 | 0.4440±0.0049 | 0.3800±0.0065 | 
| CLSA  |	0.8551±0.0204 | 0.4233±0.0010 | 0.3595±0.0016 |
| CLMS	| 0.8899±0.0005 | 0.4303±0.0026 | 0.4022±0.0055 | 
| CNN	  | 0.8842±0.0438 | 0.4046±0.0011 | 0.3449±0.0021 | 
 ### XueTangX  
| MODEL | AUC  | RMSE  | MAE |
|:------|:-------------:|:-------------:|:-------------:|
| CAL	  | 0.9589±0.0065 | 0.3359±0.0808 | 0.2742±0.0972 |
| CLSA  | 0.9474±0.0117 | 0.3554±0.0358 | 0.3018±0.0579 |
| CLMS	| 0.9679±0.0004 | 0.3160±0.0043 | 0.2351±0.0047 |
| CNN	  | 0.9718±0.0155 | 0.3031±0.0120 | 0.2394±0.0331 |
 
## References
* DPCNN：[Student dropout prediction in massive open online courses by convolutional neural networks](https://link.springer.com/content/pdf/10.1007/s00500-018-3581-3.pdf?pdf=button)

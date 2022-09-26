# 实验要求：
# 训练数据包含2336条记录和22个字段，对训练数据进行一定的可视化数据分析（章节2.2）
# 利用训练数据，选择合适的信息作为特征建立回归模型，并预测测试数据成年人死亡率
# 可以使用基于 Python 的 Pandas 库进行数据相关处理，使用 Sklearn 库进行相关模型构建。
# 推荐使用基于 Python 的Sklearn库进行相关实验
# 数据中可能会有一些字段的值存在缺失
# By Jurio, 22/09/25
import sys
import time

import matplotlib.pyplot as plt
# 导入相关包
import pandas as pd
import sklearn
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# 数据读取和可视化分析
model_filename = './model.pkl'
imputer_filename = './imputer.pkl'
scaler_filename = './scaler.pkl'

train_data = pd.read_csv('./data/train_data.csv')
# train_y = train_data.iloc[:, -1].values
# train_data = train_data.drop(["Adult Mortality"], axis=1)
# x_train, x_test, y_train, y_test = train_test_split(train_data, train_y, random_state=666, test_size=0.2)


# def DataVisualization():
#     # 将相关性矩阵绘制成热力图
#     corr = train_data.corr()
#     corr.style.background_gradient(cmap='coolwarm')
#
#     # 利用seaborn检查可视化数据之间的依赖关系
#     sns.pairplot(train_data)


def preprocess_data(data, imputer=None, scaler=None):
    column_name = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
                   'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
                   'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
                   ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
                   'Schooling']
    data = data.drop(["Country", "Status"], axis=1)

    if imputer == None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = imputer.fit(data[column_name])
    data[column_name] = imputer.transform(data[column_name])

    if scaler == None:
        scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)

    data_norm = data_norm.drop(['Year'], axis=1)

    return data_norm, imputer, scaler


# def gridsearch():
#     # 需要网格搜索的参数
#     n_estimators = [i for i in range(450, 551, 20)]
#     max_depth = [i for i in range(5, 9)]
#     min_samples_split = [i for i in range(2, 4)]
#     min_samples_leaf = [i for i in range(1, 4)]
#     max_features = [i for i in range(15, 20)]
#     max_samples = [i / 100 for i in range(95, 100)]
#     parameters = {'n_estimators': n_estimators,
#                   'max_depth': max_depth,
#                   'min_samples_split': min_samples_split,
#                   'min_samples_leaf': min_samples_leaf,
#                   'max_features': max_features,
#                   'max_samples': max_samples}
    # # 随机森林回归
    # regressor = RandomForestRegressor(bootstrap=True, oob_score=True, random_state=1)
    #
    # best_model = GridSearchCV(regressor, parameters, refit=True, cv=6, verbose=1, n_jobs=-1)
    #
    # x_data_norm, imputer, scaler = preprocess_data(x_train)
    # train_x = x_data_norm.values
    # best_model.fit(train_x, y_train)
    #
    # joblib.dump(best_model, model_filename)
    # joblib.dump(imputer, imputer_filename)
    # joblib.dump(scaler, scaler_filename)
    #
    # return best_model


def model_fit(train_data=None):
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    x_data_norm, imputer, scaler = preprocess_data(train_data)

    train_x = x_data_norm.values

    # # 线性回归
    # regressor = lm.LinearRegression()
    # regressor.fit(train_x, train_y)

    # 随机森林回归
    regressor = RandomForestRegressor(bootstrap=True, oob_score=True, random_state=1,
                                      n_estimators=450, max_depth=8, max_features=18,
                                      max_samples=0.96, min_samples_leaf=1, min_samples_split=3)
    regressor.fit(train_x, train_y)

    joblib.dump(regressor, model_filename)
    joblib.dump(imputer, imputer_filename)
    joblib.dump(scaler, scaler_filename)

    return regressor


def predict(test_data, filename):
    loaded_model = joblib.load(model_filename)
    imputer = joblib.load(imputer_filename)
    scaler = joblib.load(scaler_filename)

    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values
    predictions = loaded_model.predict(test_x)

    return predictions


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # DataVisualization()

    # # GridSearch寻找最优模型参数组
    # t1 = time.time()
    # best_model = gridsearch()
    # t2 = time.time()
    # print(f'GridSearch cost:{t2-t1} s')
    # print(f'Best params:{best_model.best_params_}')
    # print(f'Best score:{best_model.best_score_}')

    # 使用最优参数组训练模型
    model = model_fit(train_data)
    # # 打印模型的截距
    # print(f'Model_intercept:{model.intercept_}')
    # # 打印模型的斜率
    # print(f'Model_coef:{model.coef_}')

    # 模型性能评估
    label = train_data.loc[:, 'Adult Mortality']
    data = train_data.iloc[:, :-1]
    # label = y_test
    # data = x_test
    y_pred = predict(data, './model.pkl')
    r2 = r2_score(label, y_pred)
    mse = mean_squared_error(label, y_pred)
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))

    sys.exit(0)

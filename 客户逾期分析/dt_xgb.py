#encoding: utf-8

import pandas as pd

# 读取数据+去重
def read_data(src, encoding)
    # 读取数据
    data = pd.read_csv(src, encoding=encodinng)
    print(data.shape)
    # 去重
    data.drop_duplicates(inplace=True)
    print(data.shape)
    return data
data = read_data('data/data.csv', 'GB18030')

# 查看每列的类型
def column_type(data):
    types = data.dtypes
    print data.select_dtypes(include=['bool']).shape
    print data.select_dtypes(include=['number']).shape
    print data.select_dtypes(include=['object']).shape
    print data.select_dtypes(include=['datetime']).shape
    print data.select_dtypes(include=['timedelta']).shape
    print data.select_dtypes(include=['category']).shape
column_type(data)


# 去除无关列
# 'source'和'bank_card_no'值无区分度
# ‘custid’、‘trade_no’、‘id_name’和预测无关
data.drop(['custid', 'trade_no', 'bank_card_no', 'id_name', 'source'], axis = 1, inplace=True)
# 'Unnamed: 0'和预测无关
data.drop(['Unnamed: 0'], axis = 1, inplace=True)

# 处理日期类型变量
def process_time(data, columns):
    data_time = pd.DataFrame()
    for column in columns:
        # 删除data这一列
        data.drop([column], axis = 1, inplace=True)
        column_month = pd.to_datetime(data[column]).dt.month
        column_weekday = pd.to_datetime(data[column]).dt.weekday
        column_day = pd.to_datetime(data[column]).dt.day
        data_time[column+'_month'] = column_month
        data_time[column+'_weekday'] = column_weekday
        data_time[column+'_day'] = column_day
    return data, data_time
data, data_time = process_time(data, ['first_transaction_time', 'latest_query_time', 'loans_latest_time'])
data_time = data_time.as_matrix()


# 统计各个列缺失值所占比例
for i in data.columns:
    d = len(data) - data[i].count()
    r = (float(d) / len(data)) * 100
    # rate = '%.2f%%' % r
    # print 'name: ', str(i).ljust(10),'d: ', str(d).ljust(4), 'rate: ', rate
    # print '%.2f%%' % r, i
# 由下图统计可以看出，‘student_feature’列缺失一半以上，且本列为类别类型，可以将缺失值用-1填充，相当于“是否缺失”当成另一种类别。
# 其他列缺失概率比较小，可以用中值填充。

# 缺失个数作为一种特征，衡量用户的信息完善程度
miss_rate = []
miss_number = []
for i in range(len(data)):
    temp = float((data[i:i+1]).count().sum()) / len(data.columns)
    miss_rate.append(temp)
    miss_number.append(len(data.columns) - (data[i:i+1]).count().sum())
print data.shape
data['miss_rate'] = miss_rate
print data.shape
data['miss_number'] = miss_number
print data.shape


# 处理类别特征
# 'regional_mobility'列的统计，按类别特征处理
data['regional_mobility'].value_counts()
# 'reg_preference_for_trad'列的统计，按类别特征处理
data['reg_preference_for_trad'].value_counts()
# 'student_feature'列的统计，按类别特征处理
data['student_feature'].value_counts()
# 'is_high_user'列的统计，按类别特征处理
data['is_high_user'].value_counts()
# 'status'列的统计，预测变量，正负样本接近1：3，可以不做处理。
data['status'].value_counts()

# 将刚刚被归类为类别变量和预测变量的列去掉，生成data_temp，数值特征为77维，类别特征为5维
data_temp = data
data_temp = data_temp.drop(['regional_mobility', 'reg_preference_for_trad', 'student_feature', 'is_high_user', 'status', 'miss_rate'], axis = 1)
print data_temp.shape
print data.shape


# 统计各个列标准差，将标准差小于0.1的特征剔除，数值特征变为71维
print (len(data_temp.columns))
for i in data_temp.columns:
    r = data_temp[i].std()
    print '%.2f' % r, i
    
    if r < 0.1:
        data_temp = data_temp.drop([i], axis = 1)
print (len(data_temp.columns))


# 接下来对类别特征和数值特征进行填充
# 数值特征和类别特征均用用中值进行填充
# 缺失值特征特别大的特征‘student_feature’用‘-1’填充
for i in data_temp.columns:
    temp = data_temp[i].isnull().sum()
    if temp:
        print i
        data_temp[i].fillna(data_temp[i].median(), inplace = True)

data_temp = data_temp.values
        
# 数值特征归一化 
# 从sklearn.preprocessing导入StandardScaler  
from sklearn.preprocessing import StandardScaler  
# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导  
# ss = StandardScaler()  
# fit_transform()先拟合数据，再标准化  
# data_temp = ss.fit_transform(data_temp)

a5 = data['miss_rate']
b5 = a5.as_matrix()
print b5
print b5.shape
b5 = b5.reshape(len(b5), 1)
print b5.shape

a6 = data['miss_number']
b6 = a6.as_matrix()
print b6
print b6.shape
b6 = b6.reshape(len(b6), 1)
print b6.shape

# 类别特征one-hot编码
def category_encoding(data, columns, fill_type):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from numpy import array
    from numpy import argmax
    for i in range(len(columns)):
        temp = data[columns[i]]
        if fill_type[i] == '-1':
            temp.fillna(-1, inplace=True)
        if fill_type[i] == 'median':
            temp.fillna(temp.median(), inplace=True)
        if fill_type[i] == 'max':
            temp.fillna(temp.max(), inplace=True)
        temp = temp.as_matrix()

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(temp)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = onehot_encoder.fit_transform(integer_encoded)
        print onehot_encoder
        print onehot_encoder.shape

        if i == 0:
            data_category = onehot_encoder
        else:
            np.hstack([data_category, onehot_encoder])

        return data_category
data_category = category_encoding(data, 
    ['student_feature', 'regional_mobility', 'reg_preference_for_trad', 'is_high_user'], 
    ['-1', 'median', 'max', 'max'])


# 特征矩阵X
print data_temp.shape
import numpy as np
X = np.hstack([data_temp, data_category, b5, b6, data_time])
print X.shape
# 预测变量y
y = data['status']
print y.shape


# 划分训练集测试集
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=1)
sss.get_n_splits(X, y)
print(sss)

for train_index,test_index in sss.split(X,y):
    print("Train Index:",train_index,",Test Index:",test_index)
    X_train, X_test=X[train_index],X[test_index]
    y_train, y_test=y[train_index],y[test_index]
    # print(X_train,X_test,y_train,y_test)


# 模型评估
def model_metrics(x_train, y_train, x_test, y_test, y_train_pred, y_test_pred, y_train_proba, y_test_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
    %matplotlib inline
    import matplotlib.pyplot as plt
    
    print '[准确率]'
    print '训练集：{:.4f}'.format(accuracy_score(y_train, y_train_pred))
    print '测试集：{:.4f}'.format(accuracy_score(y_test, y_test_pred))
    print '[精准率]'
    print '训练集：{:.4f}'.format(precision_score(y_train, y_train_pred))
    print '测试集：{:.4f}'.format(precision_score(y_test, y_test_pred))
    print '[召回率]'
    print '训练集：{:.4f}'.format(recall_score(y_train, y_train_pred))
    print '测试集：{:.4f}'.format(recall_score(y_test, y_test_pred))
    print '[f1 score]'
    print '训练集：{:.4f}'.format(f1_score(y_train, y_train_pred))
    print '测试集：{:.4f}'.format(f1_score(y_test, y_test_pred))
    print '[auc]'
    print '训练集：{:.4f}'.format(roc_auc_score(y_train, y_train_proba)) # auc函数也可以
    print '测试集：{:.4f}'.format(roc_auc_score(y_test, y_test_proba))
    
    print '[roc曲线]'
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba, pos_label = 1)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba, pos_label = 1)
    
    label = ["Train - AUC:{:.4f}".format(auc(fpr_train, tpr_train)), 
             "Test - AUC:{:.4f}".format(auc(fpr_test, tpr_test))]
    
    plt.plot(fpr_train,tpr_train)
    plt.plot(fpr_test,tpr_test)
    plt.plot([0, 1], [0, 1], 'd--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(label, loc = 4)
    plt.title("ROC curve")

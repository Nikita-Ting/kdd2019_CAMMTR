#coding:utf-8


import pandas as pd
import numpy as np
import os


#==============================变量值计算函数================================
#时间间隔均值，方差(单位:天)
def time_gap(times):
    """
    :param times:pandas.Series
    交易订单的时间列表
    """
    try:
        times = pd.to_datetime(pd.Series(times).drop_duplicates())
        if len(times)==1:
            mean = np.nan
            std = np.nan
        else:
            times = times.sort_values()
            gaps = (times -times.shift()).dropna()
            gaps = gaps.map(lambda x: x/np.timedelta64(1,'D')) #时间间隔换算为天
            if gaps.shape[0]==1:
                mean = gaps.iloc[0]
                std = 0
            else:
                mean = gaps.mean()
                std =gaps.std()
        return mean,std
    except:
        return np.nan,np.nan


#==============================变量统计分析函数===============================
#空值率
def null_ratio(df):
    """
    :param df: pandas.DataFrame
    样本的所有候选变量值矩阵
    """
    dic ={}
    for col in df.columns:
        r = df[df[col].isnull()].shape[0]*1.0/df.shape[0]
        dic[col]=r
    return dic


#计算所有变量IV值
def calc_ivs(df, label_thr=None):
    """
    :param df: pandas.DataFrame
    样本候选变量值矩阵
    :param label_thr: float
    若label列是连续型数值，则根据指定的阈值label_thr对label进行离散化
    """
    if df['label'].unique().size > 2:  # label为连续型变量
        df['label'] = df['label'].apply(lambda x: 1 if x > label_thr else 0)
    cols = df.drop('label', axis=1).columns
    ivs = {}
    for col in cols:
        gf = calc_woe(df[[col, 'label']], col)
        ivs[col] = gf.iv.sum()
    return ivs

# 计算某个变量的woe,iv值
def calc_woe(df, col, bin_num=3):
    """
    :param df: pandas.DataFrame
    样本候选变量值矩阵
    :param col: string
    指定计算IV值的变量名
    :param bin_num: int
    指定连续型变量分段数
    """
    df_sub = df.copy()
    if 'label' not in df_sub.columns:
        print( 'label not in the columns')
        return
    total_good = len(df_sub[df_sub.label == 0])
    total_bad = len(df_sub[df_sub.label == 1])
    total_count = df_sub.shape[0]
    rank_col = col
    #连续型变量，则需要先进行分段编号处理
    if df_sub[col].unique().size > 12:
        rank_col = col + '_rank'
        df_sub[rank_col] = df_sub[col].rank(method='max') / (total_count / bin_num * 1.0)
        df_sub.loc[df_sub[df_sub[col].isnull()].index, rank_col] = np.nan  # 空值复原
        df_sub[rank_col] = df_sub[rank_col].apply(lambda x: int(x) if x > 0 else np.nan)
        tmp = df_sub[df_sub[rank_col] == bin_num]
        if tmp.shape[0] < (total_count / (bin_num * 2)) and tmp.shape[0] > 0:  # 最后一组与前一组合并
            df_sub[rank_col].loc[tmp.index] = bin_num - 1

    # 分组统计
    grouping_data = []
    for gname, gdata in df_sub.groupby(rank_col):
        g_info = {}
        g_info['name'] = gname
        g_info['0_num'] = len(gdata[gdata['label'] == 0])
        g_info['0_ratio'] = g_info['0_num'] * 1.0 / total_good
        g_info['1_num'] = len(gdata[gdata['label'] == 1])
        g_info['1_ratio'] = g_info['1_num'] * 1.0 / total_bad

        if g_info['0_num'] > 0 and g_info['1_num'] > 0:
            g_info['woe'] = np.math.log(1.0 * g_info['1_ratio'] / g_info['0_ratio'])
        elif g_info['0_num'] == 0:
            g_info['woe'] = -1
        else:
            g_info['woe'] = 1
        g_info['iv'] = 1.0 * (g_info['1_ratio'] - g_info['0_ratio']) * g_info['woe']
        grouping_data.append(g_info)
    g_df = pd.DataFrame(grouping_data, columns=g_info.keys())
    return g_df


#==============================变量预处理函数==================================
#woe值字典
def woe_to_dict(df):
    """
    将指定列转换为字典格式
    :param df: pandas.DataFrame
    以列格式存储的数据
    :return:
    转换后的数据字典
    """
    dict ={}
    for i in xrange(df.shape[0]):
        dict[df['name'].iloc[i]] = df['woe'].iloc[i]
    return dict

#变量归一化
def normalize(sf):
    """
    :param sf: pandas.Series
     指定归一化处理的变量
    """
    gap = sf.max()-sf.min()
    return (sf-sf.min())/gap




#==============================混淆矩阵计算与可视化==================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes_name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes_name
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label'
          )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
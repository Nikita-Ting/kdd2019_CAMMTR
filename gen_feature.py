import pandas as pd
import numpy as np
from tqdm import tqdm
import json 
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score

#============baseline=====================
def gen_plan_feas(data):
    n                                           = data.shape[0]
    mode_list_feas                              = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist     = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta         = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = \
    np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
  
    mode_texts = []
    for i, plan in tqdm(enumerate(data['plans_json'].values)):
        if len(plan) == 0:
            cur_plan_list   = []
        else:
            cur_plan_list   = plan
        if len(cur_plan_list) == 0:
            mode_list_feas[i, 0] =  1
            first_mode[i]        =  0
            max_dist[i]          = -1
            min_dist[i]          = -1
            mean_dist[i]         = -1
            std_dist[i]          = -1
            max_price[i]         = -1
            min_price[i]         = -1
            mean_price[i]        = -1
            std_price[i]         = -1
            max_eta[i]           = -1
            min_eta[i]           = -1
            mean_eta[i]          = -1
            std_eta[i]           = -1
            min_dist_mode[i]     = -1
            max_dist_mode[i]     = -1
            min_price_mode[i]    = -1
            max_price_mode[i]    = -1
            min_eta_mode[i]      = -1
            max_eta_mode[i]      = -1
            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            for tmp_dit in cur_plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            distance_list                = np.array(distance_list)
            price_list                   = np.array(price_list)
            eta_list                     = np.array(eta_list)
            mode_list                    = np.array(mode_list, dtype='int')
            mode_list_feas[i, mode_list] = 1
            distance_sort_idx            = np.argsort(distance_list)
            price_sort_idx               = np.argsort(price_list)
            eta_sort_idx                 = np.argsort(eta_list)
            max_dist[i]                  = distance_list[distance_sort_idx[-1]]
            min_dist[i]                  = distance_list[distance_sort_idx[0]]
            mean_dist[i]                 = np.mean(distance_list)
            std_dist[i]                  = np.std(distance_list)
            max_price[i]                 = price_list[price_sort_idx[-1]]
            min_price[i]                 = price_list[price_sort_idx[0]]
            mean_price[i]                = np.mean(price_list)
            std_price[i]                 = np.std(price_list)
            max_eta[i]                   = eta_list[eta_sort_idx[-1]]
            min_eta[i]                   = eta_list[eta_sort_idx[0]]
            mean_eta[i]                  = np.mean(eta_list)
            std_eta[i]                   = np.std(eta_list)
            first_mode[i]                = mode_list[0]
            max_dist_mode[i]             = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i]             = mode_list[distance_sort_idx[0]]
            max_price_mode[i]            = mode_list[price_sort_idx[-1]]
            min_price_mode[i]            = mode_list[price_sort_idx[0]]
            max_eta_mode[i]              = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i]              = mode_list[eta_sort_idx[0]]
    feature_data                   =  pd.DataFrame(mode_list_feas)
    feature_data.columns           =  ['mode_feas_{}'.format(i) for i in range(12)]
    feature_data['max_dist']       =  max_dist
    feature_data['min_dist']       =  min_dist
    feature_data['mean_dist']      =  mean_dist
    feature_data['std_dist']       =  std_dist
    feature_data['max_price']      = max_price
    feature_data['min_price']      = min_price
    feature_data['mean_price']     = mean_price
    feature_data['std_price']      = std_price
    feature_data['max_eta']        = max_eta
    feature_data['min_eta']        = min_eta
    feature_data['mean_eta']       = mean_eta
    feature_data['std_eta']        = std_eta
    feature_data['max_dist_mode']  = max_dist_mode
    feature_data['min_dist_mode']  = min_dist_mode
    feature_data['max_price_mode'] = max_price_mode
    feature_data['min_price_mode'] = min_price_mode
    feature_data['max_eta_mode']   = max_eta_mode
    feature_data['min_eta_mode']   = min_eta_mode
    feature_data['first_mode']     = first_mode
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    plan_fea = pd.concat([feature_data, mode_svd], axis=1)
    plan_fea['sid'] = data['sid'].values
    return plan_fea


#=======================================
from sklearn.metrics import accuracy_score,recall_score,precision_score
def get_weighted_fscore(y_pred, y_true):
    dic_ = y_true.value_counts(normalize = True)
    f_score = 0
    print('class  ','weight  ','f1_score_class  ','pricision  ','recall  ')
    for i in range(12):
        yt = y_true == i
        yp = y_pred == i
        f_score += dic_[i] * f1_score(y_true=yt, y_pred= yp)        
        print(i,dic_[i],f1_score(y_true=yt, y_pred= yp), precision_score(y_true=yt, y_pred= yp),recall_score(y_true=yt, y_pred= yp))
    print('f1_score:',f_score)

    
#====================plans feature============================    
#=================dict to pd.DataFrame==================
def transform_plans(plans):
    #plans dic
    plans_rows = []
    for p_dics in plans['plans'].values:
        plans_rows.extend(p_dics)
    #plans info
    plans['plan_num'] = plans['plans'].apply(len)
    sid_list=[]
    plan_num=[]
    for sid,num in plans[['sid','plan_num']].values:
        sid_list.extend([sid]*num)
        plan_num.extend([num]* num)
    #concat
    df_plan = pd.DataFrame(sid_list,columns=['sid'])
    df_plan['plan_num']=plan_num
    df_plan = pd.concat([df_plan,pd.DataFrame(plans_rows)],axis=1)
    
    df_plan['price'] = df_plan['price'].apply(lambda x:int(x) if x!='' else 0)
    df_plan['speed']= df_plan['distance']*1.0 / df_plan['eta']
    df_plan['time_cost']= df_plan['price']*1.0 / df_plan['eta'] 
    df_plan['plan_order']= df_plan.groupby(['sid']).cumcount()+1
    return df_plan


def gen_plan_rank_feature(df):
    #值排序  
    df_plan = df.copy()
    df_plan['eta_rank'] = df_plan.groupby(['sid'])['eta'].rank(method='first')
    df_plan['distance_rank'] = df_plan.groupby(['sid'])['distance'].rank(method='first')
    df_plan['price_rank'] =  df_plan.groupby('sid')['price'].rank(method='first')
    df_plan['speed_rank'] = df_plan.groupby(['sid'])['speed'].rank(method='first')
    df_plan['time_cost_rank'] = df_plan.groupby(['sid'])['time_cost'].rank(method='first')
    df_plan['mode_distance_rank'] = df_plan.groupby(['transport_mode'])['distance'].rank(method='first')
    df_plan['mode_eta_rank'] = df_plan.groupby(['transport_mode'])['eta'].rank(method='first')
    df_plan['mode_price_rank'] = df_plan.groupby(['transport_mode'])['price'].rank(method='first')
    rank_feature=['eta_rank','distance_rank','price_rank','speed_rank','time_cost_rank','mode_distance_rank','mode_eta_rank','mode_price_rank']
    return df_plan,rank_feature


def feature_describe(df, by_key, on_col, cols=[]):
    tmp_df = df.groupby(by_key)[on_col].agg(cols)
    tmp_df.columns = [by_key+'_'+on_col+'_'+col for col in cols]   
    df = df.merge(tmp_df.reset_index(),how='left',on=by_key)
    return df

def gen_plan_statis_feature(df): 
    #值统计特征
    df_plan = df.copy()
    feature_org = set(df_plan.columns)
    df_plan = feature_describe(df_plan,'sid','distance',['mean','std','min','max'])
    df_plan = feature_describe(df_plan,'sid','eta',['mean','std','min','max'])    
    df_plan = feature_describe(df_plan,'sid','price',['mean','std','min','max'])  
    df_plan = feature_describe(df_plan,'sid','speed',['mean','std','min','max'])  
    df_plan = feature_describe(df_plan,'transport_mode','distance',['mean', 'std', 'min', 'median', 'max'])
    df_plan = feature_describe(df_plan,'transport_mode','eta',['mean', 'std', 'min', 'median', 'max'])
    df_plan = feature_describe(df_plan,'transport_mode','price',['mean', 'std', 'min','median', 'max'])
    df_plan = feature_describe(df_plan,'transport_mode','speed',['mean', 'std', 'min', 'median', 'max'])
    statis_feature = set(df_plan.columns) - feature_org
    return df_plan,list(statis_feature)  

def gen_plan_feature():
    plans_train = pd.read_csv("data/train_plans.csv", parse_dates=['plan_time'])
    plans_test = pd.read_csv('data/test_plans.csv', parse_dates=['plan_time'])
    plans = pd.concat([plans_train,plans_test],axis=0)
    plans['plans'] = plans['plans'].apply(eval)
    #extract plan features
    df_plan = transform_plans(plans)
    plan_feature = [col for col in df_plan.columns if col!='sid']
    
    #extract rank feature
    df_plan,rank_feature = gen_plan_rank_feature(df_plan)
    
    #extract statis_feature
    df_plan,statis_feature = gen_plan_statis_feature(df_plan)
    
    #extract time features
    time_feature = []
    plans['plan_day'] = plans['plan_time'].apply(lambda x :x.day)
    plans['plan_hour'] = plans['plan_time'].apply(lambda x :x.hour)
    plans['plan_weekday'] = plans['plan_time'].apply(lambda x :x.dayofweek+1)
    time_feature.extend(['plan_day','plan_hour','plan_weekday'])
    
    #merge
    df_plan=df_plan.merge(plans[['sid','plan_time','plan_day','plan_hour','plan_weekday']],how='left',on='sid')
    return df_plan,plan_feature,rank_feature,statis_feature,time_feature


#===================od feature===================================
def get_haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def gen_query_feature():
    #query data
    train_query = pd.read_csv("data/train_queries.csv", usecols =['sid','pid','o','d'])
    test_query = pd.read_csv("data/test_queries.csv", usecols =['sid','pid','o','d'])
    query = pd.concat([train_query,test_query],axis=0)

    #split o,d
    o_split = query['o'].apply(lambda x:[float(x) for x in x.split(',')])
    d_split = query['d'].apply(lambda x:[float(x) for x in x.split(',')])
    query['o_x'] = o_split.apply(lambda x:x[0])
    query['o_y'] = o_split.apply(lambda x:x[1])
    query['d_x'] = d_split.apply(lambda x:x[0])
    query['d_y'] = d_split.apply(lambda x:x[1])
    query['haversine'] = query.apply(lambda x:get_haversine_np(x['o_y'],x['o_x'],x['d_y'],x['d_x']),axis=1)
    query_feature = ['o_x','o_y','d_x','d_y','haversine']
    return train_query,test_query,query,query_feature

#矩阵分解
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
def get_svd_feature(train,by_key,on_key,size=5):
    train[on_key] = train[on_key].astype(str)
    tmp = train.groupby(by=by_key)[on_key].apply((lambda x :' '.join(x))).reset_index()
    mode_texts = tmp[on_key].values
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=size, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    if type(by_key)==list:
        svd_feature = ['_'.join(by_key)+'_'+on_key+'_'+'svd_{}'.format(i) for i in range(size)]
    else:
        svd_feature = [by_key+'_'+on_key+'_'+'svd_{}'.format(i) for i in range(size)]
    mode_svd.columns  = svd_feature
    svd_df = pd.concat([tmp[by_key],mode_svd],axis=1)
    return svd_df,svd_feature


from gensim.models import Word2Vec
import multiprocessing
def item2vec(df,name,concat_name,size=8,window=2,model_iter=3,train_iter = 5):
    df[concat_name] = df[concat_name].astype(str)
    res = df.groupby(name)[concat_name].apply((lambda x :'ddd'.join(x))).reset_index()
    res.columns = [name,'%s_doc'%concat_name]
    sentence = []
    for line in list(res['%s_doc'%concat_name].values):
        sentence.append(line.split('ddd'))
    print('training...')
    model = Word2Vec(sentence,size=size, window=window, min_count=1,workers=multiprocessing.cpu_count(),iter=model_iter)
    for i in range(train_iter):
        for t in sentence:
            random.shuffle(t)
        model.train(sentence,total_examples=model.corpus_count,epochs=model.epochs)
    print('outputing...')
    final_doc = df[[concat_name]].drop_duplicates().reset_index(drop=True)
    final_emb = final_doc[concat_name].apply(lambda x:list(model[str(x)]))
    item_feature = [name+'_'+concat_name+'_'+'emb_'+str(i) for i in range(len(final_emb.values[0]))]
    d_doc_em = pd.concat([final_doc[[concat_name]],
        pd.DataFrame(list(final_emb.values),columns=item_feature)],axis=1)
#     df = pd.merge(df,d_doc_em,on=concat_name,how='left')
    return d_doc_em,item_feature


#doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import random
def gen_doc2vec_feature(df,on_key,size):
#     df[on_key] = df[on_key].astype(str)
#     tmp = df.groupby(by_key)[on_key].apply(list).reset_index()
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df[on_key].values)]
    print('doc2vec training...')
    model = Doc2Vec(documents, vector_size=size, window=2, min_count=1, workers=4)
    print('doc2vec over')
    df_doc = pd.DataFrame()
    doc_vec = df[on_key].apply(lambda x:model.infer_vector(x))
    for i in range(size):
        df_doc[on_key+'_doc_'+str(i)] = doc_vec.apply(lambda x:x[i])
    cols = [on_key+'_doc_'+str(i) for i in range(size)]
    return df_doc,cols  


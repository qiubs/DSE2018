# import cell and define data paths
import copy
import pandas as pd
import missingno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import scipy
from matplotlib import rc
import warnings
from scipy.stats import probplot
import matplotlib.gridspec as gridspec
import seaborn as sns
from . import zillow_data_loader
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot
from sklearn.preprocessing import QuantileTransformer
from catboost import CatBoostRegressor
from catboost import Pool

data_dir = "~/kaggle/zillow/"

def fetch_experiment_datasets(data_dir = data_dir,rand_seed = 3545):
    '''function to retrieve zillow data and split into monthly holdout emulating the original experiment - see report
    '''
    #load data, add month_year col
    merged,cat_encoders = zillow_data_loader.load_train_data(data_dir,fill_and_enc_cats=True)
    merged['month_year'] = merged['transaction_month'].astype(str)+"_"+merged['transaction_year'].astype(str)

    rand = np.random.RandomState(rand_seed)
    holdout_set_list, train_set_list = [],[]
    # take all cases from one month, split from three months
    holdout_set_list.append(merged[(merged['month_year'] == "9_2017")])
    holdout_half = ['2_2016',"9_2016",'2_2017']

    merged = merged[(merged['month_year'] != "9_2017")]
    for this_date in holdout_half:
        date_subset = merged[(merged['month_year'] == this_date)]
        if date_subset.shape[0]!=0:
            #print(this_date)
            merged = merged[(merged['month_year'] != this_date)]
            index_list = list(range(0, date_subset.shape[0]))
            sample_idx = rand.choice(index_list,
                                     size = int(date_subset.shape[0]/2),
                                     replace = False)
            other_idx = [x for x in index_list if x not in sample_idx]
            holdout_set_list.append(date_subset.iloc[sample_idx])
            train_set_list.append(date_subset.iloc[other_idx])
    train_set_list.append(merged)
    train_set = pd.concat(train_set_list)
    holdout_set = pd.concat(holdout_set_list)

    train_set['month_year'] = train_set['transaction_month'].astype(str)+"_"+train_set['transaction_year'].astype(str)
    holdout_set['month_year'] = holdout_set['transaction_month'].astype(str)+"_"+holdout_set['transaction_year'].astype(str)
    print("train_set shape",train_set.shape)
    print("holdout_set shape",holdout_set.shape)
    return train_set, holdout_set

def fix_nan_train_test(train_data,test_data,
                       numerical_vars,
                       numerical_vars_to_log=None):
    '''function to replace nans in dataset
    '''
    for num_var in numerical_vars:
        train_data[num_var]=train_data[num_var].fillna(train_data[num_var].median())
        test_data[num_var]=test_data[num_var].fillna(test_data[num_var].median())
    if numerical_vars_to_log != None:
        for num_var in numerical_vars_to_log:
            #get median of test set, apply to nas
            train_data['log_'+num_var]=np.log(train_data[num_var])
            test_data['log_'+num_var]=np.log(test_data[num_var])
            log_med = np.median(train_data[num_var].values)
            train_data[num_var]=train_data[num_var].fillna(log_med)
            test_data[num_var]=test_data[num_var].fillna(log_med)
    return train_data,test_data

def parce_df_into_xy(in_df,list_of_feats,target_var='logerror'):
    x_df = in_df[list_of_feats]
    y_df = in_df[target_var]
    return x_df,y_df

def init_lgb_model():
    'lgb_model with minor hyper parameter tuning'
    params = {}
    params['max_bin'] = 100
    params['learning_rate'] = 0.0021 # shrinkage_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'mae'          # or 'mae'
    params['sub_feature'] = 0.5      # feature_fraction
    params['bagging_fraction'] = 0.85 # sub_row
    params['bagging_freq'] = 40
    params['num_leaves'] = 512        # num_leaf
    params['min_data'] = 500         # min_data_in_leaf
    params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    model_lgb = lgb.LGBMRegressor(**params)
    return model_lgb

def cat_fit_predict_validation(k_folds,in_df,train_features,cat_feats,in_model,
                              sk_scaler=None,stat_dist=None,rand_seed = 3545):
    trained_models, y_preds_list, y_actuals_list,scaler_list,stat_params = [],[],[],[],[]
    y_df = in_df.logerror
    train_df = in_df[train_features]
    month_year_df = in_df['month_year']
    #print(month_year_df)
    skf = StratifiedKFold(n_splits=k_folds,shuffle = True,random_state = rand_seed)
    skf.get_n_splits(train_df, month_year_df)
    k_counter = 1
    for train_index, test_index in skf.split(train_df, month_year_df):
        print("training ", k_counter, "of", k_folds)
        k_counter+=1
        train_x, val_x = train_df.iloc[train_index], train_df.iloc[test_index]
        train_y, val_y = y_df.iloc[train_index], y_df.iloc[test_index]
        #feature_name = ['feature_' + str(col) for col in train_x.colnames]
        if sk_scaler:
            use_scaler= copy.deepcopy(sk_scaler)
            train_y = use_scaler.fit_transform(np.array(train_y).reshape(-1, 1))
            val_y = use_scaler.transform(np.array(val_y).reshape(-1, 1)).flatten()
            scaler_list.append(use_scaler)
        if stat_dist:
            stat_param = stat_dist.fit(train_y)
            stat_params.append(stat_param)
            train_y = stat_dist.cdf(train_y,*stat_param)
            val_y = stat_dist.cdf(val_y,*stat_param)

        cat_feature_inds = [i for i,c in enumerate(train_features) if c in cat_feats]

        val_data = Pool(val_x, val_y,
                    cat_features=cat_feature_inds)

        train_data = Pool(train_x, train_y,
                        cat_features=cat_feature_inds)
        model.fit(train_data,
                  use_best_model = True,
                  eval_set = val_data,
                 logging_level = 'Silent')
        train_preds = model.predict(train_x)
        preds = model.predict(val_x)
        trained_models.append(model)
        y_preds_list.extend(preds)
        y_actuals_list.extend(val_y)

    print("abs validation residuals:", np.average(np.abs(val_y-preds)))
    output = {'models':trained_models,
             'y_preds':y_preds_list,
             'y_actuals':y_actuals_list,
             'scalers':scaler_list,
             'stat_dist':stat_dist,
             'stat_params':stat_params}
    return output



def make_submission(trained_model_list,feat_list,
                    test_df,name="sub.csv"):
    submission = pd.DataFrame({'ParcelId': test_df['ParcelId']})
    test_dates = [('201610',pd.Timestamp('2016-10-01')),
                    ('201611',pd.Timestamp('2016-11-01')),
                    ('201612',pd.Timestamp('2016-12-01')),
                    ('201710',pd.Timestamp('2017-10-01')),
                    ('201711',pd.Timestamp('2017-11-01')),
                    ('201712',pd.Timestamp('2017-12-01'))]
    #save_list = []
    for label, test_date in test_dates:
        y_pred = 0.0
        test_df['transaction_month'] = test_date.month
        test_df['transaction_year'] = test_date.year

        for ith,this_model in enumerate(trained_model_list):
            print("Predicting for: %s %s model " % (label,ith))
            y_pred += this_model.predict(test_df[feat_list])

        y_pred /= len(trained_model_list)
        submission[label] = y_pred
    #save_list.append(y_pred)
    submission.to_csv(name, float_format='%.5f',index=False)
    print("submission created as ",name)

def init_cat_model():
    model = CatBoostRegressor(
            iterations=1000, learning_rate=0.05,
            #leaf_estimation_method='Gradient',
            depth=3, l2_leaf_reg=3,
            #one_hot_max_size = 50,
            #boosting_type='Plain',
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=3545)
    return model

def lgb_fit_predict_validation(k_folds,in_df,
                               train_features,cat_feats,in_model,sk_scaler=None,
                               stat_dist=None,rand_seed = 3545):
    trained_models, y_preds_list, y_actuals_list,scaler_list,stat_params = [],[],[],[],[]
    stat_loc_list = []
    y_df = in_df.logerror
    train_df = in_df[train_features]
    month_year_df = in_df['month_year']
    #print(month_year_df)
    skf = StratifiedKFold(n_splits=k_folds,shuffle = True,random_state = rand_seed)
    skf.get_n_splits(train_df, month_year_df)
    k_counter = 1
    for train_index, test_index in skf.split(train_df, month_year_df):
        print("training ", k_counter, "of", k_folds)
        k_counter+=1
        train_x, val_x = train_df.iloc[train_index], train_df.iloc[test_index]
        train_y, val_y = y_df.iloc[train_index], y_df.iloc[test_index]
        #feature_name = ['feature_' + str(col) for col in train_x.colnames]
        if sk_scaler:
            use_scaler= copy.deepcopy(sk_scaler)
            train_y = use_scaler.fit_transform(np.array(train_y).reshape(-1, 1))
            val_y = use_scaler.transform(np.array(val_y).reshape(-1, 1)).flatten()
            scaler_list.append(use_scaler)
        if stat_dist:
            stat_param = stat_dist.fit(train_y)
            stat_params.append(stat_param)
            train_y = stat_dist.cdf(train_y,*stat_param)
            val_y = stat_dist.cdf(val_y,*stat_param)

        model_lgb = copy.deepcopy(in_model)
        train_ds = lgb.Dataset(train_x,label=train_y, categorical_feature=cat_feats)
        model_lgb.fit(train_x, train_y, eval_set=[(val_x, val_y)],
            eval_metric= 'mae', verbose=0, early_stopping_rounds= 100)
        train_preds = model_lgb.predict(train_x)
        preds = model_lgb.predict(val_x)
        trained_models.append(model_lgb)
        y_preds_list.extend(preds)
        y_actuals_list.extend(val_y)
    print("abs validation residuals:", np.average(np.abs(val_y-preds)))
    output = {'models':trained_models,
             'y_preds':y_preds_list,
             'y_actuals':y_actuals_list,
             'scalers':scaler_list,
             'stat_dist':stat_dist,
             'stat_params':stat_params}
    return output

def make_res_plot(y_preds,y_actual):
    f, (ax_left, ax_right) = plt.subplots(1,2,figsize=(12,4))
    y_preds=np.array(y_preds).flatten()
    y_actual=np.array(y_actual).flatten()
    res = y_preds-y_actual
    sns.residplot(x=y_actual, y=res, ax=ax_left,
                  color = "black",scatter_kws={'s':2,'alpha':.5})
    sns_plot_1=probplot(res, plot= ax_right)
    ax_left.set( title = "Y_actual vs Residual")
    ax_left.set(ylabel = 'residual',xlabel='y-actual')
    ax_right.set( title = "Residual QQ-Plot")
    print(np.average(np.abs(y_preds-y_actual)), 'res')
    plt.show()

def makedist_qq_plot(y_actual):
    f, (ax_left, ax_right) = plt.subplots(1,2,figsize=(12,4))
    sns.distplot(y_actual, ax=ax_box,
                  color = "black")
    sns_plot_1=probplot(y_actual, plot= ax_hist)
    ax_left.set( title = "Y_actual vs Residual")
    ax_right.set( title = "Residual QQ-Plot")
    plt.show()


def stack_model_pred(model_list,in_data,train_features,target_name):
    y_actual=in_data[target_name].values
    y_pred = []
    x_test = in_data[train_features]
    for ith, this_model in enumerate(model_list['models']):
        fold_pred = this_model.predict(x_test)
        if len(model_list['scalers'])>0:
            fold_pred = model_list['scalers'][ith].inverse_transform(fold_pred.reshape(-1, 1))
        if len(model_list['stat_params'])>0:
            fold_pred = model_list['stat_dist'].ppf(fold_pred, *model_list['stat_params'][ith])
        y_pred.append(fold_pred)
    y_pred = np.array(y_pred)
    #print(y_pred.shape)
    y_pred = np.average(y_pred,axis=0)
    make_res_plot(y_pred,y_actual)

    print("abs holdout residuals:", np.average(np.abs(y_pred-y_actual)))

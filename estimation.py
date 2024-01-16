import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, auc, roc_curve
from scipy.optimize import curve_fit
import pickle

def set_target_label(df):
    data = df.copy()
    data['def_date'] = pd.to_datetime(data['def_date'], errors='coerce')
    data['def_present'] = pd.notna(data['def_date']).astype(int)

    # Convert dates into datetime format for easier use
    data['stmt_date'] = pd.to_datetime(data['stmt_date'])
    data.loc[data['def_present'] == 1, 'def_date'] = pd.to_datetime(data['def_date'])  # Error handles 12/31/99

    # Since financial statements are only available in May of next year, set a new variable
    data['usable_date'] = pd.to_datetime((data['fs_year'] + 1).astype(str) + '-05-01')
    data['default_label'] = 0

    # Define the prediction year range based on 'usable_date'. This will train the model to see if the current data
    # will predict default over the next 12 months.
    data['pred_year_start'] = data['usable_date']
    data['pred_year_end'] = data['usable_date'] + pd.DateOffset(years=1)

    # Assigning default label if it falls within the above window
    data.loc[(data['def_present'] == 1) & \
           (data['def_date'] >= data['pred_year_start']) & \
           (data['def_date'] <= data['pred_year_end']), 'default_label'] = 1

    return data

# Imputes the variables before feature engineering (avoid inf or -inf)
def nan_var_replace_pre_feat(df):
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df['exp_financing'].fillna(df['inc_financing'] - df['prof_financing'], inplace=True)
    df['asst_current'].fillna(df['AR'] + df['cash_and_equiv'] + df['cf_operations'], inplace=True)
    df['asst_current'].fillna(df['asst_tot'] - (df['asst_fixed_fin'] + df['asst_tang_fixed'] + df['asst_intang_fixed']), inplace=True)
    df['cf_operations'].fillna(df['profit'], inplace=True)
    df['cf_operations'].fillna(df['cash_and_equiv'], inplace=True)
    df['eqty_tot'].fillna(df['asst_tot'] - (df['liab_lt'] + df['debt_bank_st'] + df['debt_bank_lt'] + df['debt_fin_st'] + df['debt_fin_lt'] + df['AP_st'] + df['AP_lt']), inplace=True)
    df['AP_st'].fillna(df['debt_st'], inplace=True)
    df['roe'].fillna(df['profit'] / df['eqty_tot'], inplace=True)
    df['roa'].fillna(df['profit'] / df['asst_tot'], inplace=True)
    df['days_rec'].fillna((df['AR'] / df['rev_operating']) * 365, inplace=True) 
    
    return df
    
# Do feature engineering
def feature_engineering(df):

    df['debt_ratio'] = (df['liab_lt'] + df['debt_bank_st'] + df['debt_bank_lt'] + df['debt_fin_st'] + df['debt_fin_lt'] + df['AP_st'] + df['AP_lt']) / df['asst_tot']
    df['cash_return_assets'] = df['cf_operations'] / df['asst_tot']
    df['leverage'] = (df['debt_lt'] + df['debt_st']) / df['asst_tot']
    df['cash_ratio'] = (df['cash_and_equiv'] + df['cf_operations']) / (df['debt_st'] + df['AP_st'] + 1)        
    
    return df

# Impute post feature engineering (impute ratios)
def nan_var_replace_post_feat(df):
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['debt_ratio'].fillna(df['asst_tot'] - df['eqty_tot'], inplace=True)

    return df

# Final imputation using medians for train data
def nan_var_median_post_feat_train(df):

    preproc_params = {}

    for column in df.columns:
        if df[column].dtype == 'float64':
            median = df[column].median()
            preproc_params[column + '_median'] = median
            df[column].fillna(median, inplace=True)        

    return df, preproc_params

# Final imputation using medians for test data
def nan_var_median_post_feat_test(df, preproc_params={}):

    print(preproc_params)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column].fillna(preproc_params[column + '_median'], inplace=True)

    return df

def preprocessor(df, preproc_params, train):

    data = df.copy()

    # Imputes the variables before feature engineering (avoid inf or -inf)
    df_nan_impute = nan_var_replace_pre_feat(data)

    # Do feature engineering
    df_feat_eng = feature_engineering(df_nan_impute)

    # Impute post feature engineering (impute ratios)
    df_feat_eng_nan_impute = nan_var_replace_post_feat(df_feat_eng)

    # Required features
    final_columns = [
                 'roa',
                 'cash_ratio',
                 'asst_tot',
                 'debt_ratio',
                 'cash_return_assets',
                 'days_rec',
                 'leverage',
                 'stmt_date',
                 'default_label']

    # Drop unnecessary columns
    df_features = df_feat_eng_nan_impute.drop(columns = [col for col in df_feat_eng 
                                            if col not in final_columns])
    
    # Final imputation using medians
    if train:        
        result_df, preproc_params = nan_var_median_post_feat_train(df_features)
        
    else:
        result_df = nan_var_median_post_feat_test(df_features, preproc_params)

    return result_df, preproc_params    

def estimator(df, f):
    model = smf.logit(f, data=df).fit(maxiter=200)
    return(model)

def predictor(new_df, model):
    return model.predict(new_df)

def model_performance_metrics(result_df, stmt_date):
    y_test = result_df['Actual_Value']
    predict = result_df['Predicted_Probability']

    roc_auc = roc_auc_score(y_test, predict)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    threshold = 0.5  
    fpr, tpr, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(fpr, tpr)

    count_0s = sum(1 for val in y_test if val == 0)
    count_1s = sum(1 for val in y_test if val == 1)
    
    print("Count of 0s in labels:", count_0s)
    print("Count of 1s in labels:", count_1s)
    
    count_less_than_0_5 = 0
    count_greater_than_0_5 = 0
    
    for val in result_df['Predicted_Probability']:
        if val < threshold:
            count_less_than_0_5 += 1
        elif val > threshold:
            count_greater_than_0_5 += 1
    
    print("Count of values less than 0.5 in labels:", count_less_than_0_5)
    print("Count of values greater than 0.5 in labels:", count_greater_than_0_5)

    stmt_date = str(stmt_date)
    stmt_date = stmt_date[:10]
    
    stats_dict = {
        stmt_date + '_auc': round(roc_auc, 4)
    }

    return stats_dict

def predictor_harness(new_df, model, preprocessor, preproc_params = {}):

    df_test, preproc_params = preprocessor(new_df, preproc_params, train=False)
    
    X_test = df_test.drop(columns = ['default_label', 'stmt_date'], axis=1)
    y_test = df_test['default_label']
    
    predict = predictor(X_test, model)
    predictions = pd.DataFrame({
        'Actual_Value': y_test,
        'Predicted_Probability': predict
    })    
    
    return(predictions)

def walk_forward_harness(df, preprocessor, estimator, predictor):

    predictions = []
    model_list = []
    stats_list = []

    dates = df['stmt_date'].unique()
    dates = pd.Series(dates).sort_values(ascending=True)
    df.dropna()
    
    for i in range(0, len(dates), 1):

        stmt_date = dates[i]
        print("Statement date (test year):", stmt_date)

        # pre-processing
        preproc_params = {}

        train_data = df[df['stmt_date'] < dates[i]]
        if len(train_data) == 0:
            continue
        test_data = df[df['stmt_date'] == dates[i]]

        df_train, preproc_params = preprocessor(train_data, preproc_params, train=True)
        X_train = df_train.drop(columns = ['default_label', 'stmt_date'], axis=1)

        # estimator
        column_names = [item for item in list(X_train.columns)]
        my_formula = "default_label ~ " + " + ".join(column_names)
        model = estimator(df_train, my_formula)
        model_list.append(model)

        # predictor
        result_df = predictor_harness(test_data, model, preprocessor, preproc_params)
        predictions.append(result_df)

        # performance metrics
        stats = model_performance_metrics(result_df, stmt_date)
        stats_list.append(stats)              

    return (predictions, model_list, stats_list)

# Fit a nonlinear curve
def curve_func(x, a, b, c):
    return a * x**2 + b * x + c

def get_curve_params(df, model, k):
    
    # Add model predictions to the dataframe
    df['prediction'] = model.predict(df)

    # Sort the dataframe based on model predictions
    df_sorted = df.sort_values(by='prediction', ascending=False)

    # Create k buckets
    N = len(df)
    bucket_size = N // k
    buckets = [df_sorted.iloc[i*bucket_size:(i+1)*bucket_size] for i in range(k)]

    # Calculate default rates
    default_rates = []
    quantiles = []
    for i, bucket in enumerate(buckets):
        default_rate = bucket['default_label'].mean()
        default_rates.append(default_rate)
        quantiles.append(bucket['prediction'].min())

    params, _ = curve_fit(curve_func, quantiles, default_rates)

    return params

def calibrate_prob(raw_prob, params):
    calibrate_prob = [curve_func(x, *params) for x in raw_prob]
    return calibrate_prob

# read train data
df_original = pd.read_csv('train.csv')
df = df_original.copy()
df = set_target_label(df)

# conduct walk-forward analysis for the model
predictions, model_list, stats_list = walk_forward_harness(df, preprocessor, estimator, predictor)

# get auc for the concatenated predictions
predictions_concat = pd.concat(predictions, ignore_index=True)
stats = model_performance_metrics(predictions_concat, '')
print(stats)

# train the final model on the entire train.csv
df = df_original.copy()
df = set_target_label(df)

preproc_params = {}
df_train, preproc_params = preprocessor(df, preproc_params, train=True)
print(preproc_params)
# {'asst_tot_median': 3458428.5, 'days_rec_median': 133.75859807872905, 'roa_median': 2.39, 
# 'debt_ratio_median': 0.5506120856551856, 'cash_return_assets_median': 0.021743523917807147, 
# 'leverage_median': 0.7460792711253206, 'cash_ratio_median': 0.0795322078590196}

X_train = df_train.drop(columns = ['default_label', 'stmt_date'], axis=1)
y_train = df_train['default_label']

column_names = [item for item in list(X_train.columns)]
my_formula = "default_label ~ " + " + ".join(column_names)
model = estimator(df_train, my_formula)

# model calibration
params = get_curve_params(df_train, model, 20)
print(params)
#  [ 5.86333743e+02 -7.31859846e+00  1.75116159e-02]

# save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
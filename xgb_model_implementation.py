#%% Import libraries
import numpy as np
import pandas as pd
from numpy import mean
from scipy.stats import sem
from sklearn.model_selection import cross_val_score,train_test_split,cross_validate, StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb # Learning API, direct library
from xgboost.sklearn import XGBClassifier # sklearn API
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,roc_auc_score,roc_curve,confusion_matrix,classification_report, auc, f1_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, Normalizer
from imblearn.over_sampling import SMOTE,SMOTENC,BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours,RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.inspection import permutation_importance
from collections import Counter
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import resample
from scipy import interp
from pylab import rcParams
import time, pickle, os
from pathlib import Path
from joblib import dump,load
from scikitplot.metrics import plot_roc_curve
from matplotlib import pyplot as plt
# %matplotlib inline

### Import Data ###
data_path = r"asthma_attack_risk_prediction\data"
train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Over6YearsOfAge_Quarter5.csv"))
test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Over6YearsOfAge_Quarter5.csv"))

less_important_features = ['NebulisedSABA','IschaemicHeartDisease', 'Obesity', 'NasalPolyps', 'DementiaAlzheimers', 
                           'RheumatologicalDisease', 'PulmonaryEosinophilia', 'Anaphylaxis', 'Psoriasis', 'AtopicDermatitis']
train_data_rfe = train_data.drop(less_important_features, axis=1)
test_data_rfe = test_data.drop(less_important_features, axis=1)


train_data.AsthmaAttack_Q5.value_counts()
test_data.AsthmaAttack_Q5.value_counts()

### --------- Data Cleaning ###
# Handling missing values
train_data.isna().sum() # SABA_ICS_Ratio 48444
test_data.isna().sum() # SABA_ICS_Ratio 20720

train_data["SABA_ICS_Ratio"] = train_data["SABA_ICS_Ratio"].replace(np.nan,0)
test_data["SABA_ICS_Ratio"] = test_data["SABA_ICS_Ratio"].replace(np.nan,0)

train_data.dtypes

# Converting target variable to categorical
# train_data["AsthmaAttack_Q5"] = train_data["AsthmaAttack_Q5"].astype("category")

# Convert to categorical datatype
# categorical_cols = ["CohortYear","Q5_WeeksDuringWinter", "Age_0609", "Age_1019", "Age_2029", "Age_3039",
#                     "Age_4049", "Age_5059", "Age_6069", "Age_7079", "Age_8099", "Age_100Plus", "Gender", 
#                     'Eth_E', 'Eth_M', 'Eth_P', 'Eth_A', 'Eth_O', 'DeprivationQuintile', 'DHB', 
#                     'CardiovascularCerebrovascularDisease', 'IschaemicHeartDisease','NasalPolyps', 
#                     'Anaphylaxis', 'PulmonaryEosinophilia', 'AtopicDermatitis', 'Psoriasis', 'Obesity', 'LRTI', 
#                     'RheumatologicalDisease', 'AnxietyDepression', 'DementiaAlzheimers', 'HeartFailure', 
#                     'Diabetes', 'Rhinitis', 'SmokingStatus', 'Metformin', 'Diabetes_NoMetformin', 
#                     'BetaBlockers', 'Paracetamol', 'NSAIDs', 'NebulisedSABA', 'CharlsonComorbidityScore_12Max', 
#                     'AsthmaSeverityStep']

# categorical_cols = ["CohortYear","Q5_WeeksDuringWinter", 'DeprivationQuintile', 'DHB', 
#                     'CharlsonComorbidityScore_12Max', 'AsthmaSeverityStep']

# categorical_cols = ["CohortYear","DeprivationQuintile","DHB"]
# train_data[categorical_cols] = train_data[categorical_cols].astype("category")
# test_data[categorical_cols] = test_data[categorical_cols].astype("category")

categorical_features = ['DHB', 'CohortYear'] # categorical features which require encoding

numeric_features = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']

categorical_features_rfe = ['DHB','CohortYear']

numeric_features_rfe = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']


# Comparing previous method
# categorical_features = ['DHB', 'CohortYear', 'DeprivationQuintile']

ST_scalar = StandardScaler()
# Normaliser = Normalizer(norm="max", copy=False)
# Normaliser = Normalizer(copy=False)
# OH_Encoder = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False, min_frequency=0.001)
OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# data_test_normalisation = train_data.copy()

# transformer = Normalizer().fit(X=data_test_normalisation[numeric_features])
# transformer.transform(data_test_normalisation[numeric_features])

preprocessor = ColumnTransformer(
    transformers=[
        # ('t_encoder',TE_encoder,TE_features),
        ('categorical',OH_Encoder,categorical_features),
        ('numeric',ST_scalar,numeric_features)
        ],
    remainder='passthrough'
    )

#RFE preprocesor
preprocessor_rfe = ColumnTransformer(
    transformers=[
        ('categorical',OH_Encoder,categorical_features_rfe),
        ('numeric',ST_scalar,numeric_features_rfe)
        ],
    remainder='passthrough'
    )


print(Counter(train_data["AsthmaAttack_Q5"]))
#%%  Splitting into Dependent and Independent variables

y_train = train_data["AsthmaAttack_Q5"]
X_train = train_data.drop("AsthmaAttack_Q5", axis=1)
y_test = test_data["AsthmaAttack_Q5"]
X_test = test_data.drop("AsthmaAttack_Q5", axis=1)

X_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

print(Counter(y_train))
print(Counter(y_test))

# RFE split
y_train_rfe = train_data_rfe["AsthmaAttack_Q5"]
X_train_rfe = train_data_rfe.drop("AsthmaAttack_Q5", axis=1)
y_test_rfe = test_data_rfe["AsthmaAttack_Q5"]
X_test_rfe = test_data_rfe.drop("AsthmaAttack_Q5", axis=1)

X_train_rfe.reset_index(drop=True,inplace=True)
y_train_rfe.reset_index(drop=True,inplace=True)
X_test_rfe.reset_index(drop=True,inplace=True)
y_test_rfe.reset_index(drop=True,inplace=True)
#%% Function to generate confusion matrix
def plot_conf_matrix(actual_classes, predicted_classes):
    matrix = confusion_matrix(actual_classes, predicted_classes)
    plt.figure(figsize=(5,5))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.xlabel("Predicted values")
    plt.ylabel("Actual Values")
    plt.title("Confusion Matrix")
    plt.show()
#%% Base classifier
    
model_base = XGBClassifier(random_state=93186, n_jobs=-1, verbosity=1)
# rs = RandomUnderSampler(random_state=93196)
rs = SMOTE(random_state=93196)
# Best Model - SMOTE
# model_best = XGBClassifier(random_state = 93186,
#                             n_jobs = -1,
#                             verbosity = 0,
#                             seed = 456,
#                             learning_rate = 0.01,
#                             n_estimators = 100, 
#                             gamma = 0, # talen from 100000 tuned value
#                             subsample = 0.9,
#                             colsample_bytree = 0.6,
#                             objective = 'binary:logistic',
#                             scale_pos_weight = 1,
#                             eval_metric = "auc",
#                             max_depth = 7,
#                             min_child_weight = 2,
#                             use_label_encoder=False)

# Best Model - Downsample
# model_best = XGBClassifier(random_state = 93186,
#                             n_jobs = -1,
#                             verbosity = 0,
#                             seed = 456,
#                             learning_rate = 0.01,
#                             n_estimators = 1000, 
#                             gamma = 0, # talen from 100000 tuned value
#                             subsample = 0.8,
#                             colsample_bytree = 0.8,
#                             objective = 'binary:logistic',
#                             scale_pos_weight = 1,
#                             eval_metric = "auc",
#                             max_depth = 6,
#                             min_child_weight = 6,
#                             use_label_encoder=False)

pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",rs),
                        ("classifier",model_base),
                ],
            )
# ******** Important ********
#df = pd.DataFrame(pipe.named_steps["preprocessor"].transform(X_train))

# X_transformed   = preprocessor.fit_transform(X_train,y_train)
# trans_data = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
## You can get the feature names preprocesor. But with Target Encoder, it does not work.
## However, by removing TE, we can get the names and columns are in the same order
## except DomicileCode. When TE is there, it comes to the first column.

# To test intermediate steps (sampling)
# fit_pipe = pipe.fit(X_train,y_train)
# df = fit_pipe.fit_resample(X_train,y_train)
# ********
#%% Stratified KFold Cross Validation
rcParams['figure.figsize'] = 10,10
plt.rcParams.update({'font.size': 16})
plt.figure(dpi=300)
tic = time.perf_counter()
kf = StratifiedKFold(n_splits=10)
np.random.seed(42) #234
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
no_classes = len(np.unique(y_train))
actual_classes = np.empty([0], dtype=int)
predicted_classes = np.empty([0], dtype=int)
predicted_proba = np.empty([0, no_classes])
mse =[]

for fold,(train_index, test_index) in enumerate(kf.split(X_train,y_train),1):
    print(f'Fold {fold}')
    X_train_fold = X_train.loc[train_index]
    y_train_fold = y_train.loc[train_index]  
    X_test_fold = X_train.loc[test_index]
    y_test_fold = y_train.loc[test_index]  
    pipe.fit(X_train_fold, y_train_fold)  
    y_pred = pipe.predict(X_test_fold)
    prediction = pipe.predict_proba(X_test_fold)
    fpr, tpr, t = roc_curve(y_test_fold, prediction[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (fold, roc_auc))
    actual_classes = np.append(actual_classes, y_test_fold)
    predicted_classes = np.append(predicted_classes, y_pred)
    
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

mse = sem(aucs)
print(f'Standard Error of Mean AUC: {mse}')

# Confusion Matrix
plt.rcParams.update({'font.size': 16})
plot_conf_matrix(actual_classes, predicted_classes)
cls_report_train = classification_report(actual_classes,predicted_classes)

#%% Evaluate the model
def plot_conf_matrix(y_test,y_preds):
    conf_mat = confusion_matrix(y_test,y_preds)
    sns.heatmap(conf_mat, annot=True,cmap="Blues",fmt="g")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual Values")
    plt.title("Confusion Matrix")

def to_labels (pos_probs,threshold):
    return(pos_probs>=threshold).astype('int') #return 0 or 1

def evaluateFinalModel(y_test,y_pred_proba_class1,y_preds):
    #define metrics
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_class1)
    auc_1 = roc_auc_score(y_test, y_pred_proba_class1)
    auc_2 = auc(fpr, tpr)
    print(f"AUC {auc_2:0.2f}")

    #best threshold in ROC
    # gmeans = np.sqrt(tpr*(1-fpr))
    # ix = np.argmax(gmeans)

    rcParams['figure.figsize'] = 5,5
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.dpi':300})

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(round(auc_1,2)))
    # plt.scatter(fpr[ix],tpr[ix],label='Optimal threshold {%0.5f}'%t[ix], marker='o', color='black')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

    #Confusion matrix
    plot_conf_matrix(y_test, y_preds)

    #Classification Report
    cls_report_final = classification_report(y_test,y_preds)
    print(cls_report_final)
    
def tune_threshold(y_test,y_pred_proba_class1,y_preds):
    #Tuning threshold
    threshold_range = np.arange(0,1,0.001)
    scores = [f1_score(y_test, to_labels(y_pred_proba_class1, threshold)) for threshold in threshold_range]
    ix_t = np.argmax(scores)
    best_t = threshold_range[ix_t]
    best_f1_score = scores[ix_t]
    return best_t,best_f1_score

#%% Train and Test Base model
tic = time.perf_counter()
model_base = XGBClassifier(random_state=93186,
                                    n_jobs=-1,
                                    verbosity=1)
                                   # n_estimators=100, #1000
                                    #use_label_encoder=False) #for future changes

rs_down = RandomUnderSampler(random_state=93196)
rs_smote = SMOTE(random_state=93196)
pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",rs_smote),
                        ("classifier",model_base)
                ]
            )

# RFE Execution
pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_down),
                        ("classifier",model_base)
                ]
            )
                              
# model_fitted = XGBClassifier(random_state=93186,
#                         n_jobs=-1,
#                         verbosity=1,
#                         seed=456,
#                         learning_rate=0.01,
#                         n_estimators=500,
#                         max_depth=5,
#                         min_child_weight=5,
#                         gamma=0,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         objective= 'binary:logistic',
#                         scale_pos_weight=1,
#                         eval_metric='error')

pipe_base.fit(X_train,y_train)
y_pred_base = pipe_base.predict(X_test)
y_pred_proba_base = pipe_base.predict_proba(X_test)
y_pred_proba_class1 = y_pred_proba_base[::,1]
evaluateFinalModel(y_test,y_pred_proba_class1,y_pred_base)
# toc = time.perf_counter() 
# print(f"Time elapsed {toc-tic:0.4f} seconds") 

# RFE Execution
pipe_base.fit(X_train_rfe,y_train_rfe)
y_pred_base = pipe_base.predict(X_test_rfe)
y_pred_proba_base = pipe_base.predict_proba(X_test_rfe)
y_pred_proba_class1 = y_pred_proba_base[::,1]
toc = time.perf_counter() 

### Evaluate final model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1,y_pred_base)
#%% Save the Model ###
# change directory
cd_path = r"XGB\RFE\NoSample"
os.chdir(cd_path)
# print("Current working directory: {0}".format(os.getcwd()))
# pickle.dump(model_base, open("XGB_smote_100000_model_base.pkl", "wb"))
pickle.dump(model_base, open("XGB_noSample_model_base.pickle", "wb"))

# *************************************************************************************** #
#%% ------------------ ** HYPERPARAMETER TUNING ** ---------------------------
## Fix learning rate and number of estimators for tuning tree-based parameters

# def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     # This function is to find the best n_estimators for xgboost model
#     y_train=y_train.astype("int64")
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(X_train, y_train)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#             metrics='auc', early_stopping_rounds=early_stopping_rounds)
#         alg.set_params(n_estimators=cvresult.shape[0]) #set the best n_estimators
        
#     #Print model report:
#     print("\nModel Report")
#     print(f"n_estimators: {cvresult}")
#     print(f"set n_estimators to: {cvresult.shape[0]}")
                    
#     # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     # feat_imp.plot(kind='bar', title='Feature Importances')
#     # plt.ylabel('Feature Importance Score')

# # predictors = [x for x in X_train.columns]
# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  scale_pos_weight=1,
#  seed=27,
#  eval_metric ="auc")
# modelfit(xgb1, X_train,y_train)
# # Results = n_estimators=59
# ----------------------------------------------------------------------------------

#%% Downsample Hyperparameter Tuning 
# Initially keep n_estimators=100, and others as the base estimator, then this will be increased later

##  Tune max_depth and min_child_weight with the above found n_estimaros and other fixed params
param_grid1 = {
 'classifier__max_depth':range(3,10,2), #[3,5,7,9]
 'classifier__min_child_weight':range(1,6,2) #[1,3,5]
}

xgb_tune1 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc")

rs_tune1 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune1 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune1),
                        ("classifier",xgb_tune1),
                ],
            )
gs_grid1 = GridSearchCV(pipe_tune1, param_grid1, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1.fit(X_train_rfe, y_train_rfe)
gs_grid1_cvResults = gs_grid1.cv_results_ 
print(f"Best params: {gs_grid1.best_params_}\nBest score: {gs_grid1.best_score_}")
# Results - Downsample
# Best params: {'classifier__max_depth': 3, 'classifier__min_child_weight': 3}
# Best score: 0.7514124063369012

# further investiagting max_depth and min_child_weight
param_grid1_1 = {
 'classifier__max_depth':[2,3,4],
  'classifier__min_child_weight':[1,2,3]
}

xgb_tune1_1 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc")

# rs_tune1_1 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune1_1 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune1),
                        ("classifier",xgb_tune1_1),
                ],
            )
gs_grid1_1 = GridSearchCV(pipe_tune1_1, param_grid1_1, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1_1.fit(X_train_rfe, y_train_rfe)
gs_grid1_1_cvResults = gs_grid1_1.cv_results_ 
print(f"Best params: {gs_grid1_1.best_params_}\nBest score: {gs_grid1_1.best_score_}")
# Results - Downsample
# Best params: {'classifier__max_depth': 4, 'classifier__min_child_weight': 4}
# Best score: 0.7514504920169934

# # Further tuning for min_child_weight
# param_grid1_2 = {
#  'classifier__min_child_weight':[3,4,5]
# }

# xgb_tune1_2 = XGBClassifier(learning_rate = 0.1,
#                          n_estimators=100,
#                          gamma=0,
#                          subsample=0.8,
#                          colsample_bytree=0.8,
#                          objective= 'binary:logistic',
#                          scale_pos_weight=1,
#                          seed=27,
#                          eval_metric ="auc",
#                          max_depth=4)

# rs_tune1_2 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

# pipe_tune1_2 = Pipeline(steps=[ ("preprocessor",preprocessor),
#                         ("sampler",rs_tune1),
#                         ("classifier",xgb_tune1_2),
#                 ],
#             )
# gs_grid1_2 = GridSearchCV(pipe_tune1_2, param_grid1_2, scoring="roc_auc", n_jobs=-1, cv=5)
# gs_grid1_2.fit(X_train, y_train)
# # gs_grid1_2_cvResults = gs_grid1_1.cv_results_ 
# print(f"Best params: {gs_grid1_2.best_params_}\nBest score: {gs_grid1_2.best_score_}")


# Tune subsample and colsample_bytree
param_grid2 = {
 'classifier__subsample':[i/10.0 for i in range(6,10)],
 'classifier__colsample_bytree':[i/10.0 for i in range(6,10)]
}

xgb_tune2 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=4,
                         min_child_weight=1)

# rs_tune2 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune2 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune1),
                        ("classifier",xgb_tune2),
                ],
            )
gs_grid2 = GridSearchCV(pipe_tune2, param_grid2, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid2.fit(X_train_rfe, y_train_rfe)
# gs_grid2_cvResults = gs_grid2.cv_results_ 
print(f"Best params: {gs_grid2.best_params_}\nBest score: {gs_grid2.best_score_}")
#Results
# Best params: {'classifier__colsample_bytree': 0.6, 'classifier__subsample': 0.7}
# Best score: 0.751739248346006



# # Tune regularization parameter to avoid overfitting
# param_grid3 = {
#  'classifier__reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1]
# }

#lower the learning rate and add more trees
param_grid3 = { 'classifier__learning_rate': [0.01, 0.05, 0.1],
               'classifier__n_estimators': [100, 200, 500, 1000, 1500, 2000]}

# print(random_grid)
xgb_tune3 = XGBClassifier(gamma=0,
                         subsample=0.7,
                         colsample_bytree=0.6,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=4,
                         min_child_weight=1)

# rs_tune3 = RandomUnderSampler(random_state=93196)
pipe_tune3 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune1),
                        ("classifier",xgb_tune3),
                ],
            )
gs_grid3 = GridSearchCV(estimator = pipe_tune3, 
                         param_grid = param_grid3, 
                         cv = 5, 
                         scoring="roc_auc", 
                         verbose=2, 
                         n_jobs = -1,
                         error_score='raise',
                         return_train_score=True)

gs_grid3.fit(X_train_rfe,y_train_rfe)
#Best model parameters
# gs_grid3_cvResults = gs_grid3.cv_results_ 
print(f"Best params: {gs_grid3.best_params_}\nBest score: {gs_grid3.best_score_}")
#Results
# Best params: {'classifier__learning_rate': 0.01, 'classifier__n_estimators': 1000}
# Best score: 0.7523669263298759


#%% NoSample Hyperparameter Tuning 

# Initially keep n_estimators=100, and others as the base estimator, then this will be increased later

##  Tune max_depth and min_child_weight with the above found n_estimaros and other fixed params
param_grid1 = {
 'classifier__max_depth':range(3,10,2), #[3,5,7,9]
 'classifier__min_child_weight':range(1,6,2) #[1,3,5]
}

xgb_tune1 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc")

# rs_tune1 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune1 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_tune1),
                        ("classifier",xgb_tune1),
                ],
            )
gs_grid1 = GridSearchCV(pipe_tune1, param_grid1, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1.fit(X_train_rfe, y_train_rfe)
gs_grid1_cvResults = gs_grid1.cv_results_ 
print(f"Best params: {gs_grid1.best_params_}\nBest score: {gs_grid1.best_score_}")
#For NoSample
# Best params: {'classifier__max_depth': 7, 'classifier__min_child_weight': 3}
# Best score: 0.7653984634962744

# further investiagting max_depth and min_child_weight
param_grid1_1 = {
 'classifier__max_depth':[4,5,6],
  'classifier__min_child_weight':[2,3,4]
}

xgb_tune1_1 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc")

# rs_tune1_1 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune1_1 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_tune1_1),
                        ("classifier",xgb_tune1_1),
                ],
            )
gs_grid1_1 = GridSearchCV(pipe_tune1_1, param_grid1_1, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1_1.fit(X_train_rfe, y_train_rfe)
gs_grid1_1_cvResults = gs_grid1_1.cv_results_ 
print(f"Best params: {gs_grid1_1.best_params_}\nBest score: {gs_grid1_1.best_score_}")
# Best params: {'classifier__max_depth': 7, 'classifier__min_child_weight': 2}
# Best score: 0.7655200715557374
# Further tuning for min_child_weight
param_grid1_2 = {
 'classifier__min_child_weight':[3,4,5]
}

xgb_tune1_2 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=5)

# rs_tune1_2 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune1_2 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_tune1_2),
                        ("classifier",xgb_tune1_2),
                ],
            )
gs_grid1_2 = GridSearchCV(pipe_tune1_2, param_grid1_2, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1_2.fit(X_train_rfe, y_train_rfe)
# gs_grid1_2_cvResults = gs_grid1_1.cv_results_ 
print(f"Best params: {gs_grid1_2.best_params_}\nBest score: {gs_grid1_2.best_score_}")


# Tune subsample and colsample_bytree
param_grid2 = {
 'classifier__subsample':[i/10.0 for i in range(6,10)],
 'classifier__colsample_bytree':[i/10.0 for i in range(6,10)]
}

xgb_tune2 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=5,
                         min_child_weight=4)

# rs_tune2 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune2 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_tune2),
                        ("classifier",xgb_tune2),
                ],
            )
gs_grid2 = GridSearchCV(pipe_tune2, param_grid2, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid2.fit(X_train_rfe, y_train_rfe)
# gs_grid2_cvResults = gs_grid2.cv_results_ 
print(f"Best params: {gs_grid2.best_params_}\nBest score: {gs_grid2.best_score_}")

# # Tune regularization parameter to avoid overfitting
# param_grid3 = {
#  'classifier__reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1]
# }

#lower the learning rate and add more trees
param_grid3 = { 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
               'classifier__n_estimators': [100, 200, 500, 1000, 1500]}

# print(random_grid)
xgb_tune3 = XGBClassifier(gamma=0,
                         subsample=0.7,
                         colsample_bytree=0.6,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=5,
                         min_child_weight=4)

# rs_tune3 = RandomUnderSampler(random_state=93196)
# rs = SMOTE(random_state=93196)

pipe_tune3 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_tune3),
                        ("classifier",xgb_tune3),
                ],
            )

gs_grid3 = GridSearchCV(estimator = pipe_tune3, 
                         param_grid = param_grid3, 
                         cv = 5, 
                         scoring="roc_auc", 
                         verbose=2, 
                         n_jobs = -1,
                         error_score='raise',
                         return_train_score=True)

gs_grid3.fit(X_train_rfe,y_train_rfe)
#Best model parameters
# gs_grid3_cvResults = gs_grid3.cv_results_ 
print(f"Best params: {gs_grid3.best_params_}\nBest score: {gs_grid3.best_score_}")
# Best params: {'classifier__learning_rate': 0.01, 'classifier__n_estimators': 1500}
# Best score: 0.7660370026890237
#%% SMOTE Hyperparamter Tuning
# Initially keep n_estimators=100, and others as the base estimator, then this will be increased later

##  Tune max_depth and min_child_weight with the above found n_estimaros and other fixed params
param_grid1 = {
 'classifier__max_depth':range(3,10,2), #[3,5,7,9]
 'classifier__min_child_weight':range(1,6,2) #[1,3,5]
}

xgb_tune1 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc")

rs_tune1 = SMOTE(random_state=93196)

pipe_tune1 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune1),
                        ("classifier",xgb_tune1),
                ],
            )
gs_grid1 = GridSearchCV(pipe_tune1, param_grid1, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1.fit(X_train_rfe, y_train_rfe)
gs_grid1_cvResults = gs_grid1.cv_results_ 
print(f"Best params: {gs_grid1.best_params_}\nBest score: {gs_grid1.best_score_}")
# Best params: {'classifier__max_depth': 9, 'classifier__min_child_weight': 1}
# Best score: 0.7302763507888457

# further investiagting max_depth and min_child_weight
param_grid1_1 = {
 'classifier__max_depth':[8,9,10],
 'classifier__min_child_weight':[2,3,4]
}

xgb_tune1_1 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc")

rs_tune1_1 = SMOTE(random_state=93196)

pipe_tune1_1 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune1_1),
                        ("classifier",xgb_tune1_1),
                ],
            )
gs_grid1_1 = GridSearchCV(pipe_tune1_1, param_grid1_1, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid1_1.fit(X_train_rfe, y_train_rfe)
gs_grid1_1_cvResults = gs_grid1_1.cv_results_ 
print(f"Best params: {gs_grid1_1.best_params_}\nBest score: {gs_grid1_1.best_score_}")
# Best params: {'classifier__max_depth': 10, 'classifier__min_child_weight': 2}
# Best score: 0.731734110998463

# Further tuning for min_child_weight - not required for SMOTE
# param_grid1_2 = {
#  'classifier__min_child_weight':[4,5,6]
# }

# xgb_tune1_2 = XGBClassifier(learning_rate = 0.1,
#                          n_estimators=100,
#                          gamma=0,
#                          subsample=0.8,
#                          colsample_bytree=0.8,
#                          objective= 'binary:logistic',
#                          scale_pos_weight=1,
#                          seed=27,
#                          eval_metric ="auc",
#                          max_depth=6)

# # rs_tune1_2 = RandomUnderSampler(random_state=93196)
# # rs = SMOTE(random_state=93196)

# pipe_tune1_2 = Pipeline(steps=[ ("preprocessor",preprocessor),
#                         # ("sampler",rs_tune1_2),
#                         ("classifier",xgb_tune1_2),
#                 ],
#             )
# gs_grid1_2 = GridSearchCV(pipe_tune1_2, param_grid1_2, scoring="roc_auc", n_jobs=-1, cv=5)
# gs_grid1_2.fit(X_train, y_train)
# # gs_grid1_2_cvResults = gs_grid1_1.cv_results_ 
# print(f"Best params: {gs_grid1_2.best_params_}\nBest score: {gs_grid1_2.best_score_}")

# Tune subsample and colsample_bytree
param_grid2 = {
 'classifier__subsample':[i/10.0 for i in range(6,10)],
 'classifier__colsample_bytree':[i/10.0 for i in range(6,10)]
}
xgb_tune2 = XGBClassifier(learning_rate = 0.1,
                         n_estimators=100,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=10,
                         min_child_weight=2)

rs_tune2 = SMOTE(random_state=93196)
pipe_tune2 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune2),
                        ("classifier",xgb_tune2),
                ],
            )
gs_grid2 = GridSearchCV(pipe_tune2, param_grid2, scoring="roc_auc", n_jobs=-1, cv=5)
gs_grid2.fit(X_train_rfe, y_train_rfe)
gs_grid2_cvResults = gs_grid2.cv_results_ 
print(f"Best params: {gs_grid2.best_params_}\nBest score: {gs_grid2.best_score_}")
# Best params: {'classifier__colsample_bytree': 0.9, 'classifier__subsample': 0.7}
# Best score: 0.7319462069539711

# # Tune regularization parameter to avoid overfitting
# param_grid3 = {
#  'classifier__reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1]
# }

#lower the learning rate
param_grid3 = {'classifier__learning_rate': [0.01, 0.05, 0.1]}
xgb_tune3 = XGBClassifier(gamma=0, 
                          subsample=0.9,
                         n_estimators =100, 
                         colsample_bytree=0.6,
                         objective= 'binary:logistic',
                         scale_pos_weight=1, seed=27,
                         eval_metric ="auc",
                         max_depth=10, 
                         min_child_weight=2)
rs_tune3 = SMOTE(random_state=93196)
pipe_tune3 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune3),
                        ("classifier",xgb_tune3),
                ],
            )
gs_grid3 = GridSearchCV(estimator = pipe_tune3, 
                         param_grid = param_grid3, 
                         cv = 5, 
                         scoring="roc_auc", 
                         verbose=2, 
                         n_jobs = -1,
                         error_score='raise',
                         return_train_score=True)
gs_grid3.fit(X_train_rfe,y_train_rfe)
print(f"Best params: {gs_grid3.best_params_}\nBest score: {gs_grid3.best_score_}")

# Reducing data for processing speed
#Split data again to make a smaller sample, cz of memory error
# data_sm = data[:1000000]
# X_sm = data_sm.drop("AsthmaAttack",axis=1)
# y_sm = data_sm["AsthmaAttack"]
# X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.25, random_state=123, stratify=y_sm) 
# X_train_sm.reset_index(drop=True,inplace=True)
# y_train_sm.reset_index(drop=True,inplace=True)
# X_test_sm.reset_index(drop=True,inplace=True)
# y_test_sm.reset_index(drop=True,inplace=True)

#Increasing number of trees
param_grid4 = {'classifier__n_estimators': [100, 200, 500, 1000]} #, 2000
xgb_tune4 = XGBClassifier(gamma=0,
                         learning_rate=0.1,
                         subsample=0.9,
                         colsample_bytree=0.6,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27,
                         eval_metric ="auc",
                         max_depth=10,
                         min_child_weight=2)
rs_tune4 = SMOTE(random_state=93196)
pipe_tune4 = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_tune4),
                        ("classifier",xgb_tune4),
                ],
            )
gs_grid4 = GridSearchCV(estimator = pipe_tune4, 
                         param_grid = param_grid4, 
                         cv = 5, 
                         scoring="roc_auc", 
                         verbose=2, 
                         n_jobs = -1,
                         error_score='raise',
                         return_train_score=True)
gs_grid4.fit(X_train_rfe,y_train_rfe)
print(f"Best params: {gs_grid4.best_params_}\nBest score: {gs_grid4.best_score_}")
# Best params: {'classifier__n_estimators': 100}
# Best score: 0.7444451043751944 # results for 1000000 records

# # Tuning gamma for regularization
# param_grid4 = {'classifier__gamma':[0,2,4,6,8,10]}

# xgb_tune4 = XGBClassifier(learning_rate = 0.1,
#                           n_estimators=100,
#                           subsample=0.7,
#                           colsample_bytree=0.9,
#                           objective= 'binary:logistic',
#                           scale_pos_weight=1,
#                           seed=27,
#                           eval_metric ="auc",
#                           max_depth=10,
#                           min_child_weight=2)
# # rs_tune4 = RandomUnderSampler(random_state=93196)
# rs_tune4 = SMOTE(random_state=93196)

# pipe_tune4 = Pipeline(steps=[ ("preprocessor",preprocessor),
#                         ("sampler",rs_tune4),
#                         ("classifier",xgb_tune4),
#                 ],
#             )

# gs_grid4 = GridSearchCV(estimator = pipe_tune4, 
#                           param_grid = param_grid4, 
#                           cv = 5, 
#                           scoring="roc_auc", 
#                           verbose=2, 
#                           n_jobs = -1,
#                           error_score='raise',
#                           return_train_score=True)

# gs_grid4.fit(X_train_sm,y_train_sm)
# #Best model parameters
# gs_grid4_cvResults = gs_grid4.cv_results_ 
# print(f"Best params: {gs_grid4.best_params_}\nBest score: {gs_grid4.best_score_}")

#%% Retrainng the Best Tuned Model
tic = time.perf_counter()
rs_down_best = RandomUnderSampler(random_state=93196)
rs_smote_best = SMOTE(random_state=93196)

# Best model-Downsample
model_bestTuned_down = XGBClassifier(random_state = 93186,
                            n_jobs = -1,
                            verbosity = 0,
                            seed = 456,
                            learning_rate = 0.01,
                            n_estimators = 1000, 
                            gamma = 0, 
                            subsample = 0.7,
                            colsample_bytree = 0.6,
                            objective = 'binary:logistic',
                            scale_pos_weight = 1,
                            eval_metric = "auc",
                            max_depth = 4,
                            min_child_weight = 1)

# # Best model-NoSample
model_best_nosample = XGBClassifier(random_state = 93186,
                            n_jobs = -1,
                            verbosity = 0,
                            seed = 456,
                            learning_rate = 0.01,
                            n_estimators = 1000, 
                            gamma = 0, # talen from 100000 tuned value
                            subsample = 0.7,
                            colsample_bytree = 0.6,
                            objective = 'binary:logistic',
                            scale_pos_weight = 1,
                            eval_metric = "auc",
                            max_depth = 5,
                            min_child_weight = 4,
                            use_label_encoder=False)

# # Best model-SMOTE
model_best_smote = XGBClassifier(random_state = 93186,
                            n_jobs = -1,
                            verbosity = 0,
                            seed = 456,
                            learning_rate = 0.1,
                            n_estimators = 200,
                            gamma = 0, # talen from 100000 tuned value
                            subsample = 0.9,
                            colsample_bytree = 0.6,
                            objective = 'binary:logistic',
                            scale_pos_weight = 1,
                            eval_metric = "auc",
                            max_depth = 10,
                            min_child_weight = 2,
                            use_label_encoder=False)

pipe_tune_best = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                            # ("sampler",rs_down_best),
                            ("classifier",model_best_nosample)
                            ]
                     )

pipe_tune_best.fit(X_train, y_train)  
y_pred_best = pipe_tune_best.predict(X_test)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test)
y_pred_proba_class1_best = y_pred_proba_best[::,1]
evaluateFinalModel(y_test,y_pred_proba_class1_best,y_pred_best)
# toc = time.perf_counter() 
# print(f"Time elapsed {toc-tic:0.4f} seconds") 
best_threshold,best_f1_score = tune_threshold(y_test, y_pred_proba_class1_best, y_pred_best)


# RFE Execution
pipe_tune_best.fit(X_train_rfe, y_train_rfe)  
y_pred_best = pipe_tune_best.predict(X_test_rfe)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test_rfe)
y_pred_proba_class1_best = y_pred_proba_best[::,1]

### Evaluate final Best model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1_best,y_pred_best)
best_threshold,best_f1_score = tune_threshold(y_test_rfe, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with optimal threshold
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test_rfe, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test_rfe,y_pred_class1)
print(cls_report_final_bestThreshold)
#%% Feature Importance
plt.rcParams.update({'font.size': 6})
xgb.plot_importance(model_best)
plt.show()

#Permutation Imprtance
# plt.rcParams.update({'font.size': 11})
# rcParams['figure.figsize'] = 10,10
# perm_importance = permutation_importance(model_bestTuned_down, X_test_rfe, y_test_rfe)
# sorted_idx = perm_importance.importances_mean.argsort()
# first_20 = sorted_idx[78:108]
# plt.barh(X_test.columns[first_20], perm_importance.importances_mean[first_20])
# plt.xlabel("Permutation Importance")


per_imp = permutation_importance(pipe_tune_best, X_test_rfe, y_test_rfe, scoring='roc_auc') #pipe_tune_best
sorted_idx = per_imp.importances_mean.argsort()
first_x = sorted_idx[34:43]

# Re-naming columns
X_train_rfe_copy = X_train_rfe.copy()
X_train_rfe_copy.columns
# X_train_rfe_copy.rename(columns={"P12MNoOfAsthAttacks":"NoOfAsthmaAttacks_Previous12Months"}, inplace=True)
# # X_train_rfe_copy["NoOfAsthmaAttacks_Previous12Months"]
# X_train_rfe_copy.rename(columns={"P6MNoOfAsthAttacks":"NoOfAsthmaAttacks_Previous6Months"}, inplace=True)
# X_train_rfe_copy.rename(columns={"P3MNoOfAsthAttacks":"NoOfAsthmaAttacks_Previous3Months"}, inplace=True)
# X_train_rfe_copy.rename(columns={"DHB":"DistrictHealthBoard"}, inplace=True)
# X_train_rfe_copy.rename(columns={"Eth_E":"EthnicGroup_European"}, inplace=True)

X_train_rfe_copy.rename(columns={"P12MNoOfAsthAttacks":"Number_of_AsthmaAttacks_Previous12Months",
                                 "P6MNoOfAsthAttacks":"Number_of_AsthmaAttacks_Previous6Months",
                                 "P3MNoOfAsthAttacks":"Number_of_AsthmaAttacks_Previous3Months",
                                 "Q5_WeeksDuringWinter":"Number_of_Weeks_of_Winter_Previous3Months",
                                 "NoOfICSInhalers":"Number_of_InhaledCorticosteroid_Inhalers",
                                 "NoOfSABAInhalers":"Number_of_ShortActingBetaAgonist_Inhalers",
                                 "NoOfOPEDVisits":"Number_of_Outpatient_EmergencyDepartment_Visits",
                                 "SABA_ICS_Ratio":"ShortActingBetaAgonist_InhaledCorticosteroid_Ratio",
                                 "CohortYear":"StudyEntry_Year_of_Patient"}, 
                        inplace=True)
color = (0.2, # redness
         0.4, # greenness
         0.2, # blueness
         0.6 # transparency
         ) 

rcParams['figure.figsize'] = 4,5
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'figure.dpi':300})

plt.barh(X_train_rfe_copy.columns[first_x], per_imp.importances_mean[first_x], left=True, color=color)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature Name")
plt.grid(color='grey', linestyle=':', linewidth=1.0, alpha=0.5, axis='x')


# Feature importance based on XGB
# plt.bar(range(len(model_bestTuned_down.feature_importances_)),model_bestTuned_down.feature_importances_)
# plt.show()


#%% Save the Model 
# change directory
cd_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy\XGB\RFE\NoSample"
os.chdir(cd_path)
# print("Current working directory: {0}".format(os.getcwd()))
# pickle.dump(model_best, open("rf_smote_100000_model_base.pkl", "wb"))
pickle.dump(model_best_nosample, open("xgb_rfe_noSample_tunedModel.pickle", "wb"))

#%% Finding threshold & Re-run the model - Down-sample
# Load the model
# load_path = r"XGBoost\XGB_Tuned_Model_fullSet\XGB_Downsample_Num_NoDomIndxMonth_fullset_bestModel"
# load_path = r"XGB\XGB_Downsample\Tuned model"
# load_model = pickle.load(open(os.path.join(load_path,"xgb_Downsample_fullSet_numeric_NoDomicileIndxMonth_best_model.pickle"),'rb'))

# fit the transformer on training data
preprocessor.fit(X_train, y_train)

pipe_load = Pipeline(steps=[ ("preprocessor",preprocessor),
                        # ("sampler",rs_smote_best),
                        ("classifier",model_bestTuned_down)
                ]
            )

# Split into test and validation sets as 10% and 15% (2:3)
X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.4, random_state=42, stratify=y_test)

y_pred_load = pipe_load.predict(X_val)
y_pred_proba_load = pipe_load.predict_proba(X_val)
y_pred_proba_class1 = y_pred_proba_load[::,1] #single colon is also okay

# Evaluate the model on validation set
evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)

# Finding the optimal threshold value
best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

# Confusion matrix with optimal threshold on validation set
y_pred_new = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_val,y_pred_new)

# Evaluate the model on test data
y_pred_load_test = pipe_load.predict(X_test2)
y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

plt.rcParams.update({'font.size': 16})
plot_conf_matrix(y_test2, y_pred_class1_test)

X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="XGB-Down")
#%% Finding threshold & Re-run the model - NoSample
# Load the model
# load_path = r"XGB_Downsample_Num_NoDomIndxMonth_fullset_bestModel"
load_path = r"XGB_Tuned_Model_fullSet\XGB_NoSample_Num_NoDomIndxMonth_fullset_bestModel"
load_model = pickle.load(open(os.path.join(load_path,"xgb_NoSample_fullSet_numeric_NoDomicileIndxMonth_best_model.pickle"),'rb'))

# fit the transformer on training data
preprocessor.fit(X_train, y_train)

pipe_load = Pipeline(steps=[ ("preprocessor",preprocessor),
                        # ("sampler",rs_smote_best),
                        ("classifier",load_model)
                ]
            )

# Split into test and validation sets as 10% and 15% (2:3)
X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.4, random_state=42, stratify=y_test)

y_pred_load = pipe_load.predict(X_val)
y_pred_proba_load = pipe_load.predict_proba(X_val)
y_pred_proba_class1 = y_pred_proba_load[::,1] #single colon is also okay

# Evaluate the model on validation set
evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)

# Finding the optimal threshold value
best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

# Confusion matrix with optimal threshold on validation set
y_pred_new = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_val,y_pred_new)

# Evaluate the model on test data
y_pred_load_test = pipe_load.predict(X_test2)
y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

plt.rcParams.update({'font.size': 16})
plot_conf_matrix(y_test2, y_pred_class1_test)

X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="XGB-NoSample")
#%% Finding threshold & Re-run the model - SMOTE
# Load the model
# load_path = r"XGB_Tuned_Model_fullSet\XGB_Downsample_Num_NoDomIndxMonth_fullset_bestModel"
load_path = r"XGB_Tuned_Model_fullSet\XGB_SMOTE_Num_NoDomIndxMonth_fullset_bestModel"
load_model = pickle.load(open(os.path.join(load_path,"xgb_SMOTE_fullSet_numeric_NoDomicileIndxMonth_best_model.pickle"),'rb'))

# fit the transformer on training data
preprocessor.fit(X_train, y_train)

pipe_load = Pipeline(steps=[ ("preprocessor",preprocessor),
                        # ("sampler",rs_smote_best),
                        ("classifier",load_model)
                ]
            )

# Split into test and validation sets as 10% and 15% (2:3)
X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.4, random_state=42, stratify=y_test)

y_pred_load = pipe_load.predict(X_val)
y_pred_proba_load = pipe_load.predict_proba(X_val)
y_pred_proba_class1 = y_pred_proba_load[::,1] #single colon is also okay

# Evaluate the model on validation set
evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)

# Finding the optimal threshold value
best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

# Confusion matrix with optimal threshold on validation set
y_pred_new = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_val,y_pred_new)

# Evaluate the model on test data
y_pred_load_test = pipe_load.predict(X_test2)
y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_test2, y_pred_class1_test)

X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="XGB-smote")

#%% Saving the XGB_DOWN_RFE (best threshold = 0.72)predictions for the purpose of calculating RR
test_data_rfe_copy = test_data_rfe.copy()
test_data_rfe_copy["predictions"] = y_pred_class1

# cd_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy\XGB\XGB_NoSample"
# data_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy"

test_data_rfe_copy.to_csv("xgb_down_rfe_predictions.csv", index=False)


# RFE Execution
# for i in range(1,101):
pipe_tune_best.fit(X_train_rfe, y_train_rfe)  
y_pred_best = pipe_tune_best.predict(X_test_rfe)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test_rfe)
y_pred_proba_class1_best = y_pred_proba_best[::,1]

### Evaluate final Best model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1_best,y_pred_best)

best_threshold,best_f1_score = tune_threshold(y_test_rfe, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with optimal threshold
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test_rfe, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test_rfe,y_pred_class1)
print(cls_report_final_bestThreshold)


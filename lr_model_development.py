#%% Import Libraries
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.model_selection import cross_val_score,train_test_split,cross_validate, StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,roc_auc_score,roc_curve,confusion_matrix,classification_report, auc, f1_score
from sklearn.preprocessing import OrdinalEncoder,  StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer
from imblearn.over_sampling import SMOTE,SMOTENC,BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours,RandomUnderSampler
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder
# from category_encoders.one_hot import OneHotEncoder
from collections import Counter
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import resample
from scipy import interp
from pylab import rcParams
import time, pickle, os, pprint
from pathlib import Path
from joblib import dump,load
from scikitplot.metrics import plot_roc_curve
# %matplotlib inline

# rcParams['figure.figsize'] = 5,5
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.dpi':300})

#%% ----------------------Data Processing  ------------------------------

### Import Data ###
# data_path = r"asthma_attack_risk_prediction\data"
train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Over12YearsOfAge_Quarter5.csv"))
test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Over12YearsOfAge_Quarter5.csv"))

train_data.AsthmaAttack_Q5.value_counts()
test_data.AsthmaAttack_Q5.value_counts()

# Handling missing values
train_data.isna().sum() # SABA_ICS_Ratio 48444
test_data.isna().sum() # SABA_ICS_Ratio 20720

train_data["SABA_ICS_Ratio"] = train_data["SABA_ICS_Ratio"].replace(np.nan,0)
test_data["SABA_ICS_Ratio"] = test_data["SABA_ICS_Ratio"].replace(np.nan,0)

train_data.dtypes

less_important_features = ['NebulisedSABA','IschaemicHeartDisease', 'Obesity', 'NasalPolyps', 'DementiaAlzheimers',
                           'RheumatologicalDisease', 'PulmonaryEosinophilia', 'Anaphylaxis', 'Psoriasis', 'AtopicDermatitis']
train_data_rfe = train_data.drop(less_important_features, axis=1)
test_data_rfe = test_data.drop(less_important_features, axis=1)

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


# Feature collinearity
plt.imshow(X=train_data)

categorical_features = ['DHB', 'CohortYear']

numeric_features = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']


categorical_features_rfe = ['DHB','CohortYear']

numeric_features_rfe = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']

# OH_Encoder = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False, min_frequency=0.001)
OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

ST_scalar = StandardScaler()
Normaliser = Normalizer(norm="max", copy=False)

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
    plt.figure(figsize=(8,5))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")
    plt.xlabel('Predicted')
    plt.xlabel("Predicted values")
    plt.ylabel("Actual Values")
    plt.title("Confusion Matrix")
    plt.show()

#%% Base Model
# model_base = LogisticRegression(random_state=93186,n_jobs=-1,verbose=1,
#                             solver='saga',penalty='none',
#                             max_iter=1000,tol=1e-4)
# model_base = LogisticRegression(random_state=93186,n_jobs=-1,verbose=1,
#                             solver='newton-cg',penalty='none',
#                             max_iter=1000,tol=1e-4)
model_base = LogisticRegression(random_state=93186, n_jobs=-1,verbose=1 ,max_iter=10000)
# rs = RandomUnderSampler(random_state=93196)
rs = SMOTE(random_state=93196)

pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",rs),
                        ("classifier",model_base),
                ],
            )
# ******** Important ********
#check data at intermediate step
#df = pd.DataFrame(pipe.named_steps["preprocessor"].transform(X_train))

# X_transformed   = preprocessor.fit_transform(X_train,y_train)
# trans_data = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
## You can get the feature names preprocesor. But with Target Encoder, it does not work.
## However, by removing TE, we can get the names and columns are in the same order
## except DomicileCode. When TE is there, it comes to the first column.
# ********
#%% Pricipal Componenet Analysis
# from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(data)
# 95% of variance
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(data_rescaled)
reduced = pca.transform(data_rescaled)
#%% Stratified KFold Cross Validation
rcParams['figure.figsize'] = 10,10 #5,5
plt.rcParams.update({'font.size': 16})
plt.figure(dpi=300)
plt.rcParams.update({'font.size': 16})
tic = time.perf_counter()
kf = StratifiedKFold(n_splits=10)
np.random.seed(234)
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
no_classes = len(np.unique(y_train))
# no_classes = len(np.unique(y_train))
actual_classes = np.empty([0], dtype=int)
predicted_classes = np.empty([0], dtype=int)
std_err =[]
col_names = ['threshold','best_gmean']
df_t_best_gmeans = pd.DataFrame(columns=col_names)
# predicted_proba = np.empty([0, no_classes])

for fold,(train_index, test_index) in enumerate(kf.split(X_train,y_train),1):
    X_train_fold = X_train.loc[train_index]
    y_train_fold = y_train.loc[train_index]
    X_test_fold = X_train.loc[test_index]
    y_test_fold = y_train.loc[test_index]
    print(f'For fold {fold}')
    pipe.fit(X_train_fold, y_train_fold)
    y_pred = pipe.predict(X_test_fold)
    prediction = pipe.predict_proba(X_test_fold)
    pipe.fit(X_train_fold, y_train_fold)
    y_pred = pipe.predict(X_test_fold)
    prediction = pipe.predict_proba(X_test_fold)
    fpr, tpr, t = roc_curve(y_test_fold, prediction[:, 1])
    gmeans = np.sqrt(tpr*(1-fpr))
    ix = np.argmax(gmeans)
    df_t_best_gmeans.at[fold,'threshold'] = t[ix]
    df_t_best_gmeans.at[fold,'best_gmean'] = gmeans[ix]
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
# Confusion Matrix
plt.rcParams.update({'font.size': 16})
plot_conf_matrix(actual_classes, predicted_classes)
cls_report_train = classification_report(actual_classes,predicted_classes)
print(cls_report_train)

toc = time.perf_counter()
print(f"Time elapsed {toc-tic:0.4f} seconds")

print(f'Standard Error of mean AUC: {sem(aucs)}')

# Model coeficients
# log_odds = model.coef_[0]
# model_summary = pd.DataFrame(log_odds, X.columns, columns=['coef'])\
#     .sort_values(by='coef', ascending=False)
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

    rcParams['figure.figsize'] = 5,5
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.dpi':300})

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(round(auc_1,2)))
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
# model_base = LogisticRegression(random_state=93186,n_jobs=-1,verbose=1,
#                             solver='newton-cg',penalty='l2',C=0.01,
#                             max_iter=100000,tol=1e-4)
# model_base = LogisticRegression(random_state=93186,n_jobs=-1,verbose=1,
#                             solver='newton-cg',penalty='none',
#                             max_iter=100,tol=1e-4)
# model_base = LogisticRegression(random_state=93186,n_jobs=-1,max_iter=100, tol=1e-4, solver='newton-cg', penalty='none')
# model_base = LogisticRegression(random_state=93186,n_jobs=-1,max_iter=100,
#                                 tol=1e-4, penalty='none', solver='saga')
model_base = LogisticRegression(random_state=93186, n_jobs=-1,verbose=1, max_iter=10000, solver='newton-cg')
rs_down = RandomUnderSampler(random_state=93196)
# rs_smote = SMOTE(random_state=93196)
pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_down),
                        ("classifier",model_base)
                ]
            )

pipe_base.fit(X_train,y_train)
y_pred_base = pipe_base.predict(X_test)
y_pred_proba_base = pipe_base.predict_proba(X_test)
y_pred_proba_class1 = y_pred_proba_base[::,1]
# toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")
### Evaluate final model
evaluateFinalModel(y_test,y_pred_proba_class1,y_pred_base)


# RFE Execution
pipe_base.fit(X_train_rfe,y_train_rfe)
y_pred_base = pipe_base.predict(X_test_rfe)
y_pred_proba_base = pipe_base.predict_proba(X_test_rfe)
y_pred_proba_class1 = y_pred_proba_base[::,1]
toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")

### Evaluate final model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1,y_pred_base)

#%% Save the Model
# change directory
cd_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy\LR\LR-Downsample\Base model"
os.chdir(cd_path)
# print("Current working directory: {0}".format(os.getcwd()))
pickle.dump(model_base, open("lr_down_base_model.pickle", "wb"))
# plot feature importance
#%% Feature Importance
feature_importance = model_base.coef_
for i,v in enumerate(feature_importance):
    print('Feature:%0d, Score: %.4f' % (i, v))
pyplot.bar([x for x in range(len(importance))], feature_importance)
pyplot.show()
# *************************************************************************************** #
#%% Load Pickle Model
cd_path =r""
os.chdir(cd_path)
pickled_model = pickle.load(open('model.pkl', 'rb'))
#%% Hyperparameter Tuning ###
## Grid SearchCV
max_iter = [100,1000,10000]
C=[0.001,0.01,0.1,1]
penalty=['l1','l2','elasticnet','None']
solver=['newton-cholesky','liblinear',"newton-cg"]

# Create the random search grid
param_grid_gs = {'classifier__max_iter': max_iter,
               'classifier__C': C,
               'classifier__penalty': penalty,
               'classifier__solver': solver}

print(param_grid_gs)

model_tune_gs = LogisticRegression(random_state=93186,n_jobs=-1,verbose=1,tol=1e-4)
# gs_tune = RandomUnderSampler(random_state=93196)
gs_tune = SMOTE(random_state=93196)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                             ("sampler",gs_tune),
                            ("classifier",model_tune_gs),
                            ]
                     )

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
#Best model parameters
grid_search.best_params_
grid_search.best_score_
grid_results = grid_search.cv_results_
#%% Retrainng the Best Model
tic = time.perf_counter()
rs_down = RandomUnderSampler(random_state=93196)
rs_smote = SMOTE(random_state=93196)

# model_best = LogisticRegression(random_state=42,n_jobs=-1,verbose=2,
#                             solver='saga',penalty='l2',
#                             max_iter=100,tol=1e-4,C=0.01)

#NoSample Params
model_best = LogisticRegression(random_state=42,verbose=2, #n_jobs=-1,
                            solver='newton-cholesky',
                            penalty='l2',
                            max_iter=100,
                            tol=1e-4,
                            C=0.001)


#Downsample
model_best = LogisticRegression(random_state=42,verbose=2, #n_jobs=-1,
                            solver='newton-cg',
                            penalty='l2',
                            max_iter=100,
                            tol=1e-4,
                            C=0.01)
#SMOTE
model_best = LogisticRegression(random_state=42,verbose=2, #n_jobs=-1,
                            solver='liblinear',
                            penalty='l1',
                            max_iter=100,
                            tol=1e-4,
                            C=0.001)

pipe_best = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler",rs_smote),
                        ("classifier",model_best)
                ]
            )

pipe_best.fit(X_train, y_train)
y_pred_best = pipe_best.predict(X_test)
y_pred_proba_best = pipe_best.predict_proba(X_test)
y_pred_proba_class1_best = y_pred_proba_best[::,1]
toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")
# Time elapsed 4.8849 seconds
### Evaluate final Best model

plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
evaluateFinalModel(y_test,y_pred_proba_class1_best,y_pred_best)

best_threshold,best_f1_score = tune_threshold(y_test, y_pred_proba_class1_best, y_pred_best)
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test,y_pred_class1)
print(cls_report_final_bestThreshold)


## RFE Execution
pipe_best.fit(X_train_rfe, y_train_rfe)
y_pred_best = pipe_best.predict(X_test_rfe)
y_pred_proba_best = pipe_best.predict_proba(X_test_rfe)
y_pred_proba_class1_best = y_pred_proba_best[::,1]
# toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")

plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
evaluateFinalModel(y_test_rfe,y_pred_proba_class1_best,y_pred_best)

best_threshold,best_f1_score = tune_threshold(y_test_rfe, y_pred_proba_class1_best, y_pred_best)
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test_rfe, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test_rfe,y_pred_class1)
print(cls_report_final_bestThreshold)

#model coefficients
# model_best.coef_
#%% Feature Permutation Importance
plt.rcParams.update({'font.size': 11})
rcParams['figure.figsize'] = 10,10
per_imp = permutation_importance(model_best, X_test, y_test)
sorted_idx = per_imp.importances_mean.argsort()
first_30 = sorted_idx[88:108]
plt.barh(X_train.columns[first_30], per_imp.importances_mean[first_30])
plt.title('Feature Importance')
plt.xlabel("Permutation Importance")
#%% Save the Model
# change directory
cd_path = r"LR\LR-NoSample\Tuned Model"
os.chdir(cd_path)
# print("Current working directory: {0}".format(os.getcwd()))
# pickle.dump(model_best, open("rf_smote_100000_model_base.pkl", "wb"))
pickle.dump(model_best, open("lr_noSample_tunedModel.pickle", "wb"))

#%% Finding threshold & Re-run the model - Downsample
# Load the model
load_path = r"LR\LR-Downsample"
load_model = pickle.load(open(os.path.join(load_path,"lr_down_tunedModel.pickle"),'rb'))

# fit the transformer on training data
preprocessor.fit(X_train, y_train)

pipe_tune_best = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("classifier",load_model)
                ]
            )

# Split into test and validation sets as 10% and 15% (2:3)
# X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.4, random_state=42, stratify=y_test)

# y_pred_load = pipe_load.predict(X_val)
# y_pred_proba_load = pipe_load.predict_proba(X_val)
# y_pred_proba_class1 = y_pred_proba_load[::,1] #single colon is also okay

pipe_tune_best.fit(X_train, y_train)
y_pred_best = pipe_tune_best.predict(X_test)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test)
y_pred_proba_class1_best = y_pred_proba_best[::,1] #single colon is also okay
toc = time.perf_counter()

### Evaluate final Best model
plt.rcParams.update({'font.size': 14})
plt.figure(dpi=600)
plt.rcParams["figure.figsize"] = (5,5)

# evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)

best_threshold,best_f1_score = tune_threshold(y_test, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with OPTIMAL THRESHOLD
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test,y_pred_class1)
print(cls_report_final_bestThreshold)

#AUC 0.77 for combined sampling best model
# best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

# #Confusion matrix with optimal threshold
# y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
# plt.figure(figsize=(5,5))
# plt.rcParams.update({'font.size': 14})
# plot_conf_matrix(y_val, y_pred_class1)

# # Evaluate on test data
# y_pred_load_test = pipe_load.predict(X_test2)
# y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
# y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
# y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

# # plt.rcParams.update({'font.size': 16})
# plot_conf_matrix(y_test2, y_pred_class1_test)

# X_test2_transformed = preprocessor.transform(X_test2)
# plot_roc_curve(load_model, X_test2_transformed, y_test2, name="LR-Downsample")

#%% Finding threshold & Re-run the model - SMOTE
# Load the model
load_path = r"LR\LR-SMOTE"

# fit the transformer on training data
preprocessor.fit(X_train, y_train)

pipe_tune_best = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("classifier",model_best)
                ]
            )

# Split into test and validation sets as 10% and 15% (2:3)
# X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.4, random_state=42, stratify=y_test)

# y_pred_load = pipe_load.predict(X_val)
# y_pred_proba_load = pipe_load.predict_proba(X_val)
# y_pred_proba_class1 = y_pred_proba_load[::,1] #single colon is also okay

pipe_tune_best.fit(X_train, y_train)
y_pred_best = pipe_tune_best.predict(X_test)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test)
y_pred_proba_class1_best = y_pred_proba_best[::,1] #single colon is also okay
toc = time.perf_counter()

### Evaluate final Best model
plt.rcParams.update({'font.size': 14})
plt.figure(dpi=600)
plt.rcParams["figure.figsize"] = (5,5)

# evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)

best_threshold,best_f1_score = tune_threshold(y_test, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with OPTIMAL THRESHOLD
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test,y_pred_class1)
print(cls_report_final_bestThreshold)

#Confusion matrix with optimal threshold
# y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
# plt.figure(figsize=(5,5))
# plt.rcParams.update({'font.size': 14})
# plot_conf_matrix(y_val, y_pred_class1)

# # Evaluate on test data
# y_pred_load_test = pipe_load.predict(X_test2)
# y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
# y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
# y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

# # plt.rcParams.update({'font.size': 16})
# plot_conf_matrix(y_test2, y_pred_class1_test)

# X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="LR-SMOTE")

#%% Finding threshold & Re-run the model - NoSample
# Load the model
load_path = r"Logistic Regression\LR_NoSample_Numeric_NoDomIndxMonth_fullSet_bestModel"
load_model = pickle.load(open(os.path.join(load_path,"lr_NoSample_best.pickle"),'rb'))

# fit the transformer on training data
preprocessor.fit(X_train, y_train)

pipe_tune_best = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("classifier",model_best)
                ]
            )

# Split into test and validation sets as 10% and 15% (2:3)
# X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.4, random_state=42, stratify=y_test)

# y_pred_load = pipe_load.predict(X_val)
# y_pred_proba_load = pipe_load.predict_proba(X_val)
# y_pred_proba_class1 = y_pred_proba_load[::,1] #single colon is also okay

# ### Evaluate final Best model
# plt.rcParams.update({'font.size': 14})
# plt.figure(dpi=600)
# plt.rcParams["figure.figsize"] = (5,5)

# evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)
# #AUC 0.77 for combined sampling best model
# best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

#Confusion matrix with optimal threshold
# y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
# plt.figure(figsize=(5,5))
# plt.rcParams.update({'font.size': 14})
# plot_conf_matrix(y_val, y_pred_class1)

# # Evaluate on test data
# y_pred_load_test = pipe_load.predict(X_test2)
# y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
# y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
# y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

# # plt.rcParams.update({'font.size': 16})
# plot_conf_matrix(y_test2, y_pred_class1_test)

# X_test2_transformed = preprocessor.transform(X_test2)
# plot_roc_curve(load_model, X_test2_transformed, y_test2, name="LR-NoSample")

pipe_tune_best.fit(X_train, y_train)
y_pred_best = pipe_tune_best.predict(X_test)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test)
y_pred_proba_class1_best = y_pred_proba_best[::,1] #single colon is also okay
toc = time.perf_counter()

### Evaluate final Best model
plt.rcParams.update({'font.size': 14})
plt.figure(dpi=600)
plt.rcParams["figure.figsize"] = (5,5)

# evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)

best_threshold,best_f1_score = tune_threshold(y_test_rfe, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with OPTIMAL THRESHOLD
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test_rfe, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test_rfe,y_pred_class1)
print(cls_report_final_bestThreshold)

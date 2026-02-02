#%% Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split,cross_validate, StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,roc_auc_score,roc_curve,confusion_matrix,classification_report, auc, f1_score
from scikitplot.metrics import plot_roc_curve
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, Normalizer, KBinsDiscretizer
from imblearn.over_sampling import SMOTE,SMOTENC,BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours,RandomUnderSampler
# from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder
from collections import Counter
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.utils import resample
from scipy import interp
from pylab import rcParams
import time, pickle, os
import pprint
from pathlib import Path
from joblib import dump,load
from pprint import pprint
# %matplotlib inline
rcParams['figure.figsize'] = 5,5
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.dpi':300})
#%% Data Import and processing
# cd_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_NoSample_Numeric_TE_fullSet"
# data_path = r"C:\Users\Darsha Jayamini\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\data"
# cd_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy\RF\RF-NoSample"
# data_path = r"C:\Users\Darsha Jayamini\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\data"
data_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy"

### -----------Import Data ###
# train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Over6YearsOfAge_Quarter5.csv"))
# test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Over6YearsOfAge_Quarter5.csv"))
train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Over12YearsOfAge_Quarter5.csv"))
test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Over12YearsOfAge_Quarter5.csv"))


# ### Numeric feature data set
# train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Over6YearsOfAge_Quarter5_NumericFeatures.csv"))
# test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Over6YearsOfAge_Quarter5_NumericFeatures.csv"))

# ### Categorical feature data set
# train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Over6YearsOfAge_Quarter5_CategoricalFeatures.csv"))
# test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Over6YearsOfAge_Quarter5_CategoricalFeatures.csv"))


train_data.AsthmaAttack_Q5.value_counts()
test_data.AsthmaAttack_Q5.value_counts()

### ---------- Feature selection  ###
#Extracting data with hign important feature set based on RFE
# train_data_rfe = train_data[['NoOfAsthmaControllerMeds','SABA_ICS_Ratio','P12MNoOfAsthAttacks','P6MNoOfAsthAttacks',
#                              'Q5_WeeksDuringWinter','AsthmaAttack_Q5', 'NoOfSABAInhalers','DeprivationQuintile',
#                              'NoOfOPEDVisits','NoOfICSInhalers','P3MNoOfAsthAttacks','NoOfHospitalisations','Gender',
#                              'Rhinitis','AnxietyDepression','Age_4049','CohortYear','Eth_E','NSAIDs','Age_5059',
#                              'Paracetamol','DHB','Age_3039','Age_6069','AsthmaSeverityStep','Age_2029','Age_1019',
#                              'Eth_M','Age_7079','Eth_A','Age_0609','Eth_P','Age_8099','Eth_O','Age_100Plus','BetaBlockers']]

# test_data_rfe = test_data[['NoOfAsthmaControllerMeds','SABA_ICS_Ratio','P12MNoOfAsthAttacks','P6MNoOfAsthAttacks',
#                            'Q5_WeeksDuringWinter','AsthmaAttack_Q5','NoOfSABAInhalers','DeprivationQuintile',
#                            'NoOfOPEDVisits','NoOfICSInhalers','P3MNoOfAsthAttacks','NoOfHospitalisations','Gender',
#                            'Rhinitis','AnxietyDepression','Age_4049','CohortYear','Eth_E','NSAIDs','Age_5059',
#                            'Paracetamol','DHB','Age_3039','Age_6069','AsthmaSeverityStep','Age_2029','Age_1019',
#                            'Eth_M','Age_7079','Eth_A','Age_0609','Eth_P','Age_8099','Eth_O','Age_100Plus','BetaBlockers']]

less_important_features = ['NebulisedSABA','IschaemicHeartDisease', 'Obesity', 'NasalPolyps', 'DementiaAlzheimers',
                           'RheumatologicalDisease', 'PulmonaryEosinophilia', 'Anaphylaxis', 'Psoriasis', 'AtopicDermatitis']
train_data_rfe = train_data.drop(less_important_features, axis=1)
test_data_rfe = test_data.drop(less_important_features, axis=1)


### --------- Data Cleaning ###
# Handling missing values
train_data.isna().sum() # SABA_ICS_Ratio 48444
test_data.isna().sum() # SABA_ICS_Ratio 20720

#RFE data
train_data_rfe.isna().sum() # SABA_ICS_Ratio 48444
test_data_rfe.isna().sum() # SABA_ICS_Ratio 20720

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

# categorical_features = ['DHB', 'CohortYear', 'Q5_WeeksDuringWinter', 'AsthmaSeverityStep',
#                         'CharlsonComorbidityScore_12Max','DeprivationQuintile']

numeric_features = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']

categorical_features = ['DHB','CohortYear']


categorical_features_rfe = ['DHB','CohortYear']

# numerical_feature_rfe = ['NoOfAsthmaControllerMeds','P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks','SABA_ICS_Ratio',
#                          'NoOfSABAInhalers','NoOfOPEDVisits','NoOfICSInhalers','P3MNoOfAsthAttacks',
#                          'NoOfHospitalisations']

numeric_features_rfe = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']


# all_numeric_features =  ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
#                          'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers',
#                          'SABA_ICS_Ratio','P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']
                        #,'DeprivationQuintile','CharlsonComorbidityScore_12Max','AsthmaSeverityStep','Q5_WeeksDuringWinter']

ST_scalar = StandardScaler()
# Normaliser = Normalizer(norm="max", copy=False)
# OH_Encoder = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False, min_frequency=0.001)
OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

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

#%% Data Preprocessing - For 12 years old + No Bins (Age & Ethnicity) *********
#Set the data path
data_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy"

#Import data
train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins.csv"))
test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins.csv"))

train_data.AsthmaAttack_Q5.value_counts()
test_data.AsthmaAttack_Q5.value_counts()
train_data.dtypes

#Handling Unknown Ethnic group (211 records)
train_data.loc[train_data["EthnicGroup"]=='Unknown',"EthnicGroup"] = 'Other'
test_data.loc[test_data["EthnicGroup"]=='Unknown',"EthnicGroup"] = 'Other'

#Removing leading and trailing spaces in categorical values before encoding - due to issues arose
train_data['DHB'] = train_data['DHB'].str.strip()
train_data['EthnicGroup'] = train_data['EthnicGroup'].str.strip()
test_data['DHB'] = test_data['DHB'].str.strip()
test_data['EthnicGroup'] = test_data['EthnicGroup'].str.strip()

#List categorical, numerical & bin features
categorical_features = ['DHB','EthnicGroup'] #Gender is integer type already
numerical_features = ['NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits',
                    'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NoOfICSInhalers', 'SABA_ICS_Ratio',
                    'P12MNoOfAsthAttacks', 'P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks']
bin_features = ['Age']

#Define the scaler and encoder
ST_scalar = StandardScaler()
OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
K_binner = KBinsDiscretizer(n_bins=9, strategy='uniform', encode='onehot-dense')

#Define the preprocessor
preprocessor_new = ColumnTransformer(
    transformers=[
        ('categorical',OH_Encoder,categorical_features),
        ('numerical',ST_scalar,numerical_features),
        ('binner',K_binner, bin_features),
        ],
    remainder='passthrough'
    )

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
# rcParams['figure.figsize'] = 5,5
# plt.rcParams.update({'font.size': 14})
# plt.figure(dpi=300)

def plot_conf_matrix(actual_classes, predicted_classes):
    matrix = confusion_matrix(actual_classes, predicted_classes)
    plt.figure(figsize=(5,5))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")
    plt.xlabel('Predicted')
    plt.xlabel("Predicted values")
    plt.ylabel("Actual Values")
    plt.title("Confusion Matrix")
    plt.show()
#%% Base Model
model_base = RandomForestClassifier(random_state=93186,
                                n_jobs=-1,
                                verbose=1)
                                # n_estimators=1000)

# Best Model - NoSample
# model_best = RandomForestClassifier(n_jobs=-1,
#                                     verbose=1,
#                                     n_estimators=1200,
#                                     max_features='sqrt',
#                                     min_samples_split=2,
#                                     criterion='gini',
#                                     max_depth=20,
#                                     min_samples_leaf=4)

# Best Model - Downsample
# model_best = RandomForestClassifier(n_jobs=-1,
#                                     verbose=1,
#                                     n_estimators=500,
#                                     max_features='sqrt',
#                                     min_samples_split=2,
#                                     criterion='gini',
#                                     max_depth=10,
#                                     min_samples_leaf=4)

# #Using one sampling technique
rs_down = RandomUnderSampler(random_state=93196)
# rs_smote = SMOTE(random_state=93196)
pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                        # ("sampler",rs_down),
                        ("classifier",model_base)
                ]
            )

# # Combine sampling techniques
# rs_under = RandomUnderSampler(random_state=93196)
# rs_over = SMOTE(random_state=93196, sampling_strategy=0.75)
# pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
#                         ("over",rs_over),
#                         ("under",rs_under),
#                         ("classifier",model_base)
#                 ]
#             )
#%% Stratified KFold Cross Validation
rcParams['figure.figsize'] = 10,10 #5,5
plt.rcParams.update({'font.size': 16})
plt.figure(dpi=300)
tic = time.perf_counter()
kf = StratifiedKFold(n_splits=10)
np.random.seed(234)
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
no_classes = len(np.unique(y_train))
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
    # for i in range(3):      # iterative training
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
    # print(f'For fold {fold}')

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

toc = time.perf_counter()
cv_time = toc-tic
print(f"Time elapsed {toc-tic:0.4f} seconds")

plt.rcParams.update({'font.size': 16})
plot_conf_matrix(actual_classes, predicted_classes)
cls_report_train = classification_report(actual_classes,predicted_classes)

#%% Confusion Matrix
max_gmean = df_t_best_gmeans['best_gmean'].max()
best_threshold = 0.6816
predictions_best_t = prediction[:,1] > best_threshold
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
    fpr, tpr, t = roc_curve(y_test, y_pred_proba_class1)
    auc_1 = roc_auc_score(y_test, y_pred_proba_class1)
    auc_2 = auc(fpr, tpr)
    print(f"AUC {auc_2:0.2f}")

    #best threshold in ROC
    gmeans = np.sqrt(tpr*(1-fpr))
    ix = np.argmax(gmeans)

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
    # conf_mat = confusion_matrix(y_test,y_preds)
    # sns.heatmap(conf_mat, annot=True,cmap="Blues",fmt="g")
    # plt.xlabel("Predicted values")
    # plt.ylabel("Actual Values")
    # plt.title("Confusion Matrix")
    plot_conf_matrix(y_test,y_preds)

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
model_base = RandomForestClassifier(random_state=93186,
                                n_jobs=-1,
                                verbose=1)
                                # n_estimators=1000)
                               # max_features='auto')

# Best Model
# model_best = RandomForestClassifier(n_jobs=-1,
#                                     verbose=1,
#                                     n_estimators=500,
#                                     max_features='sqrt',
#                                     min_samples_split=2,
#                                     criterion='gini',
#                                     max_depth=10,
#                                     min_samples_leaf=4)

# Single sampling technique
rs_down = RandomUnderSampler(random_state=93196)
rs_smote = SMOTE(random_state=93196)

pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor),
                        # ("sampler",rs_down),
                        ("classifier",model_base)
                ]
            )

pipe_base_rfe = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_down),
                        ("classifier",model_base)
                ]
            )


#*************************************
# # Combine sampling techniques
# rs_under = RandomUnderSampler(random_state=93196, sampling_strategy=0.5)
# rs_over = SMOTE(random_state=93196, sampling_strategy=0.1)
# pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor),
#                         ("over",rs_over),
#                         ("under",rs_under),
#                         ("classifier",model_base)
#                 ]
#             )
# ************************************
# tic = time.perf_counter()

# # model_base = RandomForestClassifier(random_state=93186,
# #                                     n_jobs=-1,
# #                                     verbose=1,
# #                                     n_estimators=1000,
# #                                     max_features='auto')

# model_fitted = RandomForestClassifier(random_state=93186,
#                                     n_jobs=-1,
#                                     verbose=1,
#                                     n_estimators=1000,
#                                     max_features='sqrt',
#                                     min_samples_split=10,
#                                     criterion='gini')

pipe_base.fit(X_train,y_train)
y_pred_base = pipe_base.predict(X_test)
y_pred_proba_base = pipe_base.predict_proba(X_test)
y_pred_proba_class1 = y_pred_proba_base[::,1]

# toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")

evaluateFinalModel(y_test,y_pred_proba_class1,y_pred_base)

# RFE
pipe_base_rfe.fit(X_train_rfe,y_train_rfe)
y_pred_base = pipe_base_rfe.predict(X_test_rfe)
y_pred_proba_base = pipe_base_rfe.predict_proba(X_test_rfe)
y_pred_proba_class1 = y_pred_proba_base[::,1]

### Evaluate final model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1,y_pred_base)
#%% Saving the model

# change directory
cd_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy\RF\RF-SMOTE\Base model"
os.chdir(cd_path)
# print("Current working directory: {0}".format(os.getcwd()))
pickle.dump(model_base, open("rf_SMOTE_base_model.pickle", "wb"))

# Feature importance of the model

# **************************************************************************************

#%% Recursive Feature Elimination

from sklearn.feature_selection import RFECV
# from sklearn.model_selection import RepeatedKFold

base_model = RandomForestClassifier(random_state=93186,
                                n_jobs=-1,
                                verbose=1)


# # _____ REFCV ________OLD
# preprocessor_rfe = ColumnTransformer(
#     transformers=[
#         ('categorical',OH_Encoder,categorical_features),
#         ('numeric',Normaliser,numeric_features)
#         ],
#     remainder='passthrough'
#     )

# _____ REFCV ________NEW
preprocessor_new_rfe = ColumnTransformer(
    transformers=[
        ('categorical',OH_Encoder,categorical_features),
        ('numerical',ST_scalar,numerical_features),
        ('binner',K_binner, bin_features),
        ],
    remainder='passthrough'
    )


#balancing the dataset
rs_down = RandomUnderSampler(random_state=93196)
# X_train_sampled, y_train_sampled = rs_down.fit_sample(X_train,y_train)

rfecv_selector = RFECV(estimator=base_model, step=1, cv=StratifiedKFold(n_splits=10), scoring='f1', min_features_to_select=1)
pipe = Pipeline(steps=[ ("preprocessor",preprocessor_new_rfe),
                        ("sampler",rs_down),
                        ("feature_selector",rfecv_selector),
                        ("classifier",base_model)
                ]
            )
pipe.fit(X_train, y_train)

# X_transformed = preprocessor.fit(X_train_sampled)
# y_train_sampled=y_train_sampled.to_numpy()
# X_train_sampled=X_train_sampled.to_numpy()
# rfecv_selector.fit(X_transformed, y_train_sampled)


# Get the names of the transformed features
transformed_feature_names = pipe["preprocessor"].get_feature_names_out(input_features=X_train.columns)

#Converting transformed_feature_names list into enumerate
selected_feature_names = [feature for i, feature in enumerate(transformed_feature_names)]# if rfecv_selector.ranking_[i]]

# Combine feature names with their rankings
feature_rankings_with_names = dict(zip(selected_feature_names, pipe["feature_selector"].ranking_)) # replaced list with dict to format the variable
# Display selected feature names and their rankings
print("Selected Feature Names with Rankings:")

# printing feature with rank in a more readable manner
pprint(feature_rankings_with_names)

# for feature_name, ranking in feature_rankings_with_names:
#     print(f"{feature_name}: Rank {ranking}")

sorted_feature_rankings_with_names = sorted(feature_rankings_with_names.items(), key=lambda x: x[1])

for key, value in sorted_feature_rankings_with_names:
    print(f"{value}: {key}")


#%% Hyperparameter Tuning - Randomized CV (Common for all 3 sampling techniques)
np.random.seed(93186)
rs_down_tune = RandomUnderSampler()
# rs_smote_tune = SMOTE(n_jobs=-1)
model_tune_rs = RandomForestClassifier()
pipe_tune_rs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                            # ("sampler",rs_down_tune),
                            ("classifier",model_tune_rs)
                            ]
                     )
## Randomized Grid Search
# Number of trees in random forest
n_estimators = [100,200,500] #Removed 750,1000,1200 onlye for SMOTE tuning cause of memory issue
# Criterion for splitting
criterion = ['gini', 'entropy']
# Number of features to consider at every split
max_features = ['sqrt']
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Maximum number of levels in tree
max_depth = [10,20,30,40,50] # for SMOTE
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] #upper limit kept at 110 for sonw and nosample
max_depth.append(None)
# # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None]

# Create the random search grid
param_grid_rs = {'classifier__n_estimators': n_estimators,
                'classifier__criterion':criterion,
                'classifier__max_features': max_features,
                'classifier__min_samples_split': min_samples_split,
                'classifier__min_samples_leaf':min_samples_leaf,
                'classifier__max_depth':max_depth}
print(param_grid_rs)
# print(random_grid)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
random_cv = RandomizedSearchCV(estimator= pipe_tune_rs,
                                param_distributions = param_grid_rs,
                                n_iter = 50, #150 for only dosnwampling
                                cv = 3,
                                scoring = "roc_auc",
                                verbose = 2,
                                random_state = 456,
                                n_jobs = -1,
                                error_score ='raise',
                                return_train_score = True)

random_cv.fit(X_train,y_train)
tic = time.perf_counter()
#Best model parameters
random_cv.best_params_
random_cv.best_score_
results_cv = pd.DataFrame(random_cv.cv_results_)

#Plot randomizedCV results
#plot 1
sns.catplot(x='param_classifier__n_estimators', y='mean_test_score', kind='bar',
            col="param_classifier__criterion", data=results_cv, hue='param_classifier__max_features')
#plot 2
sns.catplot(x='param_classifier__n_estimators', y='mean_train_score', kind='bar',
            col="param_classifier__criterion", data=results_cv, hue='param_classifier__max_features')

#plot 1.1
sns.relplot(x='param_classifier__n_estimators', y='mean_test_score', kind='line',
            col="param_classifier__criterion", data=results_cv,
            hue='param_classifier__max_features', style='param_classifier__max_features')
# At this point selected n_estimator=500, criteron='gini'
#plot 4
sns.relplot(x='param_classifier__max_depth', y='mean_test_score', kind='line',
            data=results_cv,hue='param_classifier__max_features',
            style='param_classifier__max_features')
#plot 5
sns.relplot(x='param_classifier__n_estimators', y='mean_test_score', kind='line',
            data=results_cv, hue='param_classifier__min_samples_leaf',
            style='param_classifier__max_features')

#plot 6
sns.relplot(x='param_classifier__min_samples_split', y='mean_test_score', kind='line',
            data=results_cv, hue='param_classifier__max_features')
# plot 7
sns.relplot(x='param_classifier__max_depth', y='mean_test_score', kind='line',
            data=results_cv, hue='param_classifier__min_samples_split')

#%% Bayesian Optimization
np.random.seed(93186)
rs_down_tune = RandomUnderSampler()
# rs_smote_tune = SMOTE(n_jobs=-1)
model_tune_rs = RandomForestClassifier()
pipe_tune_rs = Pipeline(steps=[ ("preprocessor",preprocessor),
                            ("sampler",rs_down_tune),
                            ("classifier",model_tune_rs)
                            ]
                     )
## Randomized Grid Search
# Number of trees in random forest
n_estimators = [100,200,300,400,500,600,700,800,900,1000] #Removed 750,1000,1200 onlye for SMOTE tuning cause of memory issue
# Criterion for splitting
criterion = ['gini', 'entropy']
# Number of features to consider at every split
max_features = ['sqrt','log2']
# Minimum number of samples required to split a node
min_samples_split = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,3,4,5]
# Maximum number of levels in tree
max_depth = [10,20,30,40,50] # for SMOTE
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] #upper limit kept at 110 for sonw and nosample
# max_depth.append(None)
# # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None]

# Create the random search grid
param_grid_rs = {'classifier__n_estimators': n_estimators,
                'classifier__criterion':criterion,
                'classifier__max_features': max_features,
                'classifier__min_samples_split': min_samples_split,
                'classifier__min_samples_leaf':min_samples_leaf,
                'classifier__max_depth':max_depth}
print(param_grid_rs)
# print(random_grid)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
# random_cv = RandomizedSearchCV(estimator= pipe_tune_rs,
#                                param_distributions = param_grid_rs,
#                                n_iter = 50, #150 for only dosnwampling
#                                cv = 3,
#                                scoring = "roc_auc",
#                                verbose = 2,
#                                random_state = 456,
#                                n_jobs = -1,
#                                error_score ='raise',
#                                return_train_score = True)

bayes_cs = BayesSearchCV(estimator=pipe_tune_rs,
                         search_spaces=param_grid_rs,
                         n_iter = 100,
                         cv = 3,
                         scoring = "roc_auc",
                         verbose = 2,
                         random_state = 456,
                         n_jobs = -1)

# random_cv.fit(X_train,y_train)
bayes_cs.fit(X_train,y_train)
tic = time.perf_counter()
#Best model parameters
# random_cv.best_params_
# random_cv.best_score_
bayes_cs.best_params_
bayes_cs.best_score_
# results_cv = pd.DataFrame(random_cv.cv_results_)
results_cv = pd.DataFrame(bayes_cs.cv_results_)
#Plot randomizedCV results
#plot 1
sns.catplot(x='param_classifier__n_estimators', y='mean_test_score', kind='bar',
            col="param_classifier__criterion", data=results_cv, hue='param_classifier__max_features')

#plot 2
sns.catplot(x='param_classifier__n_estimators', y='mean_train_score', kind='bar',
            col="param_classifier__criterion", data=results_cv, hue='param_classifier__max_features')

#plot 1.1
sns.relplot(x='param_classifier__n_estimators', y='mean_test_score', kind='line',
            col="param_classifier__criterion", data=results_cv,
            hue='param_classifier__max_features', style='param_classifier__max_features')
# At this point selected n_estimator=500, criteron='gini'
#plot 4
sns.relplot(x='param_classifier__max_depth', y='mean_test_score', kind='line',
            data=results_cv,hue='param_classifier__max_features',
            style='param_classifier__max_features')
#plot 5
sns.relplot(x='param_classifier__n_estimators', y='mean_test_score', kind='line',
            data=results_cv, hue='param_classifier__min_samples_leaf',
            style='param_classifier__max_features')

#plot 6
sns.relplot(x='param_classifier__min_samples_split', y='mean_test_score', kind='line',
            data=results_cv, hue='param_classifier__max_features')
# plot 7
sns.relplot(x='param_classifier__max_depth', y='mean_test_score', kind='line',
            data=results_cv, hue='param_classifier__min_samples_split')
#%% Downsample Tuning
## Grid SearchCv
# Create the parameter grid based on the results of random search
# max_depth_gs = [5,10]
# max_depth_gs.append(None)
param_grid_gs = {
    'classifier__n_estimators': [100,500,1000],
    'classifier__criterion':['gini'],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [10,50,80,100],
    'classifier__min_samples_leaf': [1,2,4],
    'classifier__min_samples_split': [2,4],
}

model_tune_gs = RandomForestClassifier(random_state=93186, max_depth='None')
gs_tune = RandomUnderSampler(random_state=93196)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                             ("sampler",gs_tune),
                            ("classifier",model_tune_gs),
                            ]
                     )
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
tic = time.perf_counter()
# Best model parameters
grid_search.best_params_
grid_search.best_score_

results_cv_gs= pd.DataFrame(grid_search.cv_results_)
#%% No-Sample Tuning
## Grid SearchCv
# Create the parameter grid based on the results of random search
# max_depth_gs = [5,10]
# max_depth_gs.append(None)
param_grid_gs = {
    'classifier__n_estimators': [200,500,1200],
    'classifier__criterion':['gini'],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [5,10,20],
    'classifier__min_samples_leaf': [1,2,4],
    'classifier__min_samples_split': [2,4],
}

model_tune_gs = RandomForestClassifier(random_state=93186)
gs_tune = RandomUnderSampler(random_state=93196)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                            # ("sampler",gs_tune),
                            ("classifier",model_tune_gs),
                            ]
                     )
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
tic = time.perf_counter()
#Best model parameters
grid_search.best_params_
grid_search.best_score_
results_cv_gs= pd.DataFrame(grid_search.cv_results_)

#%% SMOTE Tuning
## Grid SearchCv
# Create the parameter grid based on the results of random search

param_grid_gs = {
    'classifier__n_estimators': [200,500,750,1000],
    'classifier__criterion':['gini'],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [5,10,None],
    'classifier__min_samples_leaf': [1,2],
    'classifier__min_samples_split': [2,5,10],
}

model_tune_gs = RandomForestClassifier(random_state=93186)
gs_tune = SMOTE(random_state=93196)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                            ("sampler",gs_tune),
                            ("classifier",model_tune_gs),
                            ]
                     )
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
tic = time.perf_counter
grid_search.best_params_
grid_search.best_score_
results_cv_gs= pd.DataFrame(grid_search.cv_results_)

#%% Retrainng the Best Model
tic = time.perf_counter()
np.random.seed(20)
rs_down_best = RandomUnderSampler()
rs_smote_best = SMOTE()

# Best Model - Downsample
model_best_downsample = RandomForestClassifier(n_jobs=-1,
                                    verbose=1,
                                    n_estimators=1000,
                                    max_features='sqrt',
                                    min_samples_split=2,
                                    criterion='gini',
                                    max_depth=10,
                                    min_samples_leaf=2)

# Best Model - NoSample
model_best_nosample = RandomForestClassifier(n_jobs=-1,
                                    verbose=1,
                                    n_estimators=1200,
                                    max_features='sqrt', #sqrt
                                    min_samples_split=2,
                                    criterion='gini',
                                    max_depth=10,
                                    min_samples_leaf=4,
                                    class_weight ='balanced')

# Best Model - SMOTE
model_best_smote = RandomForestClassifier(n_jobs=-1,
                                    verbose=1,
                                    n_estimators=1000,
                                    max_features='sqrt',
                                    min_samples_split=10,
                                    criterion='gini',
                                    max_depth=None,
                                    min_samples_leaf=2)

# rs_under = RandomUnderSampler(random_state=93196, sampling_strategy=0.5)
# rs_over = SMOTE(random_state=93196, sampling_strategy=0.5)
pipe_tune_best = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        # ("sampler",rs_smote_best),
                        ("classifier",model_best_nosample)
                ]
            )

pipe_tune_best.fit(X_train, y_train)
y_pred_best = pipe_tune_best.predict(X_test)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test)
y_pred_proba_class1_best = y_pred_proba_best[::,1] #single colon is also okay
toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")
### Evaluate final Best model
evaluateFinalModel(y_test,y_pred_proba_class1_best,y_pred_best)

best_threshold,best_f1_score = tune_threshold(y_test, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with OPTIMAL THRESHOLD
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test,y_pred_class1)
print(cls_report_final_bestThreshold)


# RFE Execution
pipe_tune_best.fit(X_train_rfe, y_train_rfe)
y_pred_best = pipe_tune_best.predict(X_test_rfe)
y_pred_proba_best = pipe_tune_best.predict_proba(X_test_rfe)
y_pred_proba_class1_best = y_pred_proba_best[::,1] #single colon is also okay
toc = time.perf_counter()
# print(f"Time elapsed {toc-tic:0.4f} seconds")
### Evaluate final Best model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1_best,y_pred_best)

best_threshold,best_f1_score = tune_threshold(y_test_rfe, y_pred_proba_class1_best, y_pred_best)

#Confusion matrix with OPTIMAL THRESHOLD
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_best]
plot_conf_matrix(y_test_rfe, y_pred_class1)
cls_report_final_bestThreshold = classification_report(y_test_rfe,y_pred_class1)
print(cls_report_final_bestThreshold)
# evaluateFinalModel(y_test,y_pred_class1,y_pred_best)
#%% Feature Permutation Importance
tic=time.perf_counter()
plt.rcParams.update({'font.size': 14})
rcParams['figure.figsize'] = 15,12
per_imp = permutation_importance(pipe_tune_best, X_test, y_test, scoring='roc_auc') #pipe_tune_best
sorted_idx = per_imp.importances_mean.argsort()
first_x = sorted_idx[:25]
plt.barh(X_train.columns[first_x], per_imp.importances_mean[first_x], left=True)
plt.xlabel("Permutation Importance")
toc = time.perf_counter()

print(f"Time elapsed {toc-tic:0.4f} seconds")
plt.title('Feature Importance')
#%% Save the Model
# change directory
# cd_path =r"C:\Users\Darsha Jayamini\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_100000\RF_tuning&bestModel_DownSample\OHE_ALL"
cd_path =r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy\RF\RFE\SMOTE"
os.chdir(cd_path)
# print("Current working directory: {0}".format(os.getcwd()))
# pickle.dump(model_best, open("rf_smote_100000_model_base.pkl", "wb"))
pickle.dump(model_best_smote, open("rf_noSample_rfe_tuned.pkl", "wb"))
# pickle.dump(model_best, open("rf_down_100000_num_ohe_model_Best.pickle", "wb"))
#%% Combine Sampling Techniques - SMOTE + Undersample
tic = time.perf_counter()
np.random.seed(20)
rs_down = RandomUnderSampler()
rs_smote = SMOTE(sampling_strategy=0.75)
model_best = RandomForestClassifier(n_jobs=-1,
                                    verbose=1,
                                    n_estimators=1000)
                                    # max_features='sqrt',
                                    min_samples_split=2,
                                    # criterion='gini',
                                    # max_depth=10,
                                    # min_samples_leaf=4)
# pipe_tune_best = Pipeline(steps=[ ("preprocessor",preprocessor),
#                             ("sampler",rs_smote_best),
#                             ("classifier",model_best)
#                             ]
#                      )
# rs_under = RandomUnderSampler(random_state=93196, sampling_strategy=0.5)
# rs_over = SMOTE(random_state=93196, sampling_strategy=0.5)
pipe_tune_combSamp = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("up_sampler",rs_smote),
                        ("down_sampler",rs_down),
                        ("classifier",model_best)
                ]
            )

#%% Finding threshold & Re-run the model - Down-sample
# Load the model
# load_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_Downsample_Numeric_TE_fullSet"
load_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_Downsample_Num_noDomIndxMonth_fullSet_bestModel"
load_model = pickle.load(open(os.path.join(load_path,"rf_down_model_best.pkl"),'rb'))

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

X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="RF-Downsample")

plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_test2, y_pred_load_test)
# evaluateFinalModel(y_test2, y_pred_class1_test, y_pred_load_test)

#%% Finding threshold & Re-run the model - SMOTE
# Load the model
# load_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_SMOTE_Numeric_TE_fullSet"
load_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_SMOTE_Num_noDomIndxMonth_fullSet_bestModel"
# load_model = pickle.load(open(os.path.join(load_path,"rf_down_fullSet_numeric_TE_model_Best.pickle"),'rb'))
load_model = pickle.load(open(os.path.join(load_path,"rf_smote_model_best.pkl"),'rb'))


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

### Evaluate final Best model
plt.rcParams.update({'font.size': 14})
plt.figure(dpi=600)
plt.rcParams["figure.figsize"] = (5,5)

evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)
#AUC 0.77 for combined sampling best model
best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

#Confusion matrix with optimal threshold
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_val, y_pred_class1)

# Evaluate on test data
y_pred_load_test = pipe_load.predict(X_test2)
y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_test2, y_pred_class1_test)

X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="RF-SMOTE")

# plot_conf_matrix(y_test2, y_pred_load_test)
# evaluateFinalModel(y_test2, y_pred_class1_test, y_pred_load_test)

#%% Finding threshold & Re-run the model - NoSample
# Load the model
# load_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_NoSample_Numeric_TE_fullSet\NoSample tuning"
load_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\Random Forest\RF_Tuned_Models_FullSet\RF_NoSample_Num_NoDomIndxMonth_fullset_bestModel"
load_model = pickle.load(open(os.path.join(load_path,"rf_noSample_model_best.pkl"),'rb'))

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

### Evaluate final Best model
plt.rcParams.update({'font.size': 14})
plt.figure(dpi=600)
plt.rcParams["figure.figsize"] = (5,5)

evaluateFinalModel(y_val,y_pred_proba_class1,y_pred_load)
#AUC 0.77 for combined sampling best model
best_threshold,best_f1_score = tune_threshold(y_val, y_pred_proba_class1, y_pred_load)

#Confusion matrix with optimal threshold
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_val, y_pred_class1)

# Evaluate on test data
y_pred_load_test = pipe_load.predict(X_test2)
y_pred_proba_load_test = pipe_load.predict_proba(X_test2)
y_pred_proba_class1_test = y_pred_proba_load_test[::,1] #single colon is also okay
y_pred_class1_test = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1_test]

# plt.rcParams.update({'font.size': 16})
plot_conf_matrix(y_test2, y_pred_class1_test)

X_test2_transformed = preprocessor.transform(X_test2)
plot_roc_curve(load_model, X_test2_transformed, y_test2, name="RF-NoSample")


#%%  --------------  COMBINED techniques: Random Undersampling and SMOTE-------------

# Train and Test Base model
tic = time.perf_counter()
model_base = RandomForestClassifier(random_state=93186,
                                n_jobs=-1,
                                verbose=1)
                                # n_estimators=1000)

rs_down = RandomUnderSampler( random_state=93196, sampling_strategy=0.5)
rs_smote = SMOTE(random_state=93196, sampling_strategy=0.1)


pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler_up", rs_smote),
                         ("sampler_down",rs_down),
                        ("classifier",model_base)
                ]
            )

for i in range(1,101):
    pipe_base.fit(X_train_rfe,y_train_rfe)
    print(i)

y_pred_base = pipe_base.predict(X_test_rfe)
y_pred_proba_base = pipe_base.predict_proba(X_test_rfe)
y_pred_proba_class1 = y_pred_proba_base[::,1]
toc = time.perf_counter()

print("Runtime: ",toc-tic)
# Evaluate final model
evaluateFinalModel(y_test_rfe,y_pred_proba_class1,y_pred_base)

#%% ------ Hyperparameter Tuning: COMBINED techniques- RUS and SMOTE
param_grid_gs = {
    'classifier__n_estimators': [100,500,1000],
    'classifier__criterion':['gini'],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [10,50,80,100],
    'classifier__min_samples_leaf': [1,2,4],
    'classifier__min_samples_split': [2,4],
}

rs_down = RandomUnderSampler( random_state=93196, sampling_strategy=0.5)
rs_smote = SMOTE(random_state=93196, sampling_strategy=0.1)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler_up", rs_smote),
                         ("sampler_down",rs_down),
                        ("classifier",model_base)
                ]
            )

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
tic = time.perf_counter()
# Best model parameters
grid_search.best_params_
grid_search.best_score_


model_tuned = RandomForestClassifier(random_state=93186,
                                n_jobs=-1,
                                verbose=1,
                                criterion='gini',
                                max_depth=10,
                                max_features='sqrt',
                                min_samples_leaf=4,
                                min_samples_split=2,
                                n_estimators=1000)

rs_down = RandomUnderSampler( random_state=93196, sampling_strategy=0.5)
rs_smote = SMOTE(random_state=93196, sampling_strategy=0.1)


pipe_tuned = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                        ("sampler_up", rs_smote),
                         ("sampler_down",rs_down),
                        ("classifier",model_tuned)
                ]
            )

for i in range(1,101):
    pipe_tuned.fit(X_train_rfe,y_train_rfe)
    print(i)

y_pred_tuned = pipe_tuned.predict(X_test_rfe)
y_pred_proba_tuned = pipe_tuned.predict_proba(X_test_rfe)
y_pred_proba_class1 = y_pred_proba_tuned[::,1]
toc = time.perf_counter()

print("Runtime: ",toc-tic)
# Evaluate final model

results_cv_gs= pd.DataFrame(grid_search.cv_results_)
evaluateFinalModel(y_test_rfe,y_pred_proba_class1,y_pred_tuned)


best_threshold,best_f1_score = tune_threshold(y_test_rfe, y_pred_proba_class1, y_pred_tuned)

#Confusion matrix with optimal threshold
y_pred_class1 = [(1 if x>=best_threshold else 0) for x in y_pred_proba_class1]
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
plot_conf_matrix(y_test_rfe, y_pred_class1)

cls_report_final = classification_report(y_test_rfe,y_pred_tuned)
print(cls_report_final)

#%% No-Sample Tuning
## Grid SearchCv
# Create the parameter grid based on the results of random search
# max_depth_gs = [5,10]
# max_depth_gs.append(None)
param_grid_gs = {
    'classifier__n_estimators': [200,500,1200],
    'classifier__criterion':['gini'],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [5,10,20],
    'classifier__min_samples_leaf': [1,2,4],
    'classifier__min_samples_split': [2,4],
}

model_tune_gs = RandomForestClassifier(random_state=93186)
gs_tune = RandomUnderSampler(random_state=93196)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                            # ("sampler",gs_tune),
                            ("classifier",model_tune_gs),
                            ]
                     )
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
tic = time.perf_counter()
#Best model parameters
grid_search.best_params_
grid_search.best_score_
results_cv_gs= pd.DataFrame(grid_search.cv_results_)

#%% SMOTE Tuning
## Grid SearchCv
# Create the parameter grid based on the results of random search

param_grid_gs = {
    'classifier__n_estimators': [200,500,750,1000],
    'classifier__criterion':['gini'],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [5,10,None],
    'classifier__min_samples_leaf': [1,2],
    'classifier__min_samples_split': [2,5,10],
}

model_tune_gs = RandomForestClassifier(random_state=93186)
gs_tune = SMOTE(random_state=93196)
pipe_tune_gs = Pipeline(steps=[ ("preprocessor",preprocessor_rfe),
                            ("sampler",gs_tune),
                            ("classifier",model_tune_gs),
                            ]
                     )
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = pipe_tune_gs,
                           param_grid = param_grid_gs, scoring='roc_auc',
                           cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train_rfe,y_train_rfe)
tic = time.perf_counter
grid_search.best_params_
grid_search.best_score_
results_cv_gs= pd.DataFrame(grid_search.cv_results_)

# The script is used for CD prediction
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.utils import shuffle
from keras.layers import Bidirectional
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras import backend as K
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


data_converted_no = pd.read_csv('data_converted_no.csv', header = 0)
data_converted_yes = pd.read_csv('data_converted_yes.csv', header = 0)
data_converted_no = data_converted_no[~data_converted_no['ResultDate'].isna()]
data_converted_yes = data_converted_yes[~data_converted_yes['ResultDate'].isna()]
#******************************************************************************
# For some characteristcs of the study population
data_converted_yes_no = data_converted_yes.append(data_converted_no)
# data_converted_yes_no = data_converted_yes # only with CD
ID_yes_no = data_converted_yes_no['PatientID'].drop_duplicates()
demo_yes_no = demo[demo['PatientID'].isin(ID_yes_no)]
demo_yes_no = demo_yes_no.drop_duplicates()
# gender information
gender_yes_no = demo_yes_no[['PatientID', 'Gender']].drop_duplicates() # 11055
gender_yes_no['Gender'].value_counts()
# race information
race_yes_no = demo_yes_no[['PatientID', 'Race']].drop_duplicates()
race = race_yes_no['Race'].str.strip().replace('.', '')
race_yes_no_nan = race_yes_no[race_yes_no['Race'].isna()]
race_yes_no = race_yes_no[~race_yes_no['Race'].isna()]
race_white = ['White', 'White/Caucasian', 'White-Italian', 'C-CAUCASIAN']
race_unknown = ['Patient Declined', 'X-PATIENT DECLINED', 'Refuse to Report/ Unreported']
race_non_white = [elem for elem in race.unique().tolist() if not ((elem in race_white) or (elem in race_unknown))]
race_yes_no_white = race_yes_no[race.isin(race_white)].drop_duplicates()
race_yes_no_unknown = race_yes_no[race.isin(race_unknown)].drop_duplicates()
n_race_yes_no_unknown = len(race_yes_no_unknown)+ len(race_yes_no_nan)
race_yes_no_nonwhite = race_yes_no[race.isin(race_non_white)].drop_duplicates()
# CVH information
HGBA1C = data_converted_yes_no[data_converted_yes_no['ObservationCategory'] == 'HGBA1C']
LDL = data_converted_yes_no[data_converted_yes_no['ObservationCategory'] == 'LDL']
BMI = data_converted_yes_no[data_converted_yes_no['ObservationCategory'] == 'BMI']
Smoking = data_converted_yes_no[data_converted_yes_no['ObservationCategory'] == 'Smoking Status']
BP = data_converted_yes_no[data_converted_yes_no['ObservationCategory'] == 'Blood Pressure']
HGBA1C['ResultNumeric'][HGBA1C['ResultNumeric']!=-9].mean()
HGBA1C['ResultNumeric'][HGBA1C['ResultNumeric']!=-9].std()
LDL['ResultNumeric'][LDL['ResultNumeric']!=-9].mean()
LDL['ResultNumeric'][LDL['ResultNumeric']!=-9].std()
BMI['ResultNumeric'][BMI['ResultNumeric']!=-9].mean()
BMI['ResultNumeric'][BMI['ResultNumeric']!=-9].std()
data_converted_no1 = pd.read_csv('data_converted_no_bfc.csv', header = 0)
data_converted_yes1 = pd.read_csv('data_converted_yes_bfc.csv', header = 0)
data_converted_yes_no1 = data_converted_yes1.append(data_converted_no1)
data_converted_yes_no1 = data_converted_yes_no1[~data_converted_yes_no1['ResultDate'].isna()]
data_converted_yes_no1 = data_converted_yes_no1[['PatientID','ResultDate','ObservationCategory', 'ResultString']]
data_converted_yes_no1 = data_converted_yes_no1.drop_duplicates()
BP = data_converted_yes_no1[data_converted_yes_no1['ObservationCategory'] == 'Blood Pressure']
bp = BP['ResultString'].str.split('/')
bps = bp.str[0].astype(int)
bpd = bp.str[1].fillna(0).astype(int)
bps[bps!=-9].mean()
bps[bps!=-9].std()
bpd[bpd!=-9].mean()
bpd[bpd!=-9].std()
Smoking[['PatientID','Result']].drop_duplicates()['Result'].value_counts()
counts_no = data_converted_no.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_no.iloc[0]/counts_no.iloc[0,:].sum() # BMI
counts_no.iloc[:,0].sum() # ideal
counts_yes = data_converted_yes.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_yes.iloc[0]/counts_yes.iloc[0,:].sum() # BMI
counts_yes.iloc[:,0].sum() # ideal
# plot one particular patients timeline for visualization
patient_1 = data_converted_yes_no[data_converted_yes_no['PatientID']==620]
patient_1 = patient_1[['ResultDate', 'ObsResult']]
# The end of the characteristcs of data checking
#******************************************************************************
# Next create one new column to store the result name + result
data_converted_yes['ObsResult'] = data_converted_yes['ObservationCategory'].astype(str) + data_converted_yes['Result'].astype(str)
data_converted_no['ObsResult'] = data_converted_no['ObservationCategory'].astype(str) + data_converted_no['Result'].astype(str)
data_converted_yes_sub = data_converted_yes[['PatientID', 'ObsResult']]
data_converted_no_sub = data_converted_no[['PatientID', 'ObsResult']]
data_converted_sub1 = data_converted_yes_sub.append(data_converted_no_sub)
data_converted_sub = data_converted_sub1
data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Smoking Statusideal', 'SmokingStatusideal')
data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Smoking Statusintermediate', 'SmokingStatusintermediate')
data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Smoking Statuspoor', 'SmokingStatuspoor')
data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Blood Pressureideal', 'BloodPressureideal')
data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Blood Pressureintermediate', 'BloodPressureintermediate')
data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Blood Pressurepoor', 'BloodPressurepoor')
data_converted_sub['ObsResult'] = data_converted_sub['ObsResult'].str.lower()
# data_converted_sub.to_csv('data_converted_sub.csv', header = True, index = False)
date = data_converted_yes['ResultDate'].append(data_converted_no['ResultDate'])
data_converted_sub_date = pd.concat([data_converted_sub, date],axis = 1)

# Next add the time information
from datetime import datetime
date_format = "%Y-%m-%d"
data_days = data_converted_sub_date
data_days = data_days[~data_days['ResultDate'].isna()] # if the date is nan, just remove
data_days['Days'] = 0
data_days_ID = data_days['PatientID'].unique()
n_days = len(data_days_ID)
k = 0
# for i in range(0, n_days):
#     print(i)
#     x = data_days_ID[i]
#     dat_x = data_days[data_days['PatientID']== x]
#     n_x = len(dat_x)
#     t2 = dat_x.iloc[(n_x-1),2]
#     t2 = datetime.strptime(t2, date_format)
#     for j in range(0, (n_x-1)):
#         # print(j)
#         t1 = dat_x.iloc[j,2]
#         delta = datetime.strptime(t1, date_format)-t2
#         data_days.iloc[k, 3] = delta.days
#         k = k+1
#     k = k+1
# data_days.to_csv('data_days_revision.csv', header = True, index = False)
data_days = pd.read_csv('data_days_revision.csv', header = 0)

# # Add delta t info and use embedding (Word2Vec) -- start
# data_days padding to the same rows numbers for each patient
chd = data_days
nM = chd['PatientID'].value_counts().max() # make the same number rows for each patient
id_chd = chd['PatientID'].unique()
n_row = len(id_chd)
column_pad = chd.columns.tolist()
patients_pad = pd.DataFrame(columns= column_pad)
# Convert the rows to columns for one patient
# for i in range(0, n_row):
#     x = id_chd[i]
#     print(i)
#     dat_x = chd[chd['PatientID']== x] # find out the subset for each id
#     n_row_x = len(dat_x)
#     n_pad = nM - n_row_x
#     dat_x_pad = pd.DataFrame('0', index=np.arange(n_pad), columns=column_pad)
#     dat_x_pad['PatientID'] = x
#     dat_y = dat_x_pad.append(dat_x)
#     patients_pad = patients_pad.append(dat_y)
# patients_pad.to_csv('patients_pad_revision.csv', header = True, index = False)
patients_pad = pd.read_csv('patients_pad_revision.csv', header = 0)
# convert each patient to one row for the preparation for the Word2Vec
chd = data_days
chd = chd.sort_values(['PatientID','ResultDate'], ascending=[True,True])
nM = chd['PatientID'].value_counts().max() # make the same number rows for each patient
id_chd = chd['PatientID'].unique()
n_row = len(id_chd)
patient_ID = pd.DataFrame('', index=np.arange(n_row), columns=['PatientID'])
column_pad = chd.columns.tolist()
patients_row = []
patients = []
# Convert the rows to columns for one patient
for i in range(0, n_row):
    x = id_chd[i]
    print(i)
    dat_x = chd[chd['PatientID']== x] # find out the subset for each id
    n_row_x = len(dat_x)
    n_pad = nM - n_row_x
    dat_x_pad = pd.DataFrame('0', index=np.arange(n_pad), columns=column_pad)
    dat_y = dat_x_pad.append(dat_x)
    patient_ID.iloc[i] = x
    y = dat_y['ObsResult'].tolist()
    patients.append(y)
    # patients_row.append(patients)
patients_row = patients
patients_all_v1 = patients_row
patients_row_days = pd.DataFrame(patients_row)
# model = Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
from gensim.models import Word2Vec
n = 32
model = Word2Vec(patients_all_v1, min_count = 1, size = n)
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
model.wv.save_word2vec_format("vectors_15_revision.csv", binary=False) # save the embeddings
# The direct output from embedding
vectors = pd.read_csv("vectors_15_revision.csv", header = 0)
patients_embedding = vectors
n_vector = len(patients_embedding)
nv = np.arange(n+1)
word = pd.DataFrame('', index=np.arange(n_vector), columns=list(nv))
for i in range(0, n_vector):
    vec = patients_embedding.iloc[i]
    values1 = vec.str.split(',').astype(str)
    values = values1[0].split()
    word.iloc[i, 0] = values[0]
    word.iloc[i, 1:] = values[1:]
word1 = word
word_map = word
word_map.iloc[:, 0] = word.iloc[:, 0].str[2:]
word_map.iloc[:, n] = word.iloc[:, n].str[:-2]
word_map.to_csv('word_map_revision.csv', header = True, index = False)
word_map = pd.read_csv('word_map_revision.csv', header = 0)
# Next map back the original data set data_converted_sub
word_map = word_map.rename(columns = {'0':'ObsResult'})
patient_yes_date = data_converted_yes['ResultDate']
patient_no_date = data_converted_no['ResultDate']
patient_date = patient_yes_date.append(patient_no_date)
# data_converted_date = patients_pad.drop(['ResultDate'], axis = 1)
data_converted_date = patients_pad
data_converted_map = data_converted_date.merge(word_map, on = 'ObsResult')
data_converted_map = data_converted_map.sort_values(['PatientID', 'ResultDate'], ascending=[True, True])
# feature = data_converted_map.drop(['ObsResult'], axis = 1)
feature = data_converted_map.drop(['ObsResult','ResultDate'], axis = 1)

chd = feature
id_chd = chd['PatientID'].unique()
n_row = len(id_chd)
patient_ID = pd.DataFrame('', index=np.arange(n_row), columns=['PatientID'])
patients = []
patients2 = []
# Convert the rows to columns for one patient
for i in range(0, n_row):
    x = id_chd[i]
    print(i)
    dat_x = chd[chd['PatientID']== x] # find out the subset for each id
    n_row_x = len(dat_x)
    y = dat_x.iloc[:, 1:(n+2)].values.tolist()
    patients.append(y)

patients_all = patients
patients_all2 = patients2
# Next apply deep learning
word = data_converted_map['ObsResult'].unique()
# top_words = len(word)+1
y_yes = np.zeros(len(data_converted_yes_sub['PatientID'].unique()))+1
y_no = np.zeros(len(data_converted_no_sub['PatientID'].unique()))
y = np.append(y_yes, y_no)
y2 = y.astype(str).tolist()
# y = to_categorical(y)
features1 = np.array(patients_all)
features = features1

max_review_length = nM
look_back=32+1
n_cv = 5

# Use k-fold cross validation
cv = StratifiedKFold(n_splits = n_cv)
classifier = model_v2

X = features
y = y

X, y = shuffle(X, y)

aucs_LSTM = []
aucs_RF = []
aucs_NB = []
aucs_LR = []
aucs_DNN = []

tprs_LSTM = []
tprs_RF = []
tprs_LR = []
tprs_NB = []
tprs_DNN = []

accs_LSTM = []
accs_RF = []
accs_LR = []
accs_NB = []
accs_DNN = []

precs_LSTM = []
precs_RF = []
precs_LR = []
precs_NB = []
precs_DNN = []

recs_LSTM =  []
recs_RF =  []
recs_LR =  []
recs_NB =  []
recs_DNN =  []

f1s_LSTM = []
f1s_RF = []
f1s_LR = []
f1s_NB = []
f1s_DNN = []

specs_LSTM = []
specs_RF = []
specs_LR = []
specs_NB = []
specs_DNN = []

mean_fpr = np.linspace(0, 1, 100)

k = 0
i = 0

for train, test in cv.split(X, y):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    model_v2 = Sequential()
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    # model.add(LSTM(100))
    model_v2.add(LSTM(100, input_shape=(max_review_length, look_back)))
    model_v2.add(Dense(1, activation='sigmoid'))
    model_v2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loss='mean_squared_error'
    print(model_v2.summary())
    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(filepath='best_model'+str(i)+'.h5', monitor='val_loss', save_best_only=True)]
    model_v2.fit(X_train, y_train, epochs=50, callbacks = callbacks, verbose=1,batch_size=64, validation_data=(X_test, y_test))
    model = load_model('best_model'+str(i)+'.h5')
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = (y_pred > 0.5)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    # Next study the ROC values for LSTM
    y_pred_LSTM = model.predict(X_test).ravel()
    fpr_LSTM, tpr_LSTM, thresholds_LSTM = roc_curve(y_test, y_pred_LSTM)
    auc_LSTM = auc(fpr_LSTM, tpr_LSTM)
    tprs_LSTM.append(interp(mean_fpr, fpr_LSTM, tpr_LSTM))
    tprs_LSTM[-1][0] = 0.0
    aucs_LSTM.append(auc_LSTM)
    acc_LSTM = accuracy_score(y_test, y_pred_LSTM.round())
    prec_LSTM = precision_score(y_test, y_pred_LSTM.round())
    rec_LSTM = recall_score(y_test, y_pred_LSTM.round())
    f1_LSTM = f1_score(y_test, y_pred_LSTM.round())
    accs_LSTM.append(acc_LSTM)
    precs_LSTM.append(prec_LSTM)
    recs_LSTM.append(rec_LSTM)
    f1s_LSTM.append(f1_LSTM)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_LSTM.round()).ravel()
    spec_LSTM = tn / (tn+fp)
    specs_LSTM.append(spec_LSTM)
    # save the fpr and tpr for later use
    np.savetxt("revision/fpr_keras_with_time_lstm_ldl"+str(i)+".csv", fpr_LSTM, delimiter=",")
    np.savetxt("revision/tpr_keras_with_time_lstm_ldl"+str(i)+".csv", tpr_LSTM, delimiter=",")
    np.savetxt("revision/thresholds_keras_with_time_lstm_ldl"+str(i)+".csv", thresholds_LSTM, delimiter=",")
    K.clear_session()
    del model
    del model_v2
    # Compared the deep learning to other models.
    n_train, nx, ny = X_train.shape
    X_train = X_train.reshape((n_train,nx*ny))
    n_test, nx, ny = X_test.shape
    X_test = X_test.reshape((n_test,nx*ny))
    # Next compare with random forest
    rf = RandomForestClassifier(max_depth=3, n_estimators=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf0 = rf.predict(X_test)
    print(confusion_matrix(y_test,y_pred_rf0))
    print(classification_report(y_test,y_pred_rf0))
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    # Other metrics
    y_pred_RF = y_pred_rf
    fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_pred_RF)
    auc_RF = auc(fpr_RF, tpr_RF)
    tprs_RF.append(interp(mean_fpr, fpr_RF, tpr_RF))
    tprs_RF[-1][0] = 0.0
    aucs_RF.append(auc_RF)
    acc_RF = accuracy_score(y_test, y_pred_RF.round())
    prec_RF = precision_score(y_test, y_pred_RF.round())
    rec_RF = recall_score(y_test, y_pred_RF.round())
    f1_RF = f1_score(y_test, y_pred_RF.round())
    accs_RF.append(acc_RF)
    precs_RF.append(prec_RF)
    recs_RF.append(rec_RF)
    f1s_RF.append(f1_RF)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_RF.round()).ravel()
    spec_RF = tn / (tn+fp)
    specs_RF.append(spec_RF)
    # save the data
    np.savetxt("revision/fpr_RF_with_time_lstm_ldl"+str(i)+".csv", fpr_rf, delimiter=",")
    np.savetxt("revision/tpr_RF_with_time_lstm_ldl"+str(i)+".csv", tpr_rf, delimiter=",")
    np.savetxt("revision/thresholds_RF_with_time_lstm_ldl"+str(i)+".csv", thresholds_rf, delimiter=",")
    # Next compare with Naive Bayes classifier model
    model_nb = MultinomialNB()
    # df_norm = (df - df.mean()) / (df.max() - df.min())
    X_train = np.array(X_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    X_train_norm = (X_train-X_train.min())/(X_train.max()-X_train.min())
    X_test_norm = (X_test-X_test.min())/(X_test.max()-X_test.min())
    model_nb.fit(X_train_norm, y_train)
    y_pred_nb = model_nb.predict_proba(X_test_norm)[:, 1]
    y_pred_nb0 = model_nb.predict(X_test_norm)
    print(confusion_matrix(y_test,y_pred_nb0))
    print(classification_report(y_test,y_pred_nb0))
    fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_pred_nb)
    auc_nb = auc(fpr_nb, tpr_nb)
    # Other metrics
    y_pred_NB = y_pred_nb
    fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, y_pred_NB)
    auc_NB = auc(fpr_NB, tpr_NB)
    tprs_NB.append(interp(mean_fpr, fpr_NB, tpr_NB))
    tprs_NB[-1][0] = 0.0
    aucs_NB.append(auc_NB)
    acc_NB = accuracy_score(y_test, y_pred_NB.round())
    prec_NB = precision_score(y_test, y_pred_NB.round())
    rec_NB = recall_score(y_test, y_pred_NB.round())
    f1_NB = f1_score(y_test, y_pred_NB.round())
    accs_NB.append(acc_NB)
    precs_NB.append(prec_NB)
    recs_NB.append(rec_NB)
    f1s_NB.append(f1_NB)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_NB.round()).ravel()
    spec_NB = tn / (tn+fp)
    specs_NB.append(spec_NB)
    # save the data
    np.savetxt("revision/fpr_nb_with_time_lstm_ldl"+str(i)+".csv", fpr_nb, delimiter=",")
    np.savetxt("revision/tpr_nb_with_time_lstm_ldl"+str(i)+".csv", tpr_nb, delimiter=",")
    np.savetxt("revision/thresholds_nb_with_time_lstm_ldl"+str(i)+".csv", thresholds_nb, delimiter=",")
    # Next plot logistic regression
    logisticRegr = LogisticRegression() # train the logistic regression model
    logisticRegr.fit(X_train, y_train)
    y_pred = logisticRegr.predict(X_test) # prediction
    y_pred_prob = logisticRegr.predict_proba(X_test)
    # Evaluate the logistic regression model
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr_lr, tpr_lr)
    # Other metrics
    y_pred_LR = y_pred_prob[:,1]
    fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_pred_LR)
    auc_LR = auc(fpr_LR, tpr_LR)
    tprs_LR.append(interp(mean_fpr, fpr_LR, tpr_LR))
    tprs_LR[-1][0] = 0.0
    aucs_LR.append(auc_LR)
    acc_LR = accuracy_score(y_test, y_pred_LR.round())
    prec_LR = precision_score(y_test, y_pred_LR.round())
    rec_LR = recall_score(y_test, y_pred_LR.round())
    f1_LR = f1_score(y_test, y_pred_LR.round())
    accs_LR.append(acc_LR)
    precs_LR.append(prec_LR)
    recs_LR.append(rec_LR)
    f1s_LR.append(f1_LR)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_LR.round()).ravel()
    spec_LR = tn / (tn+fp)
    specs_LR.append(spec_LR)
    # save the data
    np.savetxt("revision/fpr_lr_with_time_lstm_ldl"+str(i)+".csv", fpr_lr, delimiter=",")
    np.savetxt("revision/tpr_lr_with_time_lstm_ldl"+str(i)+".csv", tpr_lr, delimiter=",")
    np.savetxt("revision/thresholds_lr_with_time_lstm_ldl"+str(i)+".csv", thresholds_lr, delimiter=",")
    # DNN
    model_DNN = Sequential()
    model_DNN.add(Dense(256, input_dim= X_train.shape[1], activation='relu'))
    # model_DNN.add(Dense(512, activation='relu'))
    model_DNN.add(Dense(256, activation='relu'))
    # model_DNN.add(Dense(256, activation='relu'))
    model_DNN.add(Dense(128, activation='relu'))
    model_DNN.add(Dense(64, activation='relu'))
    model_DNN.add(Dense(32, activation='relu'))
    # model_DNN.add(Dense(8, activation='relu'))
    model_DNN.add(Dense(1, activation='sigmoid'))
    # Compile model_DNN
    model_DNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model_DNN
    # Set callback functions to early stop training and save the best model_DNN so far
    callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(filepath='best_model_DNN'+str(i)+'.h5', monitor='val_loss', save_best_only=True)]
    model_DNN.fit(X_train, y_train, epochs=50, callbacks = callbacks, verbose=1,batch_size=64, validation_data=(X_test, y_test))
    model_DNN = load_model('best_model_DNN'+str(i)+'.h5')
    scores = model_DNN.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    y_pred = model_DNN.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = (y_pred > 0.5)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    # Next study the ROC values for DNN
    y_pred_DNN = model_DNN.predict(X_test).ravel()
    fpr_DNN, tpr_DNN, thresholds_DNN = roc_curve(y_test, y_pred_DNN)
    auc_DNN = auc(fpr_DNN, tpr_DNN)
    tprs_DNN.append(interp(mean_fpr, fpr_DNN, tpr_DNN))
    tprs_DNN[-1][0] = 0.0
    aucs_DNN.append(auc_DNN)
    acc_DNN = accuracy_score(y_test, y_pred_DNN.round())
    prec_DNN = precision_score(y_test, y_pred_DNN.round())
    rec_DNN = recall_score(y_test, y_pred_DNN.round())
    f1_DNN = f1_score(y_test, y_pred_DNN.round())
    accs_DNN.append(acc_DNN)
    precs_DNN.append(prec_DNN)
    recs_DNN.append(rec_DNN)
    f1s_DNN.append(f1_DNN)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_DNN.round()).ravel()
    spec_DNN = tn / (tn+fp)
    specs_DNN.append(spec_DNN)
    # save the fpr and tpr for later use
    np.savetxt("revision/fpr_keras_with_time_DNN_ldl"+str(i)+".csv", fpr_DNN, delimiter=",")
    np.savetxt("revision/tpr_keras_with_time_DNN_ldl"+str(i)+".csv", tpr_DNN, delimiter=",")
    np.savetxt("revision/thresholds_keras_with_time_DNN_ldl"+str(i)+".csv", thresholds_DNN, delimiter=",")
    K.clear_session()
    del model_DNN
    i = i + 1
# Next write other file needed for the later use
np.savetxt('revision/results2/idx_train.csv', idx_train, delimiter=",")
np.savetxt('revision/results2/idx_test.csv', idx_test, delimiter=",")
np.savetxt('revision/results2/aucs_LSTM.csv', aucs_LSTM, delimiter=",")
np.savetxt('revision/results2/tprs_LSTM.csv', tprs_LSTM, delimiter=",")
np.savetxt('revision/results2/accs_LSTM.csv', accs_LSTM, delimiter=",")
np.savetxt('revision/results2/precs_LSTM.csv', precs_LSTM, delimiter=",")
np.savetxt('revision/results2/recs_LSTM.csv', recs_LSTM, delimiter=",")
np.savetxt('revision/results2/f1s_LSTM.csv', f1s_LSTM, delimiter=",")
np.savetxt('revision/results2/specs_LSTM.csv', specs_LSTM, delimiter=",")
np.savetxt('revision/results2/aucs_RF.csv', aucs_RF, delimiter=",")
np.savetxt('revision/results2/tprs_RF.csv', tprs_RF, delimiter=",")
np.savetxt('revision/results2/accs_RF.csv', accs_RF, delimiter=",")
np.savetxt('revision/results2/precs_RF.csv', precs_RF, delimiter=",")
np.savetxt('revision/results2/recs_RF.csv', recs_RF, delimiter=",")
np.savetxt('revision/results2/f1s_RF.csv', f1s_RF, delimiter=",")
np.savetxt('revision/results2/specs_RF.csv', specs_RF, delimiter=",")
np.savetxt('revision/results2/aucs_LR.csv', aucs_LR, delimiter=",")
np.savetxt('revision/results2/tprs_LR.csv', tprs_LR, delimiter=",")
np.savetxt('revision/results2/accs_LR.csv', accs_LR, delimiter=",")
np.savetxt('revision/results2/precs_LR.csv', precs_LR, delimiter=",")
np.savetxt('revision/results2/recs_LR.csv', recs_LR, delimiter=",")
np.savetxt('revision/results2/f1s_LR.csv', f1s_LR, delimiter=",")
np.savetxt('revision/results2/specs_LR.csv', specs_LR, delimiter=",")
np.savetxt('revision/results2/aucs_NB.csv', aucs_NB, delimiter=",")
np.savetxt('revision/results2/tprs_NB.csv', tprs_NB, delimiter=",")
np.savetxt('revision/results2/accs_NB.csv', accs_NB, delimiter=",")
np.savetxt('revision/results2/precs_NB.csv', precs_NB, delimiter=",")
np.savetxt('revision/results2/recs_NB.csv', recs_NB, delimiter=",")
np.savetxt('revision/results2/f1s_NB.csv', f1s_NB, delimiter=",")
np.savetxt('revision/results2/specs_NB.csv', specs_NB, delimiter=",")
np.savetxt('revision/results2/aucs_DNN.csv', aucs_DNN, delimiter=",")
np.savetxt('revision/results2/tprs_DNN.csv', tprs_DNN, delimiter=",")
np.savetxt('revision/results2/accs_DNN.csv', accs_DNN, delimiter=",")
np.savetxt('revision/results2/precs_DNN.csv', precs_DNN, delimiter=",")
np.savetxt('revision/results2/recs_DNN.csv', recs_DNN, delimiter=",")
np.savetxt('revision/results2/f1s_DNN.csv', f1s_DNN, delimiter=",")
np.savetxt('revision/results2/specs_DNN.csv', specs_DNN, delimiter=",")

#******************************************************************
# Next plot the ROC curves
# LSTM
tprs_LSTM = pd.read_csv('revision/results2/tprs_LSTM.csv', header = None)
aucs_LSTM = pd.read_csv('revision/results2/aucs_LSTM.csv', header = None)

# DNN
tprs_DNN = pd.read_csv('revision/results2/tprs_DNN.csv', header = None)
aucs_DNN = pd.read_csv('revision/results2/aucs_DNN.csv', header = None)

# RF
tprs_RF = pd.read_csv('revision/results2/tprs_RF.csv', header = None)
aucs_RF = pd.read_csv('revision/results2/aucs_RF.csv', header = None)

# LR
tprs_LR = pd.read_csv('revision/results2/tprs_LR.csv', header = None)
aucs_LR = pd.read_csv('revision/results2/aucs_LR.csv', header = None)

# naive Bayes
tprs_NB = pd.read_csv('revision/results2/tprs_NB.csv', header = None)
aucs_NB = pd.read_csv('revision/results2/aucs_NB.csv', header = None)


# For LSTM --- plot
lw1 = 2
lw2 = 1.5
mean_fpr = np.linspace(0, 1, 100)

mean_tpr_LSTM = np.mean(tprs_LSTM, axis=0)
mean_auc_LSTM = auc(mean_fpr, mean_tpr_LSTM)
std_auc_LSTM = np.std(aucs_LSTM)

mean_tpr_DNN = np.mean(tprs_DNN, axis=0)
mean_auc_DNN = auc(mean_fpr, mean_tpr_DNN)
std_auc_DNN = np.std(aucs_DNN)

mean_tpr_RF = np.mean(tprs_RF, axis=0)
mean_auc_RF = auc(mean_fpr, mean_tpr_RF)
std_auc_RF = np.std(aucs_RF)

mean_tpr_LR = np.mean(tprs_LR, axis=0)
mean_auc_LR = auc(mean_fpr, mean_tpr_LR)
std_auc_LR = np.std(aucs_LR)

mean_tpr_NB = np.mean(tprs_NB, axis=0)
mean_auc_NB = auc(mean_fpr, mean_tpr_NB)
std_auc_NB = np.std(aucs_NB)

# Without CI for three cases by 5-fold cross-validation
plt.plot(mean_fpr, mean_tpr_LSTM, color='purple',
         label='LSTM Mean AUC = %0.2f $\pm$ %0.2f' % (mean_auc_LSTM, std_auc_LSTM),
         lw=lw1, alpha=.8)

plt.plot(mean_fpr, mean_tpr_DNN, color='blue',
         label='DNN Mean AUC = %0.2f $\pm$ %0.2f' % (mean_auc_DNN, std_auc_DNN),
         lw=lw1, alpha=.8)

plt.plot(mean_fpr, mean_tpr_RF, color='darkorange',
         label='RF Mean AUC = %0.2f $\pm$ %0.2f' % (mean_auc_RF, std_auc_RF),
         lw=lw1, alpha=.8)

plt.plot(mean_fpr, mean_tpr_LR, color='cyan',
         label='LR Mean AUC = %0.2f $\pm$ %0.2f' % (mean_auc_LR, std_auc_LR),
         lw=lw1, alpha=.8)

plt.plot(mean_fpr, mean_tpr_NB, color='green',
         label='NB Mean AUC = %0.2f $\pm$ %0.2f' % (mean_auc_NB, std_auc_NB),
         lw=lw1, alpha=.8)

plt.plot([0,1],[0,1],'k--',lw=lw2)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(loc="lower right")
plt.savefig('revision/figures2/ROC_all_in_one_word2vec_shade', dpi = 1200)
plt.show()
plt.gcf().clear()

# Statistical significance of model comprison
metric_val_LSTM = pd.read_csv('revision/results2/metric_val_LSTM.csv')
metric_val_DNN = pd.read_csv('revision/results2/metric_val_DNN.csv')
metric_val_RF = pd.read_csv('revision/results2/metric_val_RF.csv')
metric_val_LR = pd.read_csv('revision/results2/metric_val_LR.csv')
metric_val_NB = pd.read_csv('revision/results2/metric_val_NB.csv')

# Accuracy
acc_LSTM = metric_val_LSTM['Accuracy'][0:5]
acc_DNN = metric_val_DNN['Accuracy'][0:5]
acc_RF = metric_val_RF['Accuracy'][0:5]
acc_LR = metric_val_LR['Accuracy'][0:5]
acc_NB = metric_val_NB['Accuracy'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(acc_LSTM, acc_DNN)
t_LSR = stats.ttest_ind(acc_LSTM, acc_RF)
t_LSL = stats.ttest_ind(acc_LSTM, acc_LR)
t_LSN = stats.ttest_ind(acc_LSTM, acc_NB)

# Precision
prec_LSTM = metric_val_LSTM['Precision'][0:5]
prec_DNN = metric_val_DNN['Precision'][0:5]
prec_RF = metric_val_RF['Precision'][0:5]
prec_LR = metric_val_LR['Precision'][0:5]
prec_NB = metric_val_NB['Precision'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(prec_LSTM, prec_DNN)
t_LSR = stats.ttest_ind(prec_LSTM, prec_RF)
t_LSL = stats.ttest_ind(prec_LSTM, prec_LR)
t_LSN = stats.ttest_ind(prec_LSTM, prec_NB)

# Recall
rec_LSTM = metric_val_LSTM['Recall'][0:5]
rec_DNN = metric_val_DNN['Recall'][0:5]
rec_RF = metric_val_RF['Recall'][0:5]
rec_LR = metric_val_LR['Recall'][0:5]
rec_NB = metric_val_NB['Recall'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(rec_LSTM, rec_DNN)
t_LSR = stats.ttest_ind(rec_LSTM, rec_RF)
t_LSL = stats.ttest_ind(rec_LSTM, rec_LR)
t_LSN = stats.ttest_ind(rec_LSTM, rec_NB)

# f1
f1_LSTM = metric_val_LSTM['f1'][0:5]
f1_DNN = metric_val_DNN['f1'][0:5]
f1_RF = metric_val_RF['f1'][0:5]
f1_LR = metric_val_LR['f1'][0:5]
f1_NB = metric_val_NB['f1'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(f1_LSTM, f1_DNN)
t_LSR = stats.ttest_ind(f1_LSTM, f1_RF)
t_LSL = stats.ttest_ind(f1_LSTM, f1_LR)
t_LSN = stats.ttest_ind(f1_LSTM, f1_NB)

# Specificity
spec_LSTM = metric_val_LSTM['Specificity'][0:5]
spec_DNN = metric_val_DNN['Specificity'][0:5]
spec_RF = metric_val_RF['Specificity'][0:5]
spec_LR = metric_val_LR['Specificity'][0:5]
spec_NB = metric_val_NB['Specificity'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(spec_LSTM, spec_DNN)
t_LSR = stats.ttest_ind(spec_LSTM, spec_RF)
t_LSL = stats.ttest_ind(spec_LSTM, spec_LR)
t_LSN = stats.ttest_ind(spec_LSTM, spec_NB)

# AUC
auc_LSTM = metric_val_LSTM['AUC'][0:5]
auc_DNN = metric_val_DNN['AUC'][0:5]
auc_RF = metric_val_RF['AUC'][0:5]
auc_LR = metric_val_LR['AUC'][0:5]
auc_NB = metric_val_NB['AUC'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(auc_LSTM, auc_DNN)
t_LSR = stats.ttest_ind(auc_LSTM, auc_RF)
t_LSL = stats.ttest_ind(auc_LSTM, auc_LR)
t_LSN = stats.ttest_ind(auc_LSTM, auc_NB)

# For the case of no embedding
metric_val_LSTM = pd.read_csv('revision/results_noEmbed/metric_val_LSTM.csv')
metric_val_DNN = pd.read_csv('revision/results_noEmbed/metric_val_DNN.csv')
metric_val_RF = pd.read_csv('revision/results_noEmbed/metric_val_RF.csv')
metric_val_LR = pd.read_csv('revision/results_noEmbed/metric_val_LR.csv')
metric_val_NB = pd.read_csv('revision/results_noEmbed/metric_val_NB.csv')

# Accuracy
acc_LSTM = metric_val_LSTM['Accuracy'][0:5]
acc_DNN = metric_val_DNN['Accuracy'][0:5]
acc_RF = metric_val_RF['Accuracy'][0:5]
acc_LR = metric_val_LR['Accuracy'][0:5]
acc_NB = metric_val_NB['Accuracy'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(acc_LSTM, acc_DNN)
t_LSR = stats.ttest_ind(acc_LSTM, acc_RF)
t_LSL = stats.ttest_ind(acc_LSTM, acc_LR)
t_LSN = stats.ttest_ind(acc_LSTM, acc_NB)

# Precision
prec_LSTM = metric_val_LSTM['Precision'][0:5]
prec_DNN = metric_val_DNN['Precision'][0:5]
prec_RF = metric_val_RF['Precision'][0:5]
prec_LR = metric_val_LR['Precision'][0:5]
prec_NB = metric_val_NB['Precision'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(prec_LSTM, prec_DNN)
t_LSR = stats.ttest_ind(prec_LSTM, prec_RF)
t_LSL = stats.ttest_ind(prec_LSTM, prec_LR)
t_LSN = stats.ttest_ind(prec_LSTM, prec_NB)

# Recall
rec_LSTM = metric_val_LSTM['Recall'][0:5]
rec_DNN = metric_val_DNN['Recall'][0:5]
rec_RF = metric_val_RF['Recall'][0:5]
rec_LR = metric_val_LR['Recall'][0:5]
rec_NB = metric_val_NB['Recall'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(rec_LSTM, rec_DNN)
t_LSR = stats.ttest_ind(rec_LSTM, rec_RF)
t_LSL = stats.ttest_ind(rec_LSTM, rec_LR)
t_LSN = stats.ttest_ind(rec_LSTM, rec_NB)

# f1
f1_LSTM = metric_val_LSTM['f1'][0:5]
f1_DNN = metric_val_DNN['f1'][0:5]
f1_RF = metric_val_RF['f1'][0:5]
f1_LR = metric_val_LR['f1'][0:5]
f1_NB = metric_val_NB['f1'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(f1_LSTM, f1_DNN)
t_LSR = stats.ttest_ind(f1_LSTM, f1_RF)
t_LSL = stats.ttest_ind(f1_LSTM, f1_LR)
t_LSN = stats.ttest_ind(f1_LSTM, f1_NB)

# Specificity
spec_LSTM = metric_val_LSTM['Specificity'][0:5]
spec_DNN = metric_val_DNN['Specificity'][0:5]
spec_RF = metric_val_RF['Specificity'][0:5]
spec_LR = metric_val_LR['Specificity'][0:5]
spec_NB = metric_val_NB['Specificity'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(spec_LSTM, spec_DNN)
t_LSR = stats.ttest_ind(spec_LSTM, spec_RF)
t_LSL = stats.ttest_ind(spec_LSTM, spec_LR)
t_LSN = stats.ttest_ind(spec_LSTM, spec_NB)

# AUC
auc_LSTM = metric_val_LSTM['AUC'][0:5]
auc_DNN = metric_val_DNN['AUC'][0:5]
auc_RF = metric_val_RF['AUC'][0:5]
auc_LR = metric_val_LR['AUC'][0:5]
auc_NB = metric_val_NB['AUC'][0:5]

from scipy import stats
t_LSD = stats.ttest_ind(auc_LSTM, auc_DNN)
t_LSR = stats.ttest_ind(auc_LSTM, auc_RF)
t_LSL = stats.ttest_ind(auc_LSTM, auc_LR)
t_LSN = stats.ttest_ind(auc_LSTM, auc_NB)
#*********************************************************************************

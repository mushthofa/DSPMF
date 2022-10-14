from sklearn.externals import joblib
rf=joblib.load('modelobatwithsel_ok_imbalance.joblib')
import pandas as pd
data_ujisel_obat= pd.read_csv ('ch2_LB.csv')
data_ujisel_obat
select=data_ujisel_obat[['CELL_LINE','COMPOUND_A','COMPOUND_B','sinergy_cat']]
select
data_ujisel_obat.drop(labels = ['CELL_LINE','COMPOUND_A','COMPOUND_B','sinergy_cat'], axis = 1, inplace = True)
data_ujisel_obat
list(data_ujisel_obat)
X_uji=data_ujisel_obat[["data feature"]]
X_uji
y_uji_pred = rf.predict(X_uji)
predictions = y_uji_pred.round(0)
y_uji_pred
y_uji_act=select['sinergy_cat']
y_uji_act
y_uji_pred = pd.DataFrame(y_uji_pred)
y_uji_pred.rename(columns = {0:'sinergi_prediksi'}, inplace=True)
y_uji_pred
y_uji_act= pd.DataFrame(y_uji_act)
y_uji_act
y_actual_reindex=y_uji_act.reset_index()
y_pred_reindex = y_uji_pred.reset_index()
joint_1= pd.concat([y_actual_reindex, y_pred_reindex], axis=1)
joint_pred = joint_1.drop(['index'],axis=1)
joint_pred
probs = rf.predict_proba(X_uji)
probs_ = pd.DataFrame(probs)
probs_.head()
probs_.rename(columns = {0:'-1',1:'0', 2:'1'}, inplace=True)
probs_
probs_
from sklearn import metrics
cm = metrics.confusion_matrix(y_uji_act, y_uji_pred)
cm
cm_df = pd.DataFrame(cm, 
            columns = ['pred_minus1', 'pred_0','pred_1'],
            index = ['act_minus1', 'act_0', 'act_1'])
cm_df
print (metrics.accuracy_score(y_uji_act, y_uji_pred))
from sklearn.metrics import classification_report
print(classification_report(y_uji_act, y_uji_pred))
from sklearn import metrics
cm = metrics.confusion_matrix(y_uji_act, y_uji_pred)
cm
cm_df = pd.DataFrame(cm, 
            columns = ['pred_minus1', 'pred_0','pred_1'],
            index = ['act_minus1', 'act_0', 'act_1'])
cm_df
print (metrics.accuracy_score(y_uji_act, y_uji_pred))
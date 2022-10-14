import pandas as pd
fileastrazeneka= pd.read_csv ('ch1_train_combination_and_monoTherapy.csv')
fileastrazeneka
fileastrazeneka1=fileastrazeneka[['CELL_LINE','COMPOUND_A','COMPOUND_B','Einf_A','Einf_B']]
fileastrazeneka1
filesinergi = pd.read_csv ('efek_sinergi_score.csv')
filesinergi
filesinergi1=filesinergi[['CELL_LINE','COMPOUND_A','COMPOUND_B','sinergy_cat']]
filesinergi1
import pandas as pd
filemonoterapi = pd.read_csv ('gabung_monoterapi_fitur_A_dan_fitur_B.csv')
filemonoterapi
filemonoterapi.drop(labels = ['Einf_A','Einf_B'], axis = 1, inplace = True)
filemonoterapi
gabung_monoterapi_sinergi=pd.merge(filesinergi1, filemonoterapi, on=['CELL_LINE','COMPOUND_A','COMPOUND_B'])
gabung_monoterapi_sinergi
kombinasiobat=pd.merge(filedosen1, gabung_monoterapi_sinergi, on=['CELL_LINE','COMPOUND_A','COMPOUND_B'])
kombinasiobat
kombinasiobat.to_csv('gabung_monoterapi_sinergi_kombinasiobat.csv', index=False)

import pandas as pd
fitur_kombinasi_obat = pd.read_csv ('data_modeling_bersih.csv')
fitur_kombinasi_obat

import pandas as pd
fitur_sel_kanker = pd.read_csv ('mutasi_sel_gen_fix.csv')
fitur_sel_kanker

join_fitur_sel_obat = fitur_kombinasi_obat.join(fitur_sel_kanker.set_index('CELL_LINE'),on='CELL_LINE')
join_fitur_sel_obat.to_csv('join_fitur_sel_dan_fitur_obat.csv', index=False)

#Build Model Random Forest
import pandas as pd
modelobat = pd.read_csv ('join_fitur_sel_dan_fitur_obat.csv')
#first check data imbalance
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(modelobat, test_size=0.3, random_state=2)
data_train.sinergy_cat.value_counts()
modelobata = data_train[data_train.sinergy_cat==-1]
modelobatb = data_train[data_train.sinergy_cat==0]
modelobatc = data_train[data_train.sinergy_cat==1]
from sklearn.utils import resample
df_minority_upsampleda = resample(modelobata, 
                                 replace=True,     # sample with replacement
                                 n_samples=894,    # to match majority class
                                 random_state=123) # reproducible results
df_minority_upsampledb = resample(modelobatb, 
                                 replace=True,     # sample with replacement
                                 n_samples=894,    # to match majority class
                                 random_state=123) # reproducible results
                                 df_upsampled = pd.concat([modelobatc, df_minority_upsampleda,df_minority_upsampledb])
df_upsampled.sinergy_cat.value_counts()
#train features and target
from sklearn.ensemble import RandomForestClassifier
n=15
rf = RandomForestClassifier(n_estimators=n, max_depth=30, n_jobs=15, warm_start=True)
rf.fit(X_train,y_train)
from sklearn import metrics
prediction_test = rf.predict(X_test)
# Print the test accuracy
print (metrics.accuracy_score(y_test, prediction_test))
#pohon 30 ranting 15
from sklearn import metrics
prediction_test = rf.predict(X_test)
# Print the test accuracy
print (metrics.accuracy_score(y_test, prediction_test))
from sklearn import metrics
prediction_test = rf.predict(X_test)
# Print the test accuracy
print (metrics.accuracy_score(y_test, prediction_test))
prediction_train = rf.predict(X_train)
# Print the train accuracy
print (metrics.accuracy_score(y_train, prediction_train))
from sklearn.externals import joblib
joblib.dump(rf, 'modelobatwithsel_ok_imbalance.joblib')
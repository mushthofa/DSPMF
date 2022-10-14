import pandas as pd
monoterapi_obatA_obatB = pd.read_csv ('sel_obatA_obatB.csv')
monoterapi_obatA_obatB.drop(labels = ['CELL_LINE','COMPOUND','ig'], axis = 1, inplace = True)
monoterapi_obatA_obatB
monoterapi_obatA_obatB.drop(labels = ['CELL_LINE','COMPOUND','ig'], axis = 1, inplace = True)
monoterapi_obatA_obatB
fiturAdanB=monoterapi_obatA_obatB
fiturAdanB
#Cleansing Data 
#Feature selection using LASSO method
X_train, X_test, y_train, y_test = train_test_split(
    fiturAdanB.drop(labels=['Einf'], axis=1), fiturAdanB['Einf'],
    test_size=0.3,
    random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )
                      search.fit(X_train,y_train)
                      search.best_params_
                      coefficients = search.best_estimator_.named_steps['model'].coef_
coefficients
importance = np.abs(coefficients)
importance
np.array(features_AB)[importance > 0]
fiturAdanBfix
fiturAdanBfix.to_csv('monoterapi_fitur_Ab.csv', index=False)
#Combine data on monotherapy features that have been selected by LASSO with drug combination data 
import pandas as pd
filecombination = pd.read_csv ('ch1_train_combination_and_monoTherapy.csv')
filecombination
jointalldataset=pd.merge(filecombination, fiturAdanBfix, on=['CELL_LINE','COMPOUND_A','COMPOUND_B'])
jointalldataset
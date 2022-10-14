import pandas as pd
efek_sinergi = pd.read_csv ('efeksinergi.csv')
efek_sinergi
efek_sinergi['sinergy_cat'] = ['1' if x>=1 else '0' if -1<x<1 else '-1' for x in efek_sinergi['SYNERGY_SCORE']]
efek_sinergi
efek_sinergi.to_csv('efek_sinergi_score.csv', index=False)
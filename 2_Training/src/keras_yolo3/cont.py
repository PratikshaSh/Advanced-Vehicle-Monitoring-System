import pandas as pd 


df = pd.read_csv('result_recog.txt',sep=',', encoding= 'unicode_escape',header=None)
df.columns = ['Image','OCR']

df.to_csv('recognition.csv')
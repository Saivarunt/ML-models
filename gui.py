import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('kyphosis.csv')
print(df.head())
a=pd.Series(df['Kyphosis'])
b=np.empty(len(a))
c=0
for i in a:
  if a[c]=='absent':
    b[c]=0
  elif a[c]=='present':
    b[c]=1
  else:
    pass
  c=c+1
print(b)
b=pd.Series(b,name='result')
df=pd.merge(b,df,left_index=True,right_index=True)
print(df)
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

import tkinter as tk
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()

label1 = tk.Label(root, text='Enter age:')
canvas1.create_window(100, 100, window=label1)
entry1 =tk.Entry (root)
canvas1.create_window(270, 100, window=entry1)

label2 = tk.Label(root, text='Enter start:')
canvas1.create_window(120, 120, window=label2)
entry2 =tk.Entry (root)
canvas1.create_window(270, 120, window=entry2)

label3 = tk.Label(root, text='Enter end:')
canvas1.create_window(140, 140, window=label3)
entry3 =tk.Entry (root)
canvas1.create_window(270, 140, window=entry3)

label4 = tk.Label(root, text='Enter result:')
canvas1.create_window(160, 160, window=label4)
entry4 =tk.Entry (root)
canvas1.create_window(270, 160, window=entry4)

def values():
    global nage 
    nage= float(entry1.get())

    global nstart
    nstart= float(entry2.get())
    
    global nend
    nend= float(entry3.get())
    
    global nresult
    nresult= float(entry4.get())


    Prediction_result = ('Predict:', dtree.predict([[nresult,nage,nstart,nend]]))
    label_Prediction = tk.Label(root, text=Prediction_result, bg='blue')
    canvas1.create_window(300, 280, window=label_Prediction)


button1 = tk.Button(root, text='Predict', command=values,bg='orange')
canvas1.create_window(270, 190, window=button1)
root.mainloop()
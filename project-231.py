import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

dataset=pd.read_csv('books.csv',error_bad_lines=False)

x=dataset.iloc[:,4:11].values

y=dataset.iloc[:,3].values

print("Value of input layer : ",x)

print("Value of output layer : ",y)

model=Sequential()

model.add(Dense(128,input_dim=8,activation='relu'))

model.add(Dense(114,activation='relu'))

model.add(Dense(53,activation='sigmoid'))

model.add(Dense(1,activation='sigmoid'))

model.summary()
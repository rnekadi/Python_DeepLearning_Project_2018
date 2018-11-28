import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data

train = pd.read_csv("/Users/sai/Documents/GitHub/DNN2018/data/train.csv")

page = train['Page']


#Dropping Page Column

#train = train.drop('Page', axis = 1, inplace=True)


train.fillna(0, inplace=True)
row = train.iloc[90000, :].values
X = row[1:549]

y = row[2:550]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = np.reshape(X_train, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))
X_train = sc.fit_transform(X_train)
y_train = sc.fit_transform(y_train)
#

# Training LSTM

#Reshaping Array
X_train = np.reshape(X_train, (383, 1, 1))


# Importing the Keras libraries and packages for LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#
# Initialising the RNN
lstm_model = Sequential()
#
# Adding the input layerand the LSTM layer
lstm_model.add(LSTM(units = 100, activation = 'relu', input_shape = (None, 1)))

# Adding the output layer
lstm_model.add(Dense(units = 1))
#
# Compiling the RNN
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# Fitting the LSTM to the Training set
lstm_model.fit(X_train, y_train, batch_size = 20, epochs = 400, verbose = 2)




# Getting the predicted Web View
inputs = X_test
inputs = np.reshape(inputs,(-1,1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (165, 1, 1))
y_pred = lstm_model.predict(inputs)
y_pred = sc.inverse_transform(y_pred)

# Visualising Result

plt.figure
plt.plot(y_test, color = 'red', label = 'Real Web View')
plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
plt.title('Web View Forecasting')
plt.xlabel('Number of Days from Start')
plt.ylabel('Web View')
plt.legend()



# Evaluating and Accuracy and Score

train_score = lstm_model.evaluate(X_train, y_train, verbose=2)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))

score, accuracy = lstm_model.evaluate(X_train, y_train, verbose=2)

print('Score: %.2f' %(score))
print('Validation Accuracy: %.2f' % (accuracy))


plt.show()
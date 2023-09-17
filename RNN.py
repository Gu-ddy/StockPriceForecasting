from tensorflow import concat
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,GlobalMaxPooling1D, LSTM
from tensorflow.keras.layers import Dense, Flatten, Dropout, Concatenate, AveragePooling1D
from tensorflow.keras.layers import InputLayer, Input, ActivityRegularization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns
import numpy as np
from preprocessing import data_to_supervised



def RNN(dim, activation='elu' ,plot=False):
  keras.utils.set_random_seed(7)
  '''
  @dim=(n_timestamps,dati_X.shape, n_outputs)
  '''
  input1 = Input(shape = (dim[0],dim[1]))

  # Layer LSTM sull'intero mese
  z0 = LSTM(64,kernel_regularizer=regularizers.l2(0.001))(input1)

  # Layer LSTM sulle medie settimanali
  z1 = AveragePooling1D(pool_size=5, strides=5, padding='same')(input1)
  z1 = LSTM(32,kernel_regularizer=regularizers.l2(0.001))(z1)

  # Layer LSTM sulle medie bisettimanali
  z2 = AveragePooling1D(pool_size=10, strides=5, padding='same')(input1)
  z2 = LSTM(32,kernel_regularizer=regularizers.l2(0.001))(z2)

  # Layer LSTM sulle medie trisettimanali
  z3 = AveragePooling1D(pool_size=20, strides=5, padding='same')(input1)
  z3 = LSTM(32,kernel_regularizer=regularizers.l2(0.001))(z3)

  # Valori medi delle variabili
  z5 = GlobalAveragePooling1D()(input1)
  z5 = Dense(16,activation=activation)(z5)

  #Valori massimi delle variabili
  z6 = GlobalMaxPooling1D()(input1)
  z6 = Dense(16,activation=activation)(z6)

  x = Concatenate()([z0,z1,z2,z3,z5,z6])
  x = Dense(1024, activation = activation)(x)
  x = Dropout(0.2,seed=12345)(x)

  output = Dense(dim[2], activation = "linear")(x)

  model = Model(inputs=[input1], outputs=output)

  if plot:
      plot_model(model,to_file="Images/Model.png", show_shapes=True, show_layer_names=True)

  return model




def diff_to_value(diff_pred,inizio,dati_completi):
  pred=np.zeros(shape=diff_pred.shape)
  for i in range(diff_pred.shape[0]):
    origine=dati_completi['CHIUSURA'][inizio]
    inizio+=1
    for j in range(diff_pred.shape[1]):
      pred[i,j]=origine+diff_pred[i,j]
      origine=pred[i,j]
  return pred


def train_evaluation_loop(dati_X,dati_y,dati_completi,n_timestamps,n_outputs, N_train, not_diff_data=None,plot=True,  stand=True, epochs=10, verbose=0, ret_prev=False):
    keras.utils.set_random_seed(7)
    X,y=data_to_supervised(dati_X,dati_y,n_timestamps=n_timestamps,n_outputs=n_outputs, N_train=N_train, stand=stand)
    X_train,y_train=X[:N_train-n_timestamps-n_outputs+1,:,:],y[:N_train-n_timestamps-n_outputs+1,:]
    X_test= X[N_train-n_timestamps:,:,:]
    if not_diff_data is not None:
        _,y=data_to_supervised(dati_X,not_diff_data,n_timestamps=n_timestamps,n_outputs=n_outputs, N_train=N_train)
    y_test=y[N_train-n_timestamps:,:]
    dim=(n_timestamps,dati_X.shape[1],n_outputs)
    model=RNN(dim)
    model.compile(optimizer = keras.optimizers.Adam() , loss = "mse", metrics=["mae"])
    print('Start Training')
    model.fit(X_train,y_train,epochs=epochs,batch_size=256,verbose=verbose)
    print('Training Completed')
    y_hat_rnn=model.predict(X_test, verbose=verbose)
    if not_diff_data is not None:
        y_hat_rnn=diff_to_value(y_hat_rnn,inizio=N_train,dati_completi=dati_completi)
    mae_rnn=mae(y_hat_rnn,y_test)
    print('MAE on test set: {}'.format(mae_rnn))

    if plot:
        y_prev_rnn=[]
        i=0
        while i<y_hat_rnn.shape[0]:
            if n_outputs>1:p=y_hat_rnn[i,:]
            else: p=y_hat_rnn[i]
            i+=n_outputs
            for z in p:
                y_prev_rnn.append(z)
        y=dati_completi.CHIUSURA
        x=dati_completi.index
        #x_prev=dati_completi.index[N_train:-(dati_completi.shape[0]-N_train-len(y_prev_rnn))]
        x_prev=dati_completi.index[N_train:N_train+len(y_prev_rnn)]
        p=sns.lineplot(x=x[1600:],y=y[1600:],color='red',lw=2,label='true')
        p=sns.lineplot(x=x_prev,y=y_prev_rnn,color='yellow',lw=2,label='LSTM')
    if ret_prev: return mae_rnn,y_prev_rnn
    return mae_rnn


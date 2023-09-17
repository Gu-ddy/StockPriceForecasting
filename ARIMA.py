import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from preprocessing import data_to_supervised
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns
import matplotlib.pyplot as plt

def previsione_arima(serie,inizio,n_outputs,order,seasonal_order,param):
  '''Utilizza un modello ARIMA di ordine pari a order e con coefficienti pari a params per predirre gli n_output
  valori successivi all'inizio-esimo della serie.
  I valori della serie fino all'inizio-esimo sono usati come input ma non per stimare i coefficienti, che sono fissati
  come argomento della funzione insieme all'ordine del modello.
   '''
  mod=ARIMA(serie[:inizio],order=order,seasonal_order=seasonal_order) # diamo l'ordine e facciamo conoscere al modello la serie fino ai primi inzio valori.
  f1=mod.fit_constrained(param) #fissiamo i coefficienti
  return f1.forecast(n_outputs)

def previsioni_arima(serie,N_train,n_outputs,order,seasonal_order,param):
  inizio = N_train-1
  '''Utilizza previsione_arima per predirre tutti i possibili blocchi di n_output predizioni che si possono creare da serie partendo
  dalla conoscenza di serie[:inizio] e incrementando di un giorno la conoscenza della serie per il blocco di predizione successivo.
 Esempio:
 serie=[90,85,82,87,91,75,98,100,111], inizio= 2,order=(2,0,0), params= [0.6,0.4], n_output=3

 si usa un modello ARIMA (2,0,0) con coefficienti= [0.6,0.4]
 1)Al modello si fa conoscere la serie [90,85] e lo si usa per predirre i 3 giorni successivi (il cui valore vero sarebbe 82,87,91)
 2)Al modello si fa conoscere la serie [90,85,82] e lo si usa per predirre i 3 giorni successivi (il cui valore vero sarebbe 87,91,75)
 3)Al modello si fa conoscere la serie [90,85,82,87] e lo si usa per predirre i 3 giorni successivi (il cui valore vero sarebbe 91,75,98)
 e così via...

  '''
  prev=np.zeros(shape=(len(serie)-n_outputs-N_train+1,n_outputs))
  for i in range(len(serie)-n_outputs-N_train+1):
    p=previsione_arima(serie,inizio+i,n_outputs,order, seasonal_order,param)
    prev[i,:]=p
  return prev



def identication(y_train_arima):
    #eventually here save plots

    print('Start Identification')
    model_autoARIMA = auto_arima(y_train_arima,trace=False,step_wise=False, seasonal=True, m=12)
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.savefig('Images/Diagnostics.png')
    order,seasonal_order=model_autoARIMA.order,model_autoARIMA.seasonal_order
    print('Identification completed')
    return [order,seasonal_order]

def param_estimation(y_train_arima,order,seasonal_order):
    arima_model=ARIMA(y_train_arima,order=order, seasonal_order=seasonal_order)
    param_value=arima_model.fit(return_params=True)#
    param_key=arima_model.param_names
    param={} #In questo dizionario conterremo i coefficienti stimati
    for i in range(len(param_key)):
        param[param_key[i]]=param_value[i]
    return param


def train_evaluation_loop_arima(chiusure,chiusure_pd,dati_completi,N_train, n_outputs,plot=True, ret_prev=False):
    y_train_arima=chiusure[:N_train] 
    order,seasonal_order= identication(y_train_arima)
    param=param_estimation(y_train_arima,order,seasonal_order)
    print('Start test')
    y_hat_arima=previsioni_arima(chiusure,N_train,n_outputs,order, seasonal_order,param)
    print('Test completed')

    #extracting test values
    chiusure_pd=dati_completi.iloc[:,[3]]
    X,y=data_to_supervised(chiusure_pd,chiusure_pd, n_timestamps=0,n_outputs=n_outputs,N_train=N_train)
    y_true=y[N_train:]

    #computing MAE
    mae_arima=mae(y_true,y_hat_arima)
    print('MAE on test set: {}'.format(mae_arima))
    
    #plotting
    if plot:
        y_prev_arima=[]
        i=0
        while i<y_hat_arima.shape[0]:
            p=y_hat_arima[i,:]
            i+=n_outputs
            for z in p:
                y_prev_arima.append(z)
        y=dati_completi.CHIUSURA
        x=dati_completi.index
        #x_prev=dati_completi.index[N_train:-(dati_completi.shape[0]-N_train-len(y_prev_arima))]
        x_prev=dati_completi.index[N_train:N_train+len(y_prev_arima)]
        p=sns.lineplot(x=x[1600:],y=y[1600:],color='red',lw=2,label='true')
        p=sns.lineplot(x=x_prev,y=y_prev_arima,color='orange',lw=2,label='ARIMA')
    if ret_prev: return mae_arima, y_prev_arima
    return mae_arima

import pandas as pd
import numpy as np




def previous(day,serie):
# needed to fill missing dates for DAX and S&P with most recent value
  while 1:
    day=day-pd.Timedelta(1,'day')
    if day in serie.index:
      return serie[day]


def replace(s,x,y):
  return s.replace(x,y)


def data_to_supervised (dati_X,dati_y,n_timestamps,n_outputs,N_train, stand=True):
  '''

  data_to_supervised is used to extract input and target values. N_timestamps determines the maximum lag used as input 
  while n_output gives the horizon of the prediction.


  Restituisce due array: X e y tramite i quali si può allenare una rete neurale ricorrente.
  L'i-esimo elemento di X sarà un array di forma (n_timestamp, numero di predittori). L'i-esimo elemento di y conterrà il corrispondente
  valore della variabile target, ossia un array di n_output valori della variabile da voler predirre nel tempo. In altre parole X conterrà
  gli array di dati che si vuole utilizzare per predirre gli omologhi elementi di y.

  Notiamo che se si vogliono usare n_timestamp giorni precedenti per predirne n_output, la lunghezza di X e y sarà data da:
  Numero di osservazioni in dati_X (o dati_y) - n_timestamps - n_outputs +1

  UN ESEMPIO ILLUSTRATIVO DELLA FUNZIONE SI PUO TROVARE COME PRIMO PARAGRAFO DELLA SEZIONE RNN MODEL


  Dati_X è il dataframe contenete le variabili da usare come predittori (tra cui sarà sicuramente presente la chiusura del FTSEMIB)
  Dati_y conterrà la variabile target (nel nostro caso le chiusure del FTSEMIB)
  A partire da dati_X e da dati_y si creano l'array dei predittori X e l'array della variabile target y.
  n_output definisce l'orizzonte della predizione, quanti istanti futuri si vuole predirre.
  n_timestamps definisce il numero di istanti precedenti al primo da voler predirre che si useranno come input.
  Esempio:
  Dati_X=Chiusure del FTSEMIB, Dati_y=Chiusure del FTSEMIB, n_output=2, n_timestamps=3.
  Useremo i valori delle chiusure del FTSEMIB nei giorni t-1, t-2 e t-3 per predirre i valori delle chiusure del FTSEMIB dei giorni t,t+1.
  '''

  n_samples=dati_X.shape[0]-n_timestamps-n_outputs+1
  n_features=dati_X.shape[1]

  # Standardizing
  if stand:
    mu=dati_X[:N_train].mean()
    sd=dati_X[:N_train].std()
    dati_X=(dati_X-mu)/sd

  X=np.zeros((n_samples,n_timestamps,n_features))
  for s in range(n_samples):
    for t in range(n_timestamps):
      for f in range(n_features):
        X[s,t,f]=dati_X.iloc[s+t,f]
  y=np.zeros((n_samples,n_outputs))
  for s in range(n_samples):
    for o in range(n_outputs):
      y[s,o]=dati_y.iloc[s+n_timestamps+o]
  return X,y



def diff_to_value(diff_pred,inizio):
  # inizio should be N_train
  ''' transform predictions on the differentiated data to non differentiated values'''
  pred=np.zeros(shape=diff_pred.shape)
  for i in range(diff_pred.shape[0]):
    origine=dati_completi['CHIUSURA'][inizio-1]
    inizio+=1
    for j in range(diff_pred.shape[1]):
      pred[i,j]=origine+diff_pred[i,j]
      origine=pred[i,j]
  return pred
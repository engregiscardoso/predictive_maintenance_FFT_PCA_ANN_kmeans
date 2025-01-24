## LIBS IMPORT
#%%
from scipy.io import loadmat
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
from sklearn.metrics import confusion_matrix
import serial
import time
import os

import scipy.io

import math #For identifying NaNs(Not a Number)
import pickle #For compressing files
import glob #For dealing with file paths and directories
import os #For dealing with file paths and directories
import tqdm #For making progress bars
import datetime #For dealing with time in general

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
import scipy.stats as stats
from scipy.signal import filtfilt
from scipy.fft import fft, fftfreq
from pylab import rcParams
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
import math
from collections import Counter
#%%


## READ DATA
#%%
### ORGANIZE THE DATASET WITHOUT THE TIME COLUMN

def ajust_df_sem_time(df_base):
  value = []
  
  df_base = df_base.reset_index()
  del df_base['index']
  
  df_lines = len(df_base)
  
  for i in range(df_lines):
        value.append(df_base.loc[i])
        
  df_ts = []
  df_ts = pd.DataFrame()

  df_ts['value'] = pd.concat(value)
  df_ts = df_ts.reset_index()
  
  

  return df_ts['value']


### CONCATENATES THE DATA BY NAMING THE COLUMNS MAINTAINING THE SAME PATTERN

def ajust_df_x_concat(df_base, numero_linhas):
    # Inicializar DataFrame vazio
    df_concat_100_linhas_0 = pd.DataFrame()
    
    # Calcular número de blocos possíveis
    n_blocos = len(df_base) // numero_linhas
    
    # Loop para criar colunas com blocos
    for i in range(n_blocos):
        # Nome da coluna
        coluna = f'Dado_x15_{i}_0'
        
        # Definir os índices do bloco
        start_idx = i * numero_linhas
        end_idx = (i + 1) * numero_linhas  # Ajuste para índices inclusivos no slicing
        
        # Fatiar e ajustar os dados
        bloco = df_base.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Verificar se o bloco tem dados e ajustar
        if not bloco.empty:
            # Ajustar o bloco e adicioná-lo ao DataFrame final
            df_concat_100_linhas_0[coluna] = ajust_df_sem_time(bloco)
        else:
            print(f"Bloco vazio em {coluna}, ignorado.")
    
    return df_concat_100_linhas_0


### CREATE THE TIME COLUMN IN THE DATASET

def creat_time_colun(df_base):
    df = df_base
    df['time'] = 0
    for i in range(len(df)):
      df['time'].loc[i] = i*(1/10000)
      
    return df['time']


### FUNCTION TO PERFORM THE FFT OF THE SIGNAL BASED ON X AND Y

def FFT_x_y(graf_y, graf_x):

    x = graf_x
    y = graf_y
    N = len(graf_x)

    T = x[1] - x[0]
    Fs = 1 / T
    
    yf = 2.0 / N * np.abs(fft(y)[0:N // 2])

    xf = fftfreq(N, T)[:N // 2]

    verX = []
    verY = []

    obs = len(yf)

    for i in range(1, obs, 1):
        verX.insert(i, xf[i])
        verY.insert(i, yf[i])
        
    df_FFT = []
    df_FFT = pd.DataFrame(df_FFT)

    df_FFT['F'] = verX
    df_FFT['A'] = verY

    return (df_FFT)


### PREPARE THE DATASET USING FREQUENCY DOMAIN BY FFT

def prepare_transpost_df_x(df_base, fault_code):
    
    df_FFT = []
    df_FFT = pd.DataFrame(df_FFT)
    
    cols = df_base.shape[1]
    
    for i in range(cols-1):
        coluna = 'Dado_x15_'+str(i)+'_0'
        df_concat_parcial = FFT_x_y(df_base[coluna].values, df_base['time'].values)
        
        coluna = 'Dado_x15_'+str(i)+'_'+ str(fault_code)
        
        df_FFT[coluna] = df_concat_parcial['A']
      
    return df_FFT


### READING, ORGANIZING AND RETURN OF DATASETs

def read_and_separe_dataset():
    
    # every 200 values is equivalent to 20ms
    path = ''
    #Time domain vibration signals under operational condition of 1500-rpm motor speed and 10Nm load
    x_1500_10_file = path + 'x_1500_10.npy' #x direction
    y_1500_10_file = path + 'y_1500_10.npy' #y direction
    z_1500_10_file = path + 'z_1500_10.npy' #z direction
    
    #ground-truth labels for four different types of faults
    gt_1500_10_file = path + 'gt_1500_10.npy'
    
    #load files 
    x_1500_10 = np.load(x_1500_10_file)
    y_1500_10 = np.load(y_1500_10_file)
    z_1500_10 = np.load(z_1500_10_file)
    gt_1500_10 = np.load(gt_1500_10_file)
    
    #Convert numpy datasets into pandas datasets
    dfx15 = pd.DataFrame(x_1500_10)
    dfy15 = pd.DataFrame(y_1500_10)
    dfz15 = pd.DataFrame(z_1500_10)
    dfgt15 = pd.DataFrame(gt_1500_10)
    
    dfx15['fault_code'] = dfgt15[0]
    dfy15['fault_code'] = dfgt15[0]
    dfz15['fault_code'] = dfgt15[0]
    
    dfx15_0 = dfx15[dfx15['fault_code']==0] # Fault Code ==0
    dfx15_1 = dfx15[dfx15['fault_code']==1] # Fault Code ==1
    dfx15_2 = dfx15[dfx15['fault_code']==2] # Fault Code ==2
    dfx15_3 = dfx15[dfx15['fault_code']==3] # Fault Code ==3
    dfx15_4 = dfx15[dfx15['fault_code']==4] # Fault Code ==4
    dfy15_0 = dfy15[dfy15['fault_code']==0] # Fault Code ==0
    dfy15_1 = dfy15[dfy15['fault_code']==1] # Fault Code ==1
    dfy15_2 = dfy15[dfy15['fault_code']==2] # Fault Code ==2
    dfy15_3 = dfy15[dfy15['fault_code']==3] # Fault Code ==3
    dfy15_4 = dfy15[dfy15['fault_code']==4] # Fault Code ==4
    dfz15_0 = dfz15[dfz15['fault_code']==0] # Fault Code ==0
    dfz15_1 = dfz15[dfz15['fault_code']==1] # Fault Code ==1
    dfz15_2 = dfz15[dfz15['fault_code']==2] # Fault Code ==2
    dfz15_3 = dfz15[dfz15['fault_code']==3] # Fault Code ==3
    dfz15_4 = dfz15[dfz15['fault_code']==4] # Fault Code ==4
    
    del dfx15_0['fault_code']
    del dfx15_1['fault_code']
    del dfx15_2['fault_code']
    del dfx15_3['fault_code']
    del dfx15_4['fault_code']
    del dfy15_0['fault_code']
    del dfy15_1['fault_code']
    del dfy15_2['fault_code']
    del dfy15_3['fault_code']
    del dfy15_4['fault_code']
    del dfz15_0['fault_code']
    del dfz15_1['fault_code']
    del dfz15_2['fault_code']
    del dfz15_3['fault_code']
    del dfz15_4['fault_code']
    
    dfx15_0 = dfx15_0.reset_index()
    dfx15_0 = dfx15_0.drop(columns=['index'])
    dfx15_1 = dfx15_1.reset_index()
    dfx15_1 = dfx15_1.drop(columns=['index'])
    dfx15_2 = dfx15_2.reset_index()
    dfx15_2 = dfx15_2.drop(columns=['index'])
    dfx15_3 = dfx15_3.reset_index()
    dfx15_3 = dfx15_3.drop(columns=['index'])
    dfx15_4 = dfx15_4.reset_index()
    dfx15_4 = dfx15_4.drop(columns=['index'])
    
    dfy15_0 = dfy15_0.reset_index()
    dfy15_0 = dfy15_0.drop(columns=['index'])
    dfy15_1 = dfy15_1.reset_index()
    dfy15_1 = dfy15_1.drop(columns=['index'])
    dfy15_2 = dfy15_2.reset_index()
    dfy15_2 = dfy15_2.drop(columns=['index'])
    dfy15_3 = dfy15_3.reset_index()
    dfy15_3 = dfy15_3.drop(columns=['index'])
    dfy15_4 = dfy15_4.reset_index()
    dfy15_4 = dfy15_4.drop(columns=['index'])
    
    dfz15_0 = dfz15_0.reset_index()
    dfz15_0 = dfz15_0.drop(columns=['index'])
    dfz15_1 = dfz15_1.reset_index()
    dfz15_1 = dfz15_1.drop(columns=['index'])
    dfz15_2 = dfz15_2.reset_index()
    dfz15_2 = dfz15_2.drop(columns=['index'])
    dfz15_3 = dfz15_3.reset_index()
    dfz15_3 = dfz15_3.drop(columns=['index'])
    dfz15_4 = dfz15_4.reset_index()
    dfz15_4 = dfz15_4.drop(columns=['index'])
    
    return dfx15_0, dfx15_1, dfx15_2, dfx15_3, dfx15_4, dfy15_0, dfy15_1, dfy15_2, dfy15_3, dfy15_4, dfz15_0, dfz15_1, dfz15_2, dfz15_3, dfz15_4


####################################################################################

### DATASET DESCRIPTION
#ICPHM2023 Data Challenge Dataset
#The dataset includes vibration signals of normal and four different types of fault conditions and their corresponding ground-truth labels for two different operational conditions.
#• Vibration signals have been divided into smaller segments using window size of 200 data points to reduce the computational time.
#• Vibration signals are available in three directions (x, y, z). Participants are free to use either one direction or any combination of them.
#Each vibration signal is recorded in three directions (x, y, z), for a period of five minutes, and with a sampling frequency of 10 kHz. The examples of time-domain signal acquired in
#different operating conditions.
#sample rate 10 kHz= 10.000 samples per second.


####################################################################################

## READING DOOS DATASETs
dfx15_0, dfx15_1, dfx15_2, dfx15_3, dfx15_4, dfy15_0, dfy15_1, dfy15_2, dfy15_3, dfy15_4, dfz15_0, dfz15_1, dfz15_2, dfz15_3, dfz15_4 = read_and_separe_dataset()

plt.plot(dfx15_0[10])

### ORGANIZES AND JOINS DATA BASED ON THE NUMBER OF ROWS DESIRED FOR ANALYSIS
### (1OO LINES IS A GOOD VALUE) WHICH RESULTS IN AN ACQUISITION OF 2 SECONDS OF DATA

### Use ajust_df_x_concat for the X axis so that the column names all have the same for concatenation

### To print the complete dataset use number_lines = 10000

### To print the training and testing dataset number_lines = 100

numero_linhas = 100

df_x_concat_100_linhas_0 = ajust_df_x_concat(dfx15_0, numero_linhas)
df_x_concat_100_linhas_1 = ajust_df_x_concat(dfx15_1, numero_linhas)
df_x_concat_100_linhas_2 = ajust_df_x_concat(dfx15_2, numero_linhas)
df_x_concat_100_linhas_3 = ajust_df_x_concat(dfx15_3, numero_linhas)
df_x_concat_100_linhas_4 = ajust_df_x_concat(dfx15_4, numero_linhas)

df_y_concat_100_linhas_0 = ajust_df_x_concat(dfy15_0, numero_linhas)
df_y_concat_100_linhas_1 = ajust_df_x_concat(dfy15_1, numero_linhas)
df_y_concat_100_linhas_2 = ajust_df_x_concat(dfy15_2, numero_linhas)
df_y_concat_100_linhas_3 = ajust_df_x_concat(dfy15_3, numero_linhas)
df_y_concat_100_linhas_4 = ajust_df_x_concat(dfy15_4, numero_linhas)

df_z_concat_100_linhas_0 = ajust_df_x_concat(dfz15_0, numero_linhas)
df_z_concat_100_linhas_1 = ajust_df_x_concat(dfz15_1, numero_linhas)
df_z_concat_100_linhas_2 = ajust_df_x_concat(dfz15_2, numero_linhas)
df_z_concat_100_linhas_3 = ajust_df_x_concat(dfz15_3, numero_linhas)
df_z_concat_100_linhas_4 = ajust_df_x_concat(dfz15_4, numero_linhas)

plt.plot(df_x_concat_100_linhas_0.loc[0])

####################################################################################

### ADD THE TIME COLUMN TO THE DATASET, CREATE A DATASET WHERE COLUMNS ARE THE DATA AND THE LINES ARE THE TIME OF ACQUISITION OF EACH SIGNAL

df_x_concat_100_linhas_0['time'] = creat_time_colun(df_x_concat_100_linhas_0)
df_x_concat_100_linhas_1['time'] = creat_time_colun(df_x_concat_100_linhas_1)
df_x_concat_100_linhas_2['time'] = creat_time_colun(df_x_concat_100_linhas_2)
df_x_concat_100_linhas_3['time'] = creat_time_colun(df_x_concat_100_linhas_3)
df_x_concat_100_linhas_4['time'] = creat_time_colun(df_x_concat_100_linhas_4)

df_y_concat_100_linhas_0['time'] = creat_time_colun(df_y_concat_100_linhas_0)
df_y_concat_100_linhas_1['time'] = creat_time_colun(df_y_concat_100_linhas_1)
df_y_concat_100_linhas_2['time'] = creat_time_colun(df_y_concat_100_linhas_2)
df_y_concat_100_linhas_3['time'] = creat_time_colun(df_y_concat_100_linhas_3)
df_y_concat_100_linhas_4['time'] = creat_time_colun(df_y_concat_100_linhas_4)

df_z_concat_100_linhas_0['time'] = creat_time_colun(df_z_concat_100_linhas_0)
df_z_concat_100_linhas_1['time'] = creat_time_colun(df_z_concat_100_linhas_1)
df_z_concat_100_linhas_2['time'] = creat_time_colun(df_z_concat_100_linhas_2)
df_z_concat_100_linhas_3['time'] = creat_time_colun(df_z_concat_100_linhas_3)
df_z_concat_100_linhas_4['time'] = creat_time_colun(df_z_concat_100_linhas_4)



figx, axs = plt.subplots(5,1, figsize=(40,20))
axs[0].plot(df_x_concat_100_linhas_0['time'], df_x_concat_100_linhas_0['Dado_x15_0_0'])
#axs[0].set_xlim([0, 2])
axs[0].set_ylim([-0.5, 0.5])

axs[1].plot(df_x_concat_100_linhas_1['time'], df_x_concat_100_linhas_1['Dado_x15_0_0'])
#axs[1].set_xlim([0, 2])
axs[1].set_ylim([-0.5, 0.5])

axs[2].plot(df_x_concat_100_linhas_2['time'], df_x_concat_100_linhas_2['Dado_x15_0_0'])
#axs[2].set_xlim([0, 2])
axs[2].set_ylim([-0.5, 0.5])

axs[3].plot(df_x_concat_100_linhas_3['time'], df_x_concat_100_linhas_3['Dado_x15_0_0'])
#axs[3].set_xlim([0, 2])
axs[3].set_ylim([-0.5, 0.5])

axs[4].plot(df_x_concat_100_linhas_4['time'], df_x_concat_100_linhas_4['Dado_x15_0_0'])
#axs[4].set_xlim([0, 2])
axs[4].set_ylim([-0.5, 0.5])

axs[0].set(title='Original Dataset')
axs[0].set(ylabel='Label 0 \n\n')
axs[1].set(ylabel='Label 1 \n\n')
axs[2].set(ylabel='Label 2 \n\nValue')
axs[3].set(ylabel='Label 3 \n\n')
axs[4].set(xlabel='Time')
axs[4].set(ylabel='Label 4 \n\n')

axs[0].figure.set_size_inches(20, 10)

axs[0].axes.get_xaxis().set_visible(False)
axs[1].axes.get_xaxis().set_visible(False)
axs[2].axes.get_xaxis().set_visible(False)
axs[3].axes.get_xaxis().set_visible(False)


axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()
axs[4].grid()


string = r'images\timeplottotaly.png'
plt.savefig(string)
plt.show()
plt.cla()
plt.clf()
plt.close()




####################################################################################

### PLOT FOR VISUALIZING DATA IN THE TIME DOMAIN ONLY ONE LINE, PRESENTS ONLY ONE ACQUISITION OF EACH LABEL

figx, axs = plt.subplots(5,1, figsize=(40,20))
axs[0].plot(df_x_concat_100_linhas_0['time'].loc[0:200], df_x_concat_100_linhas_0['Dado_x15_0_0'].loc[0:200])
axs[0].set_xlim([0, 0.02])
axs[0].set_ylim([-0.5, 0.5])

axs[1].plot(df_x_concat_100_linhas_1['time'].loc[0:200], df_x_concat_100_linhas_1['Dado_x15_0_0'].loc[0:200])
axs[1].set_xlim([0, 0.02])
axs[1].set_ylim([-0.5, 0.5])

axs[2].plot(df_x_concat_100_linhas_2['time'].loc[0:200], df_x_concat_100_linhas_2['Dado_x15_0_0'].loc[0:200])
axs[2].set_xlim([0, 0.02])
axs[2].set_ylim([-0.5, 0.5])

axs[3].plot(df_x_concat_100_linhas_3['time'].loc[0:200], df_x_concat_100_linhas_3['Dado_x15_0_0'].loc[0:200])
axs[3].set_xlim([0, 0.02])
axs[3].set_ylim([-0.5, 0.5])

axs[4].plot(df_x_concat_100_linhas_4['time'].loc[0:200], df_x_concat_100_linhas_4['Dado_x15_0_0'].loc[0:200])
axs[4].set_xlim([0, 0.02])
axs[4].set_ylim([-0.5, 0.5])

axs[0].set(title='Original Dataset - Only a data line')
axs[0].set(ylabel='Label 0 \n\n')
axs[1].set(ylabel='Label 1 \n\n')
axs[2].set(ylabel='Label 2 \n\nValue')
axs[3].set(ylabel='Label 3 \n\n')
axs[4].set(xlabel='Time')
axs[4].set(ylabel='Label 4 \n\n')

axs[0].figure.set_size_inches(20, 10)

axs[0].axes.get_xaxis().set_visible(False)
axs[1].axes.get_xaxis().set_visible(False)
axs[2].axes.get_xaxis().set_visible(False)
axs[3].axes.get_xaxis().set_visible(False)


axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()
axs[4].grid()


string = r'images\timeplot1line.png'
plt.savefig(string)
plt.show()
plt.cla()
plt.clf()
plt.close()
    


### PLOT FOR VISUALIZING DATA IN THE TIME DOMAIN, PRESENTS ONLY ONE ACQUISITION OF EACH LABEL

figx, axs = plt.subplots(5,1, figsize=(40,20))
axs[0].plot(df_x_concat_100_linhas_0['time'], df_x_concat_100_linhas_0['Dado_x15_0_0'])
axs[0].set_xlim([0, 2])
axs[0].set_ylim([-0.5, 0.5])

axs[1].plot(df_x_concat_100_linhas_1['time'], df_x_concat_100_linhas_1['Dado_x15_0_0'])
axs[1].set_xlim([0, 2])
axs[1].set_ylim([-0.5, 0.5])

axs[2].plot(df_x_concat_100_linhas_2['time'], df_x_concat_100_linhas_2['Dado_x15_0_0'])
axs[2].set_xlim([0, 2])
axs[2].set_ylim([-0.5, 0.5])

axs[3].plot(df_x_concat_100_linhas_3['time'], df_x_concat_100_linhas_3['Dado_x15_0_0'])
axs[3].set_xlim([0, 2])
axs[3].set_ylim([-0.5, 0.5])

axs[4].plot(df_x_concat_100_linhas_4['time'], df_x_concat_100_linhas_4['Dado_x15_0_0'])
axs[4].set_xlim([0, 2])
axs[4].set_ylim([-0.5, 0.5])

axs[0].set(title='Time Domain - New Dataset')
axs[0].set(ylabel='Label 0 \n\n')
axs[1].set(ylabel='Label 1 \n\n')
axs[2].set(ylabel='Label 2 \n\nValue')
axs[3].set(ylabel='Label 3 \n\n')
axs[4].set(xlabel='Time')
axs[4].set(ylabel='Label 4 \n\n')

axs[0].figure.set_size_inches(20, 10)

axs[0].axes.get_xaxis().set_visible(False)
axs[1].axes.get_xaxis().set_visible(False)
axs[2].axes.get_xaxis().set_visible(False)
axs[3].axes.get_xaxis().set_visible(False)


axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()
axs[4].grid()


string = r'images\timeplot.png'
plt.savefig(string)
plt.show()
plt.cla()
plt.clf()
plt.close()



####################################################################################




df_x_concat_100_linhas_0 = df_x_concat_100_linhas_0.drop(columns=['time'])
df_x_concat_100_linhas_1 = df_x_concat_100_linhas_1.drop(columns=['time'])
df_x_concat_100_linhas_2 = df_x_concat_100_linhas_2.drop(columns=['time'])
df_x_concat_100_linhas_3 = df_x_concat_100_linhas_3.drop(columns=['time'])
df_x_concat_100_linhas_4 = df_x_concat_100_linhas_4.drop(columns=['time'])


df_y_concat_100_linhas_0 = df_y_concat_100_linhas_0.drop(columns=['time'])
df_y_concat_100_linhas_1 = df_y_concat_100_linhas_1.drop(columns=['time'])
df_y_concat_100_linhas_2 = df_y_concat_100_linhas_2.drop(columns=['time'])
df_y_concat_100_linhas_3 = df_y_concat_100_linhas_3.drop(columns=['time'])
df_y_concat_100_linhas_4 = df_y_concat_100_linhas_4.drop(columns=['time'])


df_z_concat_100_linhas_0 = df_z_concat_100_linhas_0.drop(columns=['time'])
df_z_concat_100_linhas_1 = df_z_concat_100_linhas_1.drop(columns=['time'])
df_z_concat_100_linhas_2 = df_z_concat_100_linhas_2.drop(columns=['time'])
df_z_concat_100_linhas_3 = df_z_concat_100_linhas_3.drop(columns=['time'])
df_z_concat_100_linhas_4 = df_z_concat_100_linhas_4.drop(columns=['time'])





########## ONLY WITH THE X AXIS ##########


### CONCATENES THE DATASETS INTO A SINGLE DATASET, ALL FAULTS IN THE SAME DATASET, USING ONLY THE X AXIS

df_list = [
    df_x_concat_100_linhas_0,
    df_x_concat_100_linhas_1,
    df_x_concat_100_linhas_2,
    df_x_concat_100_linhas_3,
    df_x_concat_100_linhas_4
]

df_total_X = pd.concat(df_list, axis=1)

df_total_T_X = df_total_X.T



df_list = [
    df_y_concat_100_linhas_0,
    df_y_concat_100_linhas_1,
    df_y_concat_100_linhas_2,
    df_y_concat_100_linhas_3,
    df_y_concat_100_linhas_4
]

df_total_Y = pd.concat(df_list, axis=1)

df_total_T_Y = df_total_Y.T


df_list = [
    df_z_concat_100_linhas_0,
    df_z_concat_100_linhas_1,
    df_z_concat_100_linhas_2,
    df_z_concat_100_linhas_3,
    df_z_concat_100_linhas_4
]

df_total_Z = pd.concat(df_list, axis=1)

df_total_T_Z = df_total_Z.T



labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
    
  

#%%



## KMEANS TEST
#%%

df0_X = StandardScaler().fit_transform(df_total_T_X)

pca_teste_X = PCA(n_components=2)
pca_df_final_normalizado_X = pca_teste_X.fit_transform(df0_X)

pca_df_final_normalizado_X = pd.DataFrame(data = pca_df_final_normalizado_X, columns = ['principal component 1', 'principal component 2'])

print('Variação por componente principal: {}'.format(pca_teste_X.explained_variance_ratio_))

divisor_print = int(len(pca_df_final_normalizado_X)/5)

## VISUALIZE THE DATA ###

plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=30)
plt.ylabel('Principal Component - 2',fontsize=30)
plt.title("Principal component - Analysis of X axis",fontsize=30)
plt.scatter(pca_df_final_normalizado_X['principal component 1'].loc[0:divisor_print]
                , pca_df_final_normalizado_X['principal component 2'].loc[0:divisor_print], c = 'b', s = 50, label='operating condition 0')
plt.scatter(pca_df_final_normalizado_X['principal component 1'].loc[divisor_print:divisor_print*2]
                , pca_df_final_normalizado_X['principal component 2'].loc[divisor_print:divisor_print*2], c = 'r', s = 50, label='operating condition 1')
plt.scatter(pca_df_final_normalizado_X['principal component 1'].loc[divisor_print*2:divisor_print*3]
                , pca_df_final_normalizado_X['principal component 2'].loc[divisor_print*2:divisor_print*3], c = 'purple', s = 50, label='operating condition 2')
plt.scatter(pca_df_final_normalizado_X['principal component 1'].loc[divisor_print*3:divisor_print*4]
                , pca_df_final_normalizado_X['principal component 2'].loc[divisor_print*3:divisor_print*4], c = 'black', s = 50, label='operating condition 3')
plt.scatter(pca_df_final_normalizado_X['principal component 1'].loc[divisor_print*4:divisor_print*5]
                , pca_df_final_normalizado_X['principal component 2'].loc[divisor_print*4:divisor_print*5], c = 'yellow', s = 50, label='operating condition 4')
plt.legend(prop={'size': 25})
string = r'images\PCA_X.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window






def apply_pca(df, n_components):
    """
    Aplica o PCA a um DataFrame e retorna o DataFrame transformado com os componentes principais.

    Parâmetros:
        df (pd.DataFrame): O DataFrame original.
        n_components (int): O número de componentes principais a serem gerados (default = 2).

    Retorna:
        pd.DataFrame: DataFrame com os componentes principais.
    """
    # Normaliza os dados
    df_scaled = StandardScaler().fit_transform(df)
    
    # Aplica o PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca_transformed = pca.fit_transform(df_scaled)

    # Cria o DataFrame com os componentes principais
    columns = [f'principal component {i + 1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=pca_transformed, columns=columns)

    return df_pca




########## KMEANS TESTE WITH 2 PCA COMPONENTS ##########


PCA_n = 2

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(4, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(2, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={"size": 26},  # Tamanho das anotações no heatmap
            )
plt.xlabel('True label', fontsize=22)  # Tamanho do rótulo do eixo X
plt.ylabel('Predicted label', fontsize=22)  # Tamanho do rótulo do eixo Y
plt.xticks(fontsize=17)  # Tamanho dos ticks do eixo X
plt.yticks(fontsize=17)  # Tamanho dos ticks do eixo Y
string = r'images\KMeans_Apenas_PCA2.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))


########## KMEANS TESTE WITH 3 PCA COMPONENTS ##########

PCA_n = 3

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(3, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(2, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label');
string = r'images\KMeans_Apenas_PCA3.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))




########## KMEANS TESTE WITH 4 PCA COMPONENTS ##########

PCA_n = 4

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(1, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(2, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label');
string = r'images\KMeans_Apenas_PCA4.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))



########## KMEANS TESTE WITH 5 PCA COMPONENTS ##########

PCA_n = 5

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label');
string = r'images\KMeans_Apenas_PCA5.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))



########## KMEANS TESTE WITH 6 PCA COMPONENTS ##########

PCA_n = 6

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(4, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label');
string = r'images\KMeans_Apenas_PCA6.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))



########## KMEANS TESTE WITH 7 PCA COMPONENTS ##########

PCA_n = 7

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(2, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label');
string = r'images\KMeans_Apenas_PCA7.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))




########## KMEANS TESTE WITH 8 PCA COMPONENTS ##########

PCA_n = 8

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(2, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={"size": 26},  # Tamanho das anotações no heatmap
            )
plt.xlabel('True label', fontsize=22)  # Tamanho do rótulo do eixo X
plt.ylabel('Predicted label', fontsize=22)  # Tamanho do rótulo do eixo Y
plt.xticks(fontsize=17)  # Tamanho dos ticks do eixo X
plt.yticks(fontsize=17)  # Tamanho dos ticks do eixo Y
string = r'images\KMeans_Apenas_PCA8.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))




########## KMEANS TESTE WITH 9 PCA COMPONENTS ##########

PCA_n = 9

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)



### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(2, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(3, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label');
string = r'images\KMeans_Apenas_PCA9.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))





########## KMEANS TESTE WITH 10 PCA COMPONENTS ##########

PCA_n = 10

PCA_X_Final = apply_pca(df_total_T_X, PCA_n)
PCA_Y_Final = apply_pca(df_total_T_Y, PCA_n)
PCA_Z_Final = apply_pca(df_total_T_Z, PCA_n)


df_list = [
    PCA_X_Final,
    PCA_Y_Final,
    PCA_Z_Final
]

PCA_Final = pd.concat(df_list, axis=1)


### TRAINING THE KMEANS ALGORITHM ###

clusters = 5

kmeans = KMeans(n_clusters=clusters)

kmeans.fit(PCA_Final)

y_kmeans = kmeans.predict(PCA_Final)


### IT IS NECESSARY TO MANUALLY CHECK WHICH NUMBER CORRESPOND TO WHICH TYPE OF DEFECT
### HOW DO WE KNOW THE LABELS AND POSSIBLE TO DO AN ACCURACY TEST TO CHECK HOW KMEAS PERFORMED
 
labels = []

for i in range(5):
    for j in range(int(len(df_total_T_X)/5)):
        labels.append(i)
  
 
  
df_acurracy_f = []
df_acurracy_f = pd.DataFrame(df_acurracy_f)
df_acurracy_f['K-means'] = y_kmeans
df_acurracy_f['Real'] = labels

df_acurracy_f['Real'].replace(0, 'operating condition 0',inplace=True)
df_acurracy_f['Real'].replace(1, 'operating condition 1',inplace=True)
df_acurracy_f['Real'].replace(2, 'operating condition 2',inplace=True)
df_acurracy_f['Real'].replace(3, 'operating condition 3',inplace=True)
df_acurracy_f['Real'].replace(4, 'operating condition 4',inplace=True)

for i in range(5):
    lista_valores = df_acurracy_f['K-means'][i*99:(i+1)*99]
    contagem = Counter(lista_valores)
    mais_comum = contagem.most_common(5)
    print(mais_comum)
        
df_acurracy_f['K-means'].replace(3, 'operating condition 0',inplace=True)
df_acurracy_f['K-means'].replace(2, 'operating condition 1',inplace=True)
df_acurracy_f['K-means'].replace(0, 'operating condition 2',inplace=True)
df_acurracy_f['K-means'].replace(1, 'operating condition 3',inplace=True)
df_acurracy_f['K-means'].replace(4, 'operating condition 4',inplace=True)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

target_names = ['op. cond. 0','op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(df_acurracy_f['Real'].values, df_acurracy_f['K-means'].values)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={"size": 26},  # Tamanho das anotações no heatmap
            )
plt.xlabel('True label', fontsize=22)  # Tamanho do rótulo do eixo X
plt.ylabel('Predicted label', fontsize=22)  # Tamanho do rótulo do eixo Y
plt.xticks(fontsize=17)  # Tamanho dos ticks do eixo X
plt.yticks(fontsize=17)  # Tamanho dos ticks do eixo Y
string = r'images\KMeans_Apenas_PCA10.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(df_acurracy_f['K-means'].values, df_acurracy_f['Real'].values, target_names=target_names))


#%%


###################################################################################
##### ANALYSIS WITH NEURAL NETWORK ######
###################################################################################
#%%

##### If it is desired to perform hyperparametrization, turn on_hiperparametrizacao = True, keep any value in the parameters

##### If the model is already hyperparameterized, turn on_hiperparametrizacao = False, in this case select the desired parameters for training the model

##### Carry out model configurations in def create_model, pay attention to the number of inputs and outputs according to the dataset


#################################################################
##### Primeiro verificar a quantidade de neurônios e arquitetura ideal
#################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
import serial
import time
from sklearn.metrics import accuracy_score, classification_report
import os
from scipy.fft import fft, fftfreq

from sklearn.model_selection import StratifiedKFold


def apply_pca(df, n_components):
    """
    Aplica o PCA a um DataFrame e retorna o DataFrame transformado com os componentes principais.

    Parâmetros:
        df (pd.DataFrame): O DataFrame original.
        n_components (int): O número de componentes principais a serem gerados (default = 2).

    Retorna:
        pd.DataFrame: DataFrame com os componentes principais.
    """
    # Normaliza os dados
    df_scaled = StandardScaler().fit_transform(df)
    
    # Aplica o PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca_transformed = pca.fit_transform(df_scaled)

    # Cria o DataFrame com os componentes principais
    columns = [f'principal component {i + 1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=pca_transformed, columns=columns)

    return df_pca




def find_best_pca_nn(X_train, y_train, X_val, y_val, max_neurons, step=1):
    """
    Encontra o melhor número de neurônios para uma rede neural usando validação simples.

    Parâmetros:
        X_train (np.array): Dados de entrada de treino.
        y_train (np.array): Rótulos categóricos (one-hot encoded) de treino.
        X_val (np.array): Dados de entrada de validação.
        y_val (np.array): Rótulos categóricos (one-hot encoded) de validação.
        max_neurons (int): Número máximo de iterações (multiplicador de neurônios).
        step (int): Incremento no número de neurônios.

    Retorna:
        tuple: Lista de métricas e configuração do melhor modelo baseado na acurácia.
    """
    metricas_pca = []

    # Função para criar o modelo
    def create_model(qnt_neu):
        model = Sequential()
        model.add(Dense(qnt_neu, input_dim=X_train.shape[1], kernel_initializer='normal', activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neu, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neu, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neu, activation='sigmoid'))
        model.add(Dense(y_train.shape[1], kernel_initializer='normal', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Testar diferentes números de neurônios
    for i in range(1, max_neurons, step):
        qnt_neu = 10 * i
        print(f"Quantidade de neurônios: {qnt_neu}")

        # Criar e treinar o modelo
        model = create_model(qnt_neu)
        history = model.fit(
            X_train, y_train,
            batch_size=10,
            epochs=100,
            verbose=0,
            validation_data=(X_val, y_val)
        )

        # Avaliar o modelo
        scores = model.evaluate(X_val, y_val, verbose=0)
        metricas_pca.append({
            'qnt_neuronios': qnt_neu,
            'loss': scores[0],
            'accuracy': scores[1]
        })

        print(f"Neurônios: {qnt_neu} | Loss: {scores[0]:.4f} | Accuracy: {scores[1]:.4f}")

    # Selecionar o melhor modelo baseado na acurácia
    melhor_metodo = max(metricas_pca, key=lambda x: x['accuracy'])
    return metricas_pca, melhor_metodo



def preprocess_data_simple(df, num_classes=5, test_size=0.2, random_state=42):
    """
    Preprocessa os dados para treinamento de uma rede neural, incluindo:
    - Encoding do rótulo.
    - Padronização das features.
    - Divisão dos dados em treino e teste.
    - Conversão dos rótulos para formato categórico (one-hot encoding).

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna que contém os rótulos.
        num_classes (int): Número de classes para o one-hot encoding.
        test_size (float): Proporção dos dados para o conjunto de teste.
        random_state (int): Semente para reprodução dos resultados.

    Retorna:
        tuple: Contendo X_train, X_test, y_train, y_test.
    """
    # Inicializar o LabelEncoder
    label_encoder = LabelEncoder()
    lista_rotulos_numerico = label_encoder.fit_transform(df['classe'].values)

    # Remover a coluna de rótulos do DataFrame
    df = df.drop(columns=['classe'])


    # Definir X e y
    X = df
    y = lista_rotulos_numerico

    # Dividir os dados em treino e teste
    X_train, X_test, y_train_not_categorical, y_test_not_categorical = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # Converter rótulos para formato categórico (one-hot encoding)
    y_train = to_categorical(y_train_not_categorical, num_classes=num_classes)
    y_test = to_categorical(y_test_not_categorical, num_classes=num_classes)

    return X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical


# Pegar o tempo inicial
start_time = time.time()

#################### TESTE COM PCA = 2 ####################

qnt_pca = 2
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca2, melhor_metodo_pca2 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### TESTE COM PCA = 3 ####################

qnt_pca = 3
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca3, melhor_metodo_pca3 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### TESTE COM PCA = 3 ####################

qnt_pca = 4
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca4, melhor_metodo_pca4 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### TESTE COM PCA = 3 ####################

qnt_pca = 5
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca5, melhor_metodo_pca5 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)




#################### TESTE COM PCA = 3 ####################

qnt_pca = 6
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca6, melhor_metodo_pca6 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### TESTE COM PCA = 3 ####################

qnt_pca = 7
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca7, melhor_metodo_pca7 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### TESTE COM PCA = 3 ####################

qnt_pca = 8
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca8, melhor_metodo_pca8 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### TESTE COM PCA = 3 ####################

qnt_pca = 9
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca9, melhor_metodo_pca9 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)





#################### TESTE COM PCA = 3 ####################

qnt_pca = 10
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

RNA_Final_PCA['classe'] = labels

X_train, X_test, y_train, y_test, y_train_not_categorical, y_test_not_categorical = preprocess_data_simple(RNA_Final_PCA)

max_neurons = 20
metricas_pca10, melhor_metodo_pca10 = find_best_pca_nn(X_train, y_train, X_test, y_test, max_neurons, step=1)



#################### RESULTADOS ####################

# Pegar o tempo final
end_time = time.time()

# Calcular a duração em minutos
duration_minutes = (end_time - start_time) / 60
print(f"A operação demorou {duration_minutes:.2f} minutos.")



melhor_metodo_pca2 = max(metricas_pca2, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca2}")

melhor_metodo_pca3 = max(metricas_pca3, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca3}")

melhor_metodo_pca4 = max(metricas_pca4, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca4}")

melhor_metodo_pca5 = max(metricas_pca5, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca5}")

melhor_metodo_pca6 = max(metricas_pca6, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca6}")

melhor_metodo_pca7 = max(metricas_pca7, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca7}")

melhor_metodo_pca8 = max(metricas_pca8, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca8}")

melhor_metodo_pca9 = max(metricas_pca9, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca9}")

melhor_metodo_pca10 = max(metricas_pca10, key=lambda x: x['accuracy'])
print(f"Melhor configuração: {melhor_metodo_pca10}")


print(f"#Melhor configuração PCA2: {melhor_metodo_pca2}")
print(f"#Melhor configuração PCA3: {melhor_metodo_pca3}")
print(f"#Melhor configuração PCA4: {melhor_metodo_pca4}")
print(f"#Melhor configuração PCA5: {melhor_metodo_pca5}")
print(f"#Melhor configuração PCA6: {melhor_metodo_pca6}")
print(f"#Melhor configuração PCA7: {melhor_metodo_pca7}")
print(f"#Melhor configuração PCA8: {melhor_metodo_pca8}")
print(f"#Melhor configuração PCA9: {melhor_metodo_pca9}")
print(f"#Melhor configuração PCA10: {melhor_metodo_pca10}")

#### RESPOSTAS PARA 1 CAMADA:
#Melhor configuração PCA2: {'qnt_neuronios': 190, 'loss_per_fold': [1.5126845836639404, 1.5971297025680542, 1.5624524354934692, 1.5475765466690063, 1.5862399339675903], 'accuracy_per_fold': [0.30000001192092896, 0.37974682450294495, 0.3291139304637909, 0.29113924503326416, 0.26582279801368713], 'media_loss': 1.5612166404724122, 'media_accuracy': 0.3131645619869232}
#Melhor configuração PCA3: {'qnt_neuronios': 150, 'loss_per_fold': [1.681517243385315, 1.7884446382522583, 1.5900932550430298, 1.601043701171875, 1.6465766429901123], 'accuracy_per_fold': [0.26249998807907104, 0.2278480976819992, 0.39240506291389465, 0.3670886158943176, 0.3291139304637909], 'media_loss': 1.661535096168518, 'media_accuracy': 0.3157911390066147}
#Melhor configuração PCA4: {'qnt_neuronios': 50, 'loss_per_fold': [1.6817280054092407, 1.6376172304153442, 1.714574933052063, 1.5781538486480713, 1.7090376615524292], 'accuracy_per_fold': [0.25, 0.27848100662231445, 0.29113924503326416, 0.3544303774833679, 0.2405063360929489], 'media_loss': 1.6642223358154298, 'media_accuracy': 0.2829113930463791}
#Melhor configuração PCA5: {'qnt_neuronios': 130, 'loss_per_fold': [1.8737964630126953, 1.8102236986160278, 1.7726805210113525, 1.7819039821624756, 1.8849761486053467], 'accuracy_per_fold': [0.2750000059604645, 0.3291139304637909, 0.26582279801368713, 0.27848100662231445, 0.2531645596027374], 'media_loss': 1.8247161626815795, 'media_accuracy': 0.28031646013259887}
#Melhor configuração PCA6: {'qnt_neuronios': 150, 'loss_per_fold': [1.8822542428970337, 1.983551263809204, 1.87568199634552, 1.8992162942886353, 2.070892810821533], 'accuracy_per_fold': [0.32499998807907104, 0.2151898741722107, 0.26582279801368713, 0.2405063360929489, 0.18987341225147247], 'media_loss': 1.9423193216323853, 'media_accuracy': 0.24727848172187805}
#Melhor configuração PCA7: {'qnt_neuronios': 40, 'loss_per_fold': [1.9438774585723877, 1.8380721807479858, 1.7606014013290405, 1.6268038749694824, 1.8774768114089966], 'accuracy_per_fold': [0.22499999403953552, 0.3164556920528412, 0.2531645596027374, 0.4177215099334717, 0.2151898741722107], 'media_loss': 1.8093663454055786, 'media_accuracy': 0.2855063259601593}
#Melhor configuração PCA8: {'qnt_neuronios': 170, 'loss_per_fold': [2.082212448120117, 2.2596068382263184, 1.8206043243408203, 1.9167639017105103, 2.4283697605133057], 'accuracy_per_fold': [0.3125, 0.2151898741722107, 0.3291139304637909, 0.3291139304637909, 0.17721518874168396], 'media_loss': 2.1015114545822144, 'media_accuracy': 0.27262658476829527}
#Melhor configuração PCA9: {'qnt_neuronios': 40, 'loss_per_fold': [1.9523570537567139, 1.9258602857589722, 1.8001487255096436, 2.037656545639038, 2.0265252590179443], 'accuracy_per_fold': [0.2750000059604645, 0.2531645596027374, 0.3291139304637909, 0.2531645596027374, 0.2278480976819992], 'media_loss': 1.9485095739364624, 'media_accuracy': 0.2676582306623459}
#Melhor configuração PCA10: {'qnt_neuronios': 90, 'loss_per_fold': [2.3015735149383545, 2.500378131866455, 1.8917043209075928, 2.016017198562622, 2.319532871246338], 'accuracy_per_fold': [0.2750000059604645, 0.20253165066242218, 0.4177215099334717, 0.3037974536418915, 0.26582279801368713], 'media_loss': 2.2058412075042724, 'media_accuracy': 0.2929746836423874}
####


#### RESPOSTAS PARA 2 CAMADA:
#Melhor configuração PCA2: {'qnt_neuronios': 90, 'loss_per_fold': [1.4282724857330322, 1.5094735622406006, 1.4586657285690308, 1.5567196607589722, 1.504493236541748], 'accuracy_per_fold': [0.4124999940395355, 0.3417721390724182, 0.40506330132484436, 0.3417721390724182, 0.3291139304637909], 'media_loss': 1.4915249347686768, 'media_accuracy': 0.36604430079460143}
#Melhor configuração PCA3: {'qnt_neuronios': 150, 'loss_per_fold': [1.7971998453140259, 2.306807041168213, 1.8892881870269775, 1.857657551765442, 1.9812291860580444], 'accuracy_per_fold': [0.3499999940395355, 0.2151898741722107, 0.3037974536418915, 0.3417721390724182, 0.3037974536418915], 'media_loss': 1.9664363622665406, 'media_accuracy': 0.3029113829135895}
#Melhor configuração PCA4: {'qnt_neuronios': 60, 'loss_per_fold': [1.8122001886367798, 1.747327208518982, 1.9641252756118774, 1.5426361560821533, 1.7153513431549072], 'accuracy_per_fold': [0.26249998807907104, 0.27848100662231445, 0.2278480976819992, 0.3670886158943176, 0.3037974536418915], 'media_loss': 1.75632803440094, 'media_accuracy': 0.2879430323839188}
#Melhor configuração PCA5: {'qnt_neuronios': 120, 'loss_per_fold': [2.737508773803711, 2.239497661590576, 2.211653709411621, 2.341655969619751, 2.2536230087280273], 'accuracy_per_fold': [0.22499999403953552, 0.3037974536418915, 0.3037974536418915, 0.3037974536418915, 0.27848100662231445], 'media_loss': 2.356787824630737, 'media_accuracy': 0.2829746723175049}
#Melhor configuração PCA6: {'qnt_neuronios': 130, 'loss_per_fold': [2.5979504585266113, 2.6688308715820312, 2.645550489425659, 2.639875888824463, 2.560661792755127], 'accuracy_per_fold': [0.25, 0.2405063360929489, 0.2531645596027374, 0.29113924503326416, 0.2531645596027374], 'media_loss': 2.622573900222778, 'media_accuracy': 0.2575949400663376}
#Melhor configuração PCA7: {'qnt_neuronios': 130, 'loss_per_fold': [2.9967989921569824, 3.1207804679870605, 2.505134344100952, 2.692117929458618, 3.3856170177459717], 'accuracy_per_fold': [0.3125, 0.27848100662231445, 0.3417721390724182, 0.27848100662231445, 0.2531645596027374], 'media_loss': 2.940089750289917, 'media_accuracy': 0.2928797423839569}
#Melhor configuração PCA8: {'qnt_neuronios': 70, 'loss_per_fold': [2.5719821453094482, 2.4533932209014893, 2.246039867401123, 2.1455042362213135, 2.9347927570343018], 'accuracy_per_fold': [0.21250000596046448, 0.20253165066242218, 0.3291139304637909, 0.3417721390724182, 0.18987341225147247], 'media_loss': 2.470342445373535, 'media_accuracy': 0.25515822768211366}
#Melhor configuração PCA9: {'qnt_neuronios': 110, 'loss_per_fold': [3.2482643127441406, 3.270606756210327, 2.807305097579956, 2.9124138355255127, 3.412374258041382], 'accuracy_per_fold': [0.26249998807907104, 0.26582279801368713, 0.3164556920528412, 0.3291139304637909, 0.2531645596027374], 'media_loss': 3.1301928520202638, 'media_accuracy': 0.2854113936424255}
#Melhor configuração PCA10: {'qnt_neuronios': 50, 'loss_per_fold': [2.7483859062194824, 2.747516393661499, 1.9975813627243042, 2.094758987426758, 2.8440709114074707], 'accuracy_per_fold': [0.22499999403953552, 0.2151898741722107, 0.3670886158943176, 0.39240506291389465, 0.3037974536418915], 'media_loss': 2.486462712287903, 'media_accuracy': 0.30069620013237}  
####


#### RESPOSTAS PARA 3 CAMADA:
#Melhor configuração PCA2: {'qnt_neuronios': 110, 'loss_per_fold': [1.5788295269012451, 1.524511456489563, 1.5320175886154175, 1.5360558032989502, 1.4633889198303223], 'accuracy_per_fold': [0.42500001192092896, 0.40506330132484436, 0.37974682450294495, 0.3544303774833679, 0.4430379867553711], 'media_loss': 1.5269606590270997, 'media_accuracy': 0.40145570039749146}
#Melhor configuração PCA3: {'qnt_neuronios': 150, 'loss_per_fold': [1.9983203411102295, 2.322340488433838, 2.7332992553710938, 2.2034177780151367, 2.3080475330352783], 'accuracy_per_fold': [0.2874999940395355, 0.3291139304637909, 0.3164556920528412, 0.3670886158943176, 0.2531645596027374], 'media_loss': 2.3130850791931152, 'media_accuracy': 0.31066455841064455}
#Melhor configuração PCA4: {'qnt_neuronios': 140, 'loss_per_fold': [2.7544071674346924, 2.441530227661133, 2.570712089538574, 2.4410929679870605, 2.605475425720215], 'accuracy_per_fold': [0.21250000596046448, 0.27848100662231445, 0.3037974536418915, 0.3544303774833679, 0.27848100662231445], 'media_loss': 2.562643575668335, 'media_accuracy': 0.28553797006607057}
#Melhor configuração PCA5: {'qnt_neuronios': 170, 'loss_per_fold': [3.774707078933716, 3.1598727703094482, 2.475252866744995, 3.289315938949585, 3.3732380867004395], 'accuracy_per_fold': [0.25, 0.2531645596027374, 0.29113924503326416, 0.3417721390724182, 0.3164556920528412], 'media_loss': 3.2144773483276365, 'media_accuracy': 0.2905063271522522}
#Melhor configuração PCA6: {'qnt_neuronios': 190, 'loss_per_fold': [3.400057315826416, 3.6117236614227295, 3.419626474380493, 3.789222478866577, 3.6180343627929688], 'accuracy_per_fold': [0.30000001192092896, 0.20253165066242218, 0.3544303774833679, 0.2531645596027374, 0.26582279801368713], 'media_loss': 3.5677328586578367, 'media_accuracy': 0.2751898795366287}
#Melhor configuração PCA7: {'qnt_neuronios': 150, 'loss_per_fold': [3.4578843116760254, 3.645740270614624, 3.122701406478882, 3.1306724548339844, 4.3339385986328125], 'accuracy_per_fold': [0.30000001192092896, 0.2531645596027374, 0.26582279801368713, 0.3670886158943176, 0.26582279801368713], 'media_loss': 3.5381874084472655, 'media_accuracy': 0.2903797566890717}
#Melhor configuração PCA8: {'qnt_neuronios': 30, 'loss_per_fold': [2.033297538757324, 2.010950803756714, 1.7837817668914795, 1.8342219591140747, 2.1143617630004883], 'accuracy_per_fold': [0.3499999940395355, 0.13924050331115723, 0.3417721390724182, 0.27848100662231445, 0.18987341225147247], 'media_loss': 1.955322766304016, 'media_accuracy': 0.25987341105937956}
#Melhor configuração PCA9: {'qnt_neuronios': 120, 'loss_per_fold': [3.481919765472412, 3.152243137359619, 3.0380897521972656, 3.617762327194214, 3.781205892562866], 'accuracy_per_fold': [0.30000001192092896, 0.3037974536418915, 0.29113924503326416, 0.2531645596027374, 0.2151898741722107], 'media_loss': 3.4142441749572754, 'media_accuracy': 0.27265822887420654}
#Melhor configuração PCA10: {'qnt_neuronios': 180, 'loss_per_fold': [4.731559753417969, 4.56497049331665, 3.5692803859710693, 3.9173507690429688, 4.732443809509277], 'accuracy_per_fold': [0.25, 0.3164556920528412, 0.3670886158943176, 0.29113924503326416, 0.26582279801368713], 'media_loss': 4.303121042251587, 'media_accuracy': 0.29810127019882204}


#### RESPOSTAS PARA 4 CAMADA:
#Melhor configuração PCA2: {'qnt_neuronios': 70, 'loss': 1.5825303792953491, 'accuracy': 0.3400000035762787}
#Melhor configuração PCA3: {'qnt_neuronios': 190, 'loss': 2.2300260066986084, 'accuracy': 0.3100000023841858}
#Melhor configuração PCA4: {'qnt_neuronios': 130, 'loss': 2.031851291656494, 'accuracy': 0.4000000059604645}
#Melhor configuração PCA5: {'qnt_neuronios': 20, 'loss': 1.5351841449737549, 'accuracy': 0.3199999928474426}
#Melhor configuração PCA6: {'qnt_neuronios': 130, 'loss': 2.0753979682922363, 'accuracy': 0.4000000059604645}
#Melhor configuração PCA7: {'qnt_neuronios': 10, 'loss': 1.7039299011230469, 'accuracy': 0.33000001311302185}
#Melhor configuração PCA8: {'qnt_neuronios': 10, 'loss': 1.7185713052749634, 'accuracy': 0.3100000023841858}
#Melhor configuração PCA9: {'qnt_neuronios': 10, 'loss': 1.7103441953659058, 'accuracy': 0.3100000023841858}
#Melhor configuração PCA10: {'qnt_neuronios': 10, 'loss': 1.759658694267273, 'accuracy': 0.33000001311302185}


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import EarlyStopping
from joblib import Parallel, delayed

### Neurons for 1 layer
neuronios = [190, 150, 50, 130, 150, 40, 170, 40, 90]

### Neurons for 2 layer
#neuronios = [90, 150, 60, 120, 130, 130, 70, 110, 70]

### Neurons for 3 layer
#neuronios = [110, 150, 140, 170, 190, 150, 30, 120, 180]

### Neurons for 4 layer
#neuronios = [70, 190, 130, 20, 130, 10, 10, 10, 10]

resultados_pca_1_layer = []

for quantidade_pca in range(2,11):
    print(f"============ TESTE PCA {quantidade_pca}  ==============")
    
    qnt_pca = quantidade_pca
    RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
    RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
    RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)
    
    
    df_list = [
        RNA_X_Final_PCA,
        RNA_Y_Final_PCA,
        RNA_Z_Final_PCA
    ]
    
    RNA_Final_PCA = pd.concat(df_list, axis=1)
    
    #RNA_Final_PCA['classe'] = labels
    
    
    X = RNA_Final_PCA
    y = labels
    
    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encoding para as classes
    y = to_categorical(y)
    
    qnt_neurons = neuronios[quantidade_pca-2]
    
    print(f"============ NEURONIOS {qnt_neurons}  ==============")
    
    # Definir a função para criar o modelo
    def create_model(optimizer, init, activation):
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(qnt_neurons, kernel_initializer=init, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Configurar hiperparâmetros para busca
    param_grid = {
        'batch_size': [1, 5, 10, 20],
        'epochs': [10, 50, 100, 200, 300],
        'model__optimizer': ['adam', 'rmsprop', 'SGD'],
        'model__init': ['uniform', 'normal'],
        'model__activation': ['relu', 'tanh', 'sigmoid', 'linear']
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    param_combinations = list(ParameterGrid(param_grid))
    best_accuracy = 0
    best_params = None
    
    # Função para processar uma combinação de hiperparâmetros
    def process_combination(params):
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Criar e treinar o modelo
            model = create_model(
                optimizer=params['model__optimizer'],
                init=params['model__init'],
                activation=params['model__activation']
            )
    
            # Configurar EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
            # Treinamento do modelo
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
    
            # Avaliar o modelo
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_labels, y_pred)
            accuracies.append(accuracy)
    
        mean_accuracy = np.mean(accuracies)
        return params, mean_accuracy
    
    # Paralelizar o processo de validação cruzada
    results = Parallel(n_jobs=-1)(delayed(process_combination)(params) for params in param_combinations)
    
    # Identificar os melhores hiperparâmetros
    for params, mean_accuracy in results:
        print(f'Hiperparâmetros: {params} - Acurácia Média: {mean_accuracy:.4f}')
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params
    
    resultados_pca_1_layer.append({
    'qnt_pca': qnt_pca,
    'qnt_neurons': qnt_neurons,
    'parametros': best_params,
    'melhores_parametros': best_accuracy
    })
    # Exibir os melhores hiperparâmetros e resultados
    print(f'Melhores hiperparâmetros: {best_params}')
    print(f'Melhor acurácia média: {best_accuracy:.4f}')


print(resultados_pca_1_layer)




### Neurons for 1 layer
#neuronios = [190, 150, 50, 130, 150, 40, 170, 40, 90]

### Neurons for 2 layer
neuronios = [90, 150, 60, 120, 130, 130, 70, 110, 70]

### Neurons for 3 layer
#neuronios = [110, 150, 140, 170, 190, 150, 30, 120, 180]

### Neurons for 4 layer
#neuronios = [70, 190, 130, 20, 130, 10, 10, 10, 10]

resultados_pca_2_layer = []

for quantidade_pca in range(2,11):
    print(f"============ TESTE PCA {quantidade_pca}  ==============")
    
    qnt_pca = quantidade_pca
    RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
    RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
    RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)
    
    
    df_list = [
        RNA_X_Final_PCA,
        RNA_Y_Final_PCA,
        RNA_Z_Final_PCA
    ]
    
    RNA_Final_PCA = pd.concat(df_list, axis=1)
    
    #RNA_Final_PCA['classe'] = labels
    
    
    X = RNA_Final_PCA
    y = labels
    
    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encoding para as classes
    y = to_categorical(y)
    
    qnt_neurons = neuronios[quantidade_pca-2]
    
    print(f"============ NEURONIOS {qnt_neurons}  ==============")
    
    # Definir a função para criar o modelo
    def create_model(optimizer, init, activation):
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(qnt_neurons, kernel_initializer=init, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Configurar hiperparâmetros para busca
    param_grid = {
        'batch_size': [1, 5, 10, 20],
        'epochs': [10, 50, 100, 200, 300],
        'model__optimizer': ['adam', 'rmsprop', 'SGD'],
        'model__init': ['uniform', 'normal'],
        'model__activation': ['relu', 'tanh', 'sigmoid', 'linear']
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    param_combinations = list(ParameterGrid(param_grid))
    best_accuracy = 0
    best_params = None
    
    # Função para processar uma combinação de hiperparâmetros
    def process_combination(params):
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Criar e treinar o modelo
            model = create_model(
                optimizer=params['model__optimizer'],
                init=params['model__init'],
                activation=params['model__activation']
            )
    
            # Configurar EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
            # Treinamento do modelo
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
    
            # Avaliar o modelo
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_labels, y_pred)
            accuracies.append(accuracy)
    
        mean_accuracy = np.mean(accuracies)
        return params, mean_accuracy
    
    # Paralelizar o processo de validação cruzada
    results = Parallel(n_jobs=-1)(delayed(process_combination)(params) for params in param_combinations)
    
    # Identificar os melhores hiperparâmetros
    for params, mean_accuracy in results:
        print(f'Hiperparâmetros: {params} - Acurácia Média: {mean_accuracy:.4f}')
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params
    
    resultados_pca_2_layer.append({
    'qnt_pca': qnt_pca,
    'qnt_neurons': qnt_neurons,
    'parametros': best_params,
    'melhores_parametros': best_accuracy
    })
    # Exibir os melhores hiperparâmetros e resultados
    print(f'Melhores hiperparâmetros: {best_params}')
    print(f'Melhor acurácia média: {best_accuracy:.4f}')


print(resultados_pca_2_layer)



### Neurons for 1 layer
#neuronios = [190, 150, 50, 130, 150, 40, 170, 40, 90]

### Neurons for 2 layer
#neuronios = [90, 150, 60, 120, 130, 130, 70, 110, 70]

### Neurons for 3 layer
neuronios = [110, 150, 140, 170, 190, 150, 30, 120, 180]

### Neurons for 4 layer
#neuronios = [70, 190, 130, 20, 130, 10, 10, 10, 10]

resultados_pca_3_layer = []

for quantidade_pca in range(2,11):
    print(f"============ TESTE PCA {quantidade_pca}  ==============")
    
    qnt_pca = quantidade_pca
    RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
    RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
    RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)
    
    
    df_list = [
        RNA_X_Final_PCA,
        RNA_Y_Final_PCA,
        RNA_Z_Final_PCA
    ]
    
    RNA_Final_PCA = pd.concat(df_list, axis=1)
    
    #RNA_Final_PCA['classe'] = labels
    
    
    X = RNA_Final_PCA
    y = labels
    
    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encoding para as classes
    y = to_categorical(y)
    
    qnt_neurons = neuronios[quantidade_pca-2]
    
    print(f"============ NEURONIOS {qnt_neurons}  ==============")
    
    # Definir a função para criar o modelo
    def create_model(optimizer, init, activation):
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(qnt_neurons, kernel_initializer=init, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Configurar hiperparâmetros para busca
    param_grid = {
        'batch_size': [1, 5, 10, 20],
        'epochs': [10, 50, 100, 200, 300],
        'model__optimizer': ['adam', 'rmsprop', 'SGD'],
        'model__init': ['uniform', 'normal'],
        'model__activation': ['relu', 'tanh', 'sigmoid', 'linear']
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    param_combinations = list(ParameterGrid(param_grid))
    best_accuracy = 0
    best_params = None
    
    # Função para processar uma combinação de hiperparâmetros
    def process_combination(params):
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Criar e treinar o modelo
            model = create_model(
                optimizer=params['model__optimizer'],
                init=params['model__init'],
                activation=params['model__activation']
            )
    
            # Configurar EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
            # Treinamento do modelo
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
    
            # Avaliar o modelo
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_labels, y_pred)
            accuracies.append(accuracy)
    
        mean_accuracy = np.mean(accuracies)
        return params, mean_accuracy
    
    # Paralelizar o processo de validação cruzada
    results = Parallel(n_jobs=-1)(delayed(process_combination)(params) for params in param_combinations)
    
    # Identificar os melhores hiperparâmetros
    for params, mean_accuracy in results:
        print(f'Hiperparâmetros: {params} - Acurácia Média: {mean_accuracy:.4f}')
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params
    
    resultados_pca_3_layer.append({
    'qnt_pca': qnt_pca,
    'qnt_neurons': qnt_neurons,
    'parametros': best_params,
    'melhores_parametros': best_accuracy
    })
    # Exibir os melhores hiperparâmetros e resultados
    print(f'Melhores hiperparâmetros: {best_params}')
    print(f'Melhor acurácia média: {best_accuracy:.4f}')


print(resultados_pca_3_layer)





### Neurons for 1 layer
#neuronios = [190, 150, 50, 130, 150, 40, 170, 40, 90]

### Neurons for 2 layer
#neuronios = [90, 150, 60, 120, 130, 130, 70, 110, 70]

### Neurons for 3 layer
#neuronios = [110, 150, 140, 170, 190, 150, 30, 120, 180]

### Neurons for 4 layer
neuronios = [70, 190, 130, 20, 130, 10, 10, 10, 10]

resultados_pca_4_layer = []

for quantidade_pca in range(2,11):
    print(f"============ TESTE PCA {quantidade_pca}  ==============")
    
    qnt_pca = quantidade_pca
    RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
    RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
    RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)
    
    
    df_list = [
        RNA_X_Final_PCA,
        RNA_Y_Final_PCA,
        RNA_Z_Final_PCA
    ]
    
    RNA_Final_PCA = pd.concat(df_list, axis=1)
    
    #RNA_Final_PCA['classe'] = labels
    
    
    X = RNA_Final_PCA
    y = labels
    
    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encoding para as classes
    y = to_categorical(y)
    
    qnt_neurons = neuronios[quantidade_pca-2]
    
    print(f"============ NEURONIOS {qnt_neurons}  ==============")
    
    # Definir a função para criar o modelo
    def create_model(optimizer, init, activation):
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(qnt_neurons, kernel_initializer=init, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(2 * qnt_neurons, activation=activation))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Configurar hiperparâmetros para busca
    param_grid = {
        'batch_size': [1, 5, 10, 20],
        'epochs': [10, 50, 100, 200, 300],
        'model__optimizer': ['adam', 'rmsprop', 'SGD'],
        'model__init': ['uniform', 'normal'],
        'model__activation': ['relu', 'tanh', 'sigmoid', 'linear']
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    param_combinations = list(ParameterGrid(param_grid))
    best_accuracy = 0
    best_params = None
    
    # Função para processar uma combinação de hiperparâmetros
    def process_combination(params):
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Criar e treinar o modelo
            model = create_model(
                optimizer=params['model__optimizer'],
                init=params['model__init'],
                activation=params['model__activation']
            )
    
            # Configurar EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
            # Treinamento do modelo
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
    
            # Avaliar o modelo
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_labels, y_pred)
            accuracies.append(accuracy)
    
        mean_accuracy = np.mean(accuracies)
        return params, mean_accuracy
    
    # Paralelizar o processo de validação cruzada
    results = Parallel(n_jobs=-1)(delayed(process_combination)(params) for params in param_combinations)
    
    # Identificar os melhores hiperparâmetros
    for params, mean_accuracy in results:
        print(f'Hiperparâmetros: {params} - Acurácia Média: {mean_accuracy:.4f}')
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params
    
    resultados_pca_4_layer.append({
    'qnt_pca': qnt_pca,
    'qnt_neurons': qnt_neurons,
    'parametros': best_params,
    'melhores_parametros': best_accuracy
    })
    # Exibir os melhores hiperparâmetros e resultados
    print(f'Melhores hiperparâmetros: {best_params}')
    print(f'Melhor acurácia média: {best_accuracy:.4f}')


print(resultados_pca_4_layer)

print(resultados_pca_1_layer)
print(resultados_pca_2_layer)
print(resultados_pca_3_layer)
print(resultados_pca_4_layer)




### HIPERPARAMETRIZATION WITH 1 LAYER
# [{'qnt_pca': 2, 'qnt_neurons': 190, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3499987975855518}, 
# {'qnt_pca': 3, 'qnt_neurons': 150, 'parametros': {'batch_size': 20, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.3659909097467715}, 
# {'qnt_pca': 4, 'qnt_neurons': 50, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.3959550777962148}, 
# {'qnt_pca': 5, 'qnt_neurons': 130, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.4139672462304307}, 
# {'qnt_pca': 6, 'qnt_neurons': 150, 'parametros': {'batch_size': 1, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.41598730250342686}, 
# {'qnt_pca': 7, 'qnt_neurons': 40, 'parametros': {'batch_size': 1, 'epochs': 10, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3059543563475459}, 
# {'qnt_pca': 8, 'qnt_neurons': 170, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3300266936007503}, 
# {'qnt_pca': 9, 'qnt_neurons': 40, 'parametros': {'batch_size': 1, 'epochs': 200, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.30791429189813146}, 
# {'qnt_pca': 10, 'qnt_neurons': 90, 'parametros': {'batch_size': 1, 'epochs': 200, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3600509823726042}]

### HIPERPARAMETRIZATION WITH 2 LAYER
#[{'qnt_pca': 2, 'qnt_neurons': 90, 'parametros': {'batch_size': 10, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.36989875670346056}, 
# {'qnt_pca': 3, 'qnt_neurons': 150, 'parametros': {'batch_size': 1, 'epochs': 200, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3839670057475411}, 
# {'qnt_pca': 4, 'qnt_neurons': 60, 'parametros': {'batch_size': 1, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3979270374912825}, 
# {'qnt_pca': 5, 'qnt_neurons': 120, 'parametros': {'batch_size': 10, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.41194718995743457}, 
# {'qnt_pca': 6, 'qnt_neurons': 130, 'parametros': {'batch_size': 1, 'epochs': 10, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.42607555972392563}, 
# {'qnt_pca': 7, 'qnt_neurons': 130, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.37601904624485966}, 
# {'qnt_pca': 8, 'qnt_neurons': 70, 'parametros': {'batch_size': 1, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.387959021715605}, 
# {'qnt_pca': 9, 'qnt_neurons': 110, 'parametros': {'batch_size': 1, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.4058749969939639}, 
# {'qnt_pca': 10, 'qnt_neurons': 70, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.41805545535435157}]

### HIPERPARAMETRIZATION WITH 3 LAYER
#[{'qnt_pca': 2, 'qnt_neurons': 110, 'parametros': {'batch_size': 10, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.37396291753841715}, 
# {'qnt_pca': 3, 'qnt_neurons': 150, 'parametros': {'batch_size': 1, 'epochs': 10, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.39391097323425434}, 
# {'qnt_pca': 4, 'qnt_neurons': 140, 'parametros': {'batch_size': 1, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.42199937474448695}, 
# {'qnt_pca': 5, 'qnt_neurons': 170, 'parametros': {'batch_size': 5, 'epochs': 200, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.43000745496957893}, 
# {'qnt_pca': 6, 'qnt_neurons': 190, 'parametros': {'batch_size': 5, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.442031599451699}, 
# {'qnt_pca': 7, 'qnt_neurons': 150, 'parametros': {'batch_size': 1, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.3939470456677007}, 
# {'qnt_pca': 8, 'qnt_neurons': 30, 'parametros': {'batch_size': 5, 'epochs': 200, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.40805136714522766}, 
# {'qnt_pca': 9, 'qnt_neurons': 120, 'parametros': {'batch_size': 1, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.43793136618329614}, 
# {'qnt_pca': 10, 'qnt_neurons': 180, 'parametros': {'batch_size': 1, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.46396363898708604}]


### HIPERPARAMETRIZATION WITH 4 LAYER
#[{'qnt_pca': 2, 'qnt_neurons': 70, 'parametros': {'batch_size': 20, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.38403915061443383}, 
#{'qnt_pca': 3, 'qnt_neurons': 190, 'parametros': {'batch_size': 1, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3899189572661905}, 
#{'qnt_pca': 4, 'qnt_neurons': 130, 'parametros': {'batch_size': 5, 'epochs': 100, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.4299713825361326}, 
#{'qnt_pca': 5, 'qnt_neurons': 20, 'parametros': {'batch_size': 1, 'epochs': 200, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.4120193348243273}, 
#{'qnt_pca': 6, 'qnt_neurons': 130, 'parametros': {'batch_size': 10, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.4579876391794724}, 
#{'qnt_pca': 7, 'qnt_neurons': 10, 'parametros': {'batch_size': 5, 'epochs': 50, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'rmsprop'}, 'melhores_parametros': 0.3560709905490224}, 
#{'qnt_pca': 8, 'qnt_neurons': 10, 'parametros': {'batch_size': 1, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.3700911430151745}, 
#{'qnt_pca': 9, 'qnt_neurons': 10, 'parametros': {'batch_size': 10, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'uniform', 'model__optimizer': 'adam'}, 'melhores_parametros': 0.3900512228554938}, 
#{'qnt_pca': 10, 'qnt_neurons': 10, 'parametros': {'batch_size': 1, 'epochs': 300, 'model__activation': 'relu', 'model__init': 'normal', 'model__optimizer': 'SGD'}, 'melhores_parametros': 0.38193492533006274}]


from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np



quantidade_pca = 6
qnt_pca = quantidade_pca
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

#RNA_Final_PCA['classe'] = labels
    

# Preparação dos dados
X = RNA_Final_PCA
y = labels

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding para as classes
y = to_categorical(y)

# Dividir os dados em treinamento+validação (80%) e teste (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=labels)

# Parâmetros fixos
qnt_neurons = 130
fixed_params = {
    'batch_size': 10,
    'epochs': 300,
    'optimizer': 'adam',
    'init': 'uniform',
    'activation': 'relu'
}

print(f"============ NEURONIOS {qnt_neurons}  ==============")

# Função para criar o modelo
def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train_val.shape[1],)))
    model.add(Dense(qnt_neurons, kernel_initializer=fixed_params['init'], activation=fixed_params['activation']))
    model.add(Dropout(0.2))
    model.add(Dense(2 * qnt_neurons, activation=fixed_params['activation']))
    model.add(Dropout(0.2))
    model.add(Dense(2 * qnt_neurons, activation=fixed_params['activation']))
    model.add(Dropout(0.2))
    model.add(Dense(2 * qnt_neurons, activation=fixed_params['activation']))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer=fixed_params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Divisão do conjunto de treinamento+validação para validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# Para armazenar as curvas de perda e acurácia de todos os folds
losses = []
val_losses = []
accuracies = []
val_accuracies = []

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    # Criar e treinar o modelo
    model = create_model()
    
    # Treinamento do modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=fixed_params['epochs'],
        batch_size=fixed_params['batch_size'],
        verbose=0
    )
    
    # Salvar as métricas do histórico
    losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])
    accuracies.append(history.history['accuracy'])
    val_accuracies.append(history.history['val_accuracy'])
    
    # Avaliação no conjunto de validação
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    accuracy = accuracy_score(y_val_labels, y_val_pred)
    fold_accuracies.append(accuracy)

# Calcular as médias sem necessidade de truncamento
mean_loss = np.mean(losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
mean_accuracy = np.mean(accuracies, axis=0)
mean_val_accuracy = np.mean(val_accuracies, axis=0)

# Plotar as curvas de perda e acurácia médias
plt.figure(figsize=(12, 5))

# Curva de perda
plt.subplot(1, 2, 1)
plt.plot(mean_loss, label='Train')
plt.plot(mean_val_loss, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Curva de acurácia
plt.subplot(1, 2, 2)
plt.plot(mean_accuracy, label='Train')
plt.plot(mean_val_accuracy, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Resultados da validação cruzada
mean_validation_accuracy = np.mean(fold_accuracies)
print(f"Acurácias por dobra (validação): {fold_accuracies}")
print(f"Acurácia média na validação cruzada: {mean_validation_accuracy:.4f}")

# Avaliação no conjunto de teste final
y_test_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
final_test_accuracy = accuracy_score(y_test_labels, y_test_pred)
print(f"Acurácia no conjunto de teste: {final_test_accuracy:.4f}")

target_names = ['op. cond. 0', 'op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(y_test_labels, y_test_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={"size": 26},  # Tamanho das anotações no heatmap
            )
plt.xlabel('True label', fontsize=22)  # Tamanho do rótulo do eixo X
plt.ylabel('Predicted label', fontsize=22)  # Tamanho do rótulo do eixo Y
plt.xticks(fontsize=17)  # Tamanho dos ticks do eixo X
plt.yticks(fontsize=17)  # Tamanho dos ticks do eixo Y
string = r'images\ANN_PCA_10.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(y_test_labels, y_test_pred, target_names=target_names))






quantidade_pca = 10
qnt_pca = quantidade_pca
RNA_X_Final_PCA = apply_pca(df_total_T_X, qnt_pca)
RNA_Y_Final_PCA = apply_pca(df_total_T_Y, qnt_pca)
RNA_Z_Final_PCA = apply_pca(df_total_T_Z, qnt_pca)


df_list = [
    RNA_X_Final_PCA,
    RNA_Y_Final_PCA,
    RNA_Z_Final_PCA
]

RNA_Final_PCA = pd.concat(df_list, axis=1)

#RNA_Final_PCA['classe'] = labels
    

# Preparação dos dados
X = RNA_Final_PCA
y = labels

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding para as classes
y = to_categorical(y)

# Dividir os dados em treinamento+validação (70%) e teste (30%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=labels)

# Parâmetros fixos
qnt_neurons = 180
fixed_params = {
    'batch_size': 10,
    'epochs': 100,
    'optimizer': 'SGD',
    'init': 'normal',
    'activation': 'relu'
}

print(f"============ NEURONIOS {qnt_neurons}  ==============")

# Função para criar o modelo
def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train_val.shape[1],)))
    model.add(Dense(qnt_neurons, kernel_initializer=fixed_params['init'], activation=fixed_params['activation']))
    model.add(Dropout(0.2))
    model.add(Dense(2 * qnt_neurons, activation=fixed_params['activation']))
    model.add(Dropout(0.2))
    model.add(Dense(2 * qnt_neurons, activation=fixed_params['activation']))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer=fixed_params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Divisão do conjunto de treinamento+validação para validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# Para armazenar as curvas de perda e acurácia de todos os folds
losses = []
val_losses = []
accuracies = []
val_accuracies = []

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    # Criar e treinar o modelo
    model = create_model()
    
    # Treinamento do modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=fixed_params['epochs'],
        batch_size=fixed_params['batch_size'],
        verbose=0
    )
    
    # Salvar as métricas do histórico
    losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])
    accuracies.append(history.history['accuracy'])
    val_accuracies.append(history.history['val_accuracy'])
    
    # Avaliação no conjunto de validação
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    accuracy = accuracy_score(y_val_labels, y_val_pred)
    fold_accuracies.append(accuracy)

# Calcular as médias sem necessidade de truncamento
mean_loss = np.mean(losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
mean_accuracy = np.mean(accuracies, axis=0)
mean_val_accuracy = np.mean(val_accuracies, axis=0)

# Plotar as curvas de perda e acurácia médias
plt.figure(figsize=(12, 5))

# Curva de perda
plt.subplot(1, 2, 1)
plt.plot(mean_loss, label='Train')
plt.plot(mean_val_loss, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Curva de acurácia
plt.subplot(1, 2, 2)
plt.plot(mean_accuracy, label='Train')
plt.plot(mean_val_accuracy, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Resultados da validação cruzada
mean_validation_accuracy = np.mean(fold_accuracies)
print(f"Acurácias por dobra (validação): {fold_accuracies}")
print(f"Acurácia média na validação cruzada: {mean_validation_accuracy:.4f}")

# Avaliação no conjunto de teste final
y_test_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
final_test_accuracy = accuracy_score(y_test_labels, y_test_pred)
print(f"Acurácia no conjunto de teste: {final_test_accuracy:.4f}")

target_names = ['op. cond. 0', 'op. cond. 1', 'op. cond. 2', 'op. cond. 3', 'op. cond. 4']

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(13,13))
mat = confusion_matrix(y_test_labels, y_test_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={"size": 26},  # Tamanho das anotações no heatmap
            )
plt.xlabel('True label', fontsize=22)  # Tamanho do rótulo do eixo X
plt.ylabel('Predicted label', fontsize=22)  # Tamanho do rótulo do eixo Y
plt.xticks(fontsize=17)  # Tamanho dos ticks do eixo X
plt.yticks(fontsize=17)  # Tamanho dos ticks do eixo Y
string = r'images\ANN_PCA_10.png'
plt.savefig(string)
plt.show()
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window


from sklearn.metrics import classification_report

print(classification_report(y_test_labels, y_test_pred, target_names=target_names))



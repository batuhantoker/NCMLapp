from sklearn import *
import matplotlib.pyplot as plt
from scipy.signal import butter,  lfilter
import numpy as np
from scipy import  signal
import math

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def zero_lag_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data, padlen=15)
    return y

def data_preprocess(emg_data,fs,lowcut,highcut):
    print('Data filtering...')
    emg_ = zero_lag_filter(emg_data, lowcut, highcut, fs, order=4)
    #emg_ = abs(emg_)
    scaler = preprocessing.MinMaxScaler()
    emg_scaled = scaler.fit_transform(emg_)
    return emg_scaled

def features(data, epoch):
    print('Feature extraction...')
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch, :], number_of_segments)
    RMS = np.empty([number_of_segments, data.shape[1]])
    MAV = np.empty([number_of_segments, data.shape[1]])
    IAV = np.empty([number_of_segments, data.shape[1]])
    VAR = np.empty([number_of_segments, data.shape[1]])
    WL = np.empty([number_of_segments, data.shape[1]])
    MF = np.empty([number_of_segments, data.shape[1]])
    PF = np.empty([number_of_segments, data.shape[1]])
    MP = np.empty([number_of_segments, data.shape[1]])
    TP = np.empty([number_of_segments, data.shape[1]])
    SM = np.empty([number_of_segments, data.shape[1]])
    # max_ind = np.empty([number_of_segments, 4])
    for i in range(number_of_segments):
        RMS[i, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        # max_ind [i,:] = RMS[i,:][np.argpartition(RMS[i,:],5, axis=0)]
        MAV[i, :] = np.mean(np.abs(splitted_data[i]), axis=0)
        IAV[i, :] = np.sum(np.abs(splitted_data[i]), axis=0)
        VAR[i, :] = np.var(splitted_data[i], axis=0)
        WL[i, :] = np.sum(np.diff(splitted_data[i], prepend=0), axis=0)
        freq, power = signal.periodogram(splitted_data[i], axis=0)
        fp = np.empty([len(freq), power.shape[1]])
        for k in range(len(freq)):
            fp[k] = power[k, :] * freq[k]
        MF[i, :] = np.sum(fp, axis=0) / np.sum(power, axis=0)  # Mean frequency
        PF[i, :] = freq[np.argmax(power, axis=0)]  # Peak frequency
        MP[i, :] = np.mean(power, axis=0)  # Mean power
        TP[i, :] = np.sum(power, axis=0)  # Total power
        SM[i, :] = np.sum(fp, axis=0)  # Spectral moment
    print('Feature extraction is completed')
    return RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM

def labels(data, epoch):
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0:number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.rint(np.sqrt(np.mean(np.square(splitted_data[i]))))
    return class_value

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)



def classifier(features,labels,k_fold,models):
    Y = labels
    X = features
    number_of_k_fold = k_fold
    random_seed = 42
    outcome = []
    model_names = []
    # Variables for average classification report
    originalclass = []
    classification = []
    for model_name, model in models:
        k_fold_validation = model_selection.KFold(n_splits=number_of_k_fold, random_state=random_seed, shuffle=True)
        results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation,
                                                  scoring='accuracy')
        outcome.append(results)
        model_names.append(model_name)
        output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
        print(output_message)
    print(classification)
    fig = plt.figure()
    fig.suptitle('Machine Learning Model Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(outcome)
    plt.ylabel('Accuracy')
    plt.xlabel('Models')
    ax.set_xticklabels(model_names)
    fig2 = plt.figure()
    plt.show()

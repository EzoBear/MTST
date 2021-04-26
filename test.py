import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import os

class ESC50Data(Dataset):
    def __init__(self, base, df, in_col, out_col):
        self.df = df
        self.data = []
        self.labels = []
        self.c2i={}
        self.i2c={}
        self.categories = sorted(df[out_col].unique())
        for i, category in enumerate(self.categories):
            self.c2i[category]=i
            self.i2c[i]=category
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = os.path.join(base,row[in_col])
            self.data.append(spec_to_image(get_melspectrogram_db(file_path)))
            self.labels.append(self.c2i[row['category']])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    step = 200
    max_size = 701
    spec_scaled = np.resize(spec_scaled,(224,max_size))
    r = spec_scaled[:, 0:max_size-step]
    g = spec_scaled[:, int(step/2):max_size-int(step/2)]
    b = spec_scaled[:, step:max_size]
    return np.dstack((r,g,b)).transpose(2,0,1)

def get_melspectrogram_db(file_path, sr=None, n_fft=4096, hop_length=441, n_mels=224, fmin=20, fmax=20000, top_db=160):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def get_model():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet101_2', pretrained=True)
    model.fc = nn.Linear(2048,50)
    model = nn.DataParallel(model)
    model.cuda()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 1}
    model.load_state_dict(torch.load('weights/'+'fold'+str(fold)+'/'+'weight.pth', map_location=map_location)['model_state_dict'])
    model.eval()
    return model

def get_datalodaer():
    test_data = df[df['fold']==fold]
    test_data = ESC50Data('ESC-50-master/audio', test_data, 'filename', 'category')
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    return test_loader

def print_result(predict, labels):
    print("==============================")
    print('fold : ' + str(fold))
    print("------------------------------")
    print('Precision: %.3f' % precision_score(labels, predict, average='weighted'))
    print('Recall: %.3f' % recall_score(labels, predict, average='weighted'))
    print('Accuracy: %.3f' % accuracy_score(labels, predict))
    print('F1 Score: %.3f' % f1_score(labels, predict, average='weighted'))
    print("==============================")

def save_result(path,file_name,result):
    np.savetxt(path+"/"+file_name, result, delimiter=',', fmt='%i')   # X is an array

if __name__ == "__main__":
    df = pd.read_csv('ESC-50-master/meta/esc50.csv')
    df.head()
    wav, sr = librosa.load('ESC-50-master/audio/1-100032-A-0.wav', sr=None)
    print(f'Sampling rate of the audio is {sr} and length of the audio is {len(wav)/sr} seconds')

    for fold in range(1,6):
        model = get_model()
        test_loader = get_datalodaer()
        trace_labels = []
        trace_predicts = []

        for i, (data, label) in enumerate(test_loader):
            data  = data.type(torch.cuda.FloatTensor)
            label = label.type(torch.cuda.LongTensor)
            predict = model(data)
            trace_labels.append(label.cpu().detach().numpy())
            trace_predicts.append(predict.cpu().detach().numpy())    

        trace_labels = np.concatenate(trace_labels)
        trace_predicts = np.concatenate(trace_predicts).argmax(axis=1)
        
        print_result(trace_predicts, trace_labels)
        save_result(os.getcwd()+"/weights/"+"fold"+str(fold),"result.out", (trace_labels,trace_predicts))


     
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import torch
import numpy as np

def load_processing(filepath, target_sr=16000, trim=False):
    # Load waveform
    if isinstance(filepath, np.ndarray):
        filepath = filepath[0]
    wave, sr = torchaudio.load(filepath)
    
    # Check number of channels (MONO)
    if wave.shape[0] > 1:
        wave = ((wave[0,:] + wave[1,:])/2).unsqueeze(0)
    
    # Check resampling (16 khz)
    if sr != target_sr:
        wave = torchaudio.transforms.Resample(sr,  target_sr)(wave)
        sr = target_sr
    
    # Trim audio to 10 secs
    if trim:
        if wave.shape[1] > sr*10:
            wave = wave[:, :sr*10]
    
    return wave

class TripletDataset(Dataset):
    def __init__(self, config, data_mode='train_df', level=None):
        super().__init__()
        self.config = config
        self.root = self.config['root']

        # Load csv dataset into pandas dataframe
        annotation_path = self.config[data_mode]
        self.dataset = pd.read_csv(annotation_path)
        
        # Extract samples based on diffuculty level: 1 Easy sampling, 2 Hard sampling, 1-2 both 
        if level is not None:
            self.dataset = self.dataset[self.dataset['db'].isin(level)]
        pass
        
        self.dataset.drop_duplicates(inplace=True)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Index extraction
        row = self.dataset.iloc[index]
        
        # Filepath composition
        A_filepath = os.path.join(self.root + row['Anchor'])
        P_filepath = os.path.join(self.root + row['Positive'])
        N_filepath = os.path.join(self.root + row['Negative'])

        # Load and preprocess each file
        A = load_processing(A_filepath, trim=self.config['trim'])
        P = load_processing(P_filepath, trim=self.config['trim'])
        N = load_processing(N_filepath, trim=self.config['trim'])

        return A, P, N
        
    # Zero pad at batch level for waveforms (wav2vec)
    def collate_fn(self, batch):  ## zero padding
        A_waves, P_waves, N_waves = zip(*batch)
        A_waves = self.zero_pad_wav(A_waves)
        P_waves = self.zero_pad_wav(P_waves)
        N_waves = self.zero_pad_wav(N_waves)
        return A_waves, P_waves, N_waves

    def zero_pad_wav(self, wavs):
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        return output_wavs
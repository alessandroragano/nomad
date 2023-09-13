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

        # Needed for SpecgramModel - Dataset level max pad
        self.num_max_frames = int(config['max_length']*config['sampling_rate']//config['hop_length'])
        
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

        # Patch Mel spec (optional for the specgr model)
        if self.config['architecture'] == 'SpecgramModel':
            A, A_lengths = self.mel_specgram_patches(A)
            P, P_lengths = self.mel_specgram_patches(P)
            N, N_lengths = self.mel_specgram_patches(N)
        else:
            A_lengths = None
            P_lengths = None
            N_lengths = None

        return A, P, N, A_lengths, P_lengths, N_lengths
    
    # Zero pad mel spectrograms
    def zero_pad_spec(self, data, n_frames):
        x_zero = torch.zeros((self.num_max_frames//self.config['patch_length'], data.shape[1], data.shape[2], data.shape[3]))
        if n_frames > x_zero.shape[0]:
            data = data[:x_zero.shape[0], ...]
            n_frames = data.shape[0]
        x_zero[:n_frames,...] = data
        return x_zero, n_frames   
        
    # Zero pad at batch level for waveforms (wav2vec)
    def collate_fn(self, batch):  ## zero padding
        A_waves, P_waves, N_waves, _, _, _ = zip(*batch)
        A_waves = self.zero_pad_wav(A_waves)
        P_waves = self.zero_pad_wav(P_waves)
        N_waves = self.zero_pad_wav(N_waves)
        return A_waves, P_waves, N_waves, None, None, None

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

    def mel_specgram_patches(self, wave):
        # Compute mel spectrograms
        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=self.config['sampling_rate'], n_fft=self.config['n_fft'],
                                                            win_length=self.config['win_length'], hop_length=self.config['hop_length'],
                                                            n_mels=self.config['n_mels']
            )(wave)
        
        # Take the log 
        mel_specgram = torch.log10(mel_specgram + np.finfo(float).eps)
        
        # Split into overlapping patches on the time dimension
        mel_specgram_patches = mel_specgram.unfold(dimension=2, size=self.config['patch_length'], step=self.config['patch_length']//2).permute(2, 0, 1, 3)
        
        # Zero pad
        mel_specgram_patches, N_lengths = self.zero_pad_spec(mel_specgram_patches, mel_specgram_patches.shape[0])
        
        # Return mel specgram patches and the length of each spectrogram to avoid processing zeros
        return mel_specgram_patches, N_lengths

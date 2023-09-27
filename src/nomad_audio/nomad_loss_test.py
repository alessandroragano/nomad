import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio
from tqdm import tqdm
from datetime import datetime
from pesq import pesq_batch
from nomad_audio import nomad

with open('src/nomad_audio/se_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# REPRODUCIBILITY 
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SpeechEnhancement():
    def __init__(self):  
        # Find device
        if torch.cuda.is_available:
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'
    
        # Create datasets
        self.train_set, self.train_loader = self.create_dl(config['noisy_train_dir'], config['clean_train_dir'], config['train_bs'], shuffle=True)
        self.valid_set, self.valid_loader = self.create_dl(config['noisy_valid_dir'], config['clean_valid_dir'], config['valid_bs'], shuffle=False)
        self.test_set, self.test_loader = self.create_dl(config['noisy_test_dir'], config['clean_test_dir'], config['test_bs'], shuffle= False)

        # Create loss
        self.mse_loss = nn.MSELoss()

        # Create model
        self.model = Model(n_layers=12)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

    def create_dl(self, noisy_data_dir, clean_data_dir, batch_size, shuffle):
        set = AudioDataset(noisy_data_dir, clean_data_dir)
        dl = DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'], collate_fn=set.collate_fn)
        return set, dl

    def train(self):
        self.model.train()
        total_loss = 0.0

        for id_batch, waves in enumerate(tqdm(self.train_loader)):
            estimate = self.model(waves[0]).to(self.DEVICE)
            clean = waves[1].to(self.DEVICE)

            # Calculate loss
            loss = self.mse_loss(estimate, clean) + config['nomad_weight'] * nomad.forward(estimate, clean)

            # Update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update total loss
            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    
    def eval(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for id_batch, waves in enumerate(tqdm(self.valid_loader)):
                estimate = self.model(waves[0]).to(self.DEVICE)
                clean = waves[1].to(self.DEVICE)

                # Calculate loss
                loss = self.mse_loss(estimate, clean) + config['nomad_weight'] * nomad.forward(estimate, clean)
                total_loss += loss.item()

        return total_loss / len(self.valid_loader)
    
    def test(self, best_model):
        pesq_scores = []
        best_model.eval()

        with torch.no_grad():
            for id_batch, waves in enumerate(tqdm(self.test_loader)):
                estimate = best_model(waves[0])
                clean = waves[1]

                pesq_scores.append(pesq_batch(fs=config['target_sr'], ref=clean.squeeze(1).detach().numpy(), deg=estimate.squeeze(1).detach().numpy(), mode='wb'))
            
            pesq_avg = np.mean([x for x in np.concatenate(pesq_scores) if isinstance(x, float)])
        return pesq_avg
    
    def training_loop(self):
        # Start Training Loop
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        
        self.PATH_DIR = os.path.join('se_models', dt_string)
        if not os.path.isdir(self.PATH_DIR):
            os.makedirs(self.PATH_DIR)

        # Save configuration file used
        dump_path_config = os.path.join(self.PATH_DIR, 'config.yaml')
        with open(f"{dump_path_config}", 'w') as file:
            out_config_file = yaml.dump(config, file)

        best_valid_loss = np.Inf
        EPOCHS = config['num_epochs']

        counter = 0
        for i in range(EPOCHS):
            print('\n')
            train_loss = self.train()
            valid_loss = self.eval()

            if valid_loss < best_valid_loss:
                path_best_model = os.path.join(self.PATH_DIR, 'best_model.pt')
                torch.save(self.model.state_dict(), path_best_model)
                best_valid_loss = valid_loss
                print("Saved Weights Success")
                counter = 0
                best_model = self.model
            else:
                counter += 1

            print(f"COUNTER:  {counter}/{config['patience']}")
            
            if counter > config['patience']:
                print('Stop training, counter greater than patience')
                break
            
            print(f"EPOCHS: {i+1} train_loss : {train_loss}")
            print(f"EPOCHS: {i+1} valid_loss : {valid_loss}")

            # Test set PESQ every x epochs
            if (i+1) % config['test_every'] == 0:
                print('Test set evaluation')
                pesq_avg = self.test(best_model)
                print(f"EPOCHS: {i+1} PESQ : {pesq_avg}")

class AudioDataset(Dataset):
    def __init__(self, noisy_data_dir, clean_data_dir):
        self.noisy_data_dir = noisy_data_dir
        self.noisy_data = os.listdir(noisy_data_dir)
        self.clean_data_dir = clean_data_dir
        self.clean_data = os.listdir(clean_data_dir)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        noisy_audio_path = os.path.join(self.noisy_data_dir, self.noisy_data[idx])
        clean_audio_path = os.path.join(self.clean_data_dir, self.clean_data[idx])
        
        assert self.clean_data[idx] == self.noisy_data[idx]

        noisy_wave, orig_sr = torchaudio.load(noisy_audio_path)
        target_sr = config['target_sr']
        if orig_sr != target_sr:
            noisy_wave = torchaudio.transforms.Resample(orig_sr, target_sr)(noisy_wave)
        
        clean_wave, orig_sr = torchaudio.load(clean_audio_path)
        target_sr = config['target_sr']
        if orig_sr != target_sr:
            clean_wave = torchaudio.transforms.Resample(orig_sr, target_sr)(clean_wave)       

        return noisy_wave, clean_wave

    # Zero pad at batch level for waveforms (wav2vec)
    def collate_fn(self, batch):  ## zero padding
        noisy_wave, clean_wave = zip(*batch)
        noisy_wave = self.zero_pad_wav(noisy_wave)
        clean_wave = self.zero_pad_wav(clean_wave)
        return noisy_wave, clean_wave

    def zero_pad_wav(self, wavs):
        wavs = list(wavs)
        #max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        max_len = 16384 # Same value used in the U-net repo, otherwise use the max in the batch size (see max_len line above)
        output_wavs = []
        for wav in wavs:
            if wav.shape[1] < max_len:
                amount_to_pad = max_len - wav.shape[1]
                padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            else:
                padded_wav = wav[:,:max_len]
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        return output_wavs


# **** Define U-Net for speech enhancement ****
# **** CODE https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement/tree/master ****
class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class Model(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Down Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o

# Testing speech enhancement
se = SpeechEnhancement()
se.training_loop()
from typing import Any
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import math

class TripletModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.ssl_features, emb_dim)
        )
    
    def forward(self, wav, lengths=None):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x_tr = torch.mean(x, 1)
        x = self.embedding_layer(x_tr)
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class Origw2v(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(Origw2v, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
    
    def forward(self, wav, lengths=None):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x_tr = torch.mean(x, 1)
        #x  = torch.nn.functional.normalize(x, dim=1)
        return x_tr

class MosPredictor(nn.Module): 
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.out_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.out_layer(x)
        return x.squeeze(1)

class MosPredictorTriplet(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(MosPredictorTriplet, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.ssl_features, emb_dim)
        )        
        self.out_layer = nn.Linear(emb_dim, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        x = 1+4*torch.sigmoid(self.out_layer(x))
        return x.squeeze(1)

class FeatureEncoder(nn.Module):
    def __init__(self, cnn_1=16, cnn_2=32, cnn_3=64):
        super().__init__()

        self.dropout = nn.Dropout2d(p=0.2)
        
        # Dim = (1 x 48 x 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, cnn_1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_1),
            nn.ReLU(),
        )
        
        # Dim = (16 x 48 x 16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(cnn_1, cnn_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=[24, 8])
        )

        # Dim = (32 x 24 x 8)
        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn_2, cnn_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_2),
            nn.ReLU(),
        )
        
        # Dim = (32 x 24 x 8)
        self.layer4 = nn.Sequential(
            nn.Conv2d(cnn_2, cnn_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=[12, 4])
        )
        
        # Dim = (64 x 12 x 4)
        self.layer5 = nn.Sequential(
            nn.Conv2d(cnn_3, cnn_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=[6, 1])
        )

        self.out_fmap_size = [6, 1]
        self.out_num_filters = cnn_3
    
    def forward(self, x):
        # First block
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        
        # Second block
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)

        # Third block
        x = self.layer5(x)

        # Flatten (384, 1)
        x = x.view(-1, self.out_fmap_size[0] * self.out_fmap_size[1] * self.out_num_filters)
        return x

class TimeModeling(nn.Module):
    def __init__(self, input_size=384, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first, bidirectional=bidirectional)
    
    def forward(self, x, lengths):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)             
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=int(lengths.max().item()))          
        # Dim = (Batch size, Num sequences, 2*hidden_size)
        return x

class LSTMPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, lengths):
        # x = (Batch size, Num seq, Hidden size)
        bs = x.shape[0]
        hidden_size = x.shape[-1]
        new_size = 2
        max_len = int(lengths.max().item())

        # Split forward and backward LSTM into a new dimension
        x = x.view(bs, max_len, new_size, hidden_size//2)

        # Extract the last valid sequence for forward LSTM
        lstm_forward = x[torch.arange(x.shape[0]), lengths.type(torch.long)-1, 0, :]

        # Extract first sequence for backward (which corresponds to the last hidden state)
        lstm_backward = x[:,0,1,:]

        # Get output feature vector
        x = torch.cat([lstm_forward, lstm_backward], dim=1)

        return x

class SpecgramModel(nn.Module):
    def __init__(self, cnn_1=16, cnn_2=32, cnn_3=64, input_size=384, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True, emb_dim=256):
        super().__init__()
        self.feature_encoder = FeatureEncoder(cnn_1, cnn_2, cnn_3)
        self.time_model = TimeModeling(input_size, hidden_size, num_layers, batch_first, bidirectional)
        self.time_pooling = LSTMPooling()
        self.embedding_layer = nn.Sequential(
            #nn.ReLU(), 
            nn.Linear(hidden_size*2, emb_dim)
        )

    def forward(self, x, lengths):
        # Pack padded sequence
        lengths = lengths.cpu()
        batch_max_len = int(lengths.max().item())
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Cnn encoder
        x = self.feature_encoder(x_packed.data)

        # Pad packed sequence
        x_packed = PackedSequence(x, batch_sizes=x_packed.batch_sizes, sorted_indices=x_packed.sorted_indices, unsorted_indices=x_packed.unsorted_indices)
        x, _ = pad_packed_sequence(x_packed, batch_first=True, total_length=batch_max_len)        

        # LSTM
        x = self.time_model(x, lengths)

        # Pool last hidden state forward and first hidden state for backward
        x = self.time_pooling(x, lengths)

        # Get embedding layer for triplet
        x = self.embedding_layer(x)

        # L2 normalization
        x = torch.nn.functional.normalize(x, dim=1)
        return x

# # Test feature extractor
#path = '/media/alergn/hdd/datasets/audio/speech/LibriSpeech/train-clean-100-wav/911/130578/911-130578-0000.wav'

# # Load audio
# import torchaudio
# wav, _ = torchaudio.load(path)
# melspec = torchaudio.transforms.MelSpectrogram(n_mels=48)(wav)
# log_mel = torch.log10(melspec)
# patches = log_mel.unfold(dimension=2, size=16, step=8).permute(2, 0, 1, 3)

# #wav = torchaudio.transforms.Resample(new_freq=8000)(wav)

# # Create feature extractor
# #conv_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
# model = SpecgrModel()
# x = torch.cat([patches.unsqueeze(0), patches.unsqueeze(0)])
# x = model(x, torch.Tensor([139, 100]))

# ************ MODELS THAT DIDN'T SHOW ANY IMPROVEMENTS BELOW ***********
# ***********************************************************************

class QualityEmbeddings(nn.Module):
    def __init__(self, context_dim=512, emb_dim=256):
        super().__init__()

        self.embeddings = nn.Sequential(
            nn.ReLU(),
            nn.Linear(context_dim, emb_dim)
        )
    
    def forward(self, x):
        x = self.embeddings(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

class MosPredictorTriplet2(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(MosPredictorTriplet2, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = QualityEmbeddings2(ssl_out_dim, emb_dim)
        self.out_layer = nn.Linear(emb_dim, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        x = 1+4*torch.sigmoid(self.out_layer(x))
        return x.squeeze(1)


class QualityEmbeddings2(nn.Module):
    def __init__(self, context_dim=512, emb_dim=256):
        super().__init__()

        self.embeddings = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, emb_dim),
        )
    
    def forward(self, x):
        x = self.embeddings(x)
        return x    

class QualityNet(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super().__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.ssl_features, emb_dim)
        )

        self.mos_net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(),

            # MOS prediction
            nn.Linear(32, 1)
        )
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        x = self.mos_net(x)
        return x.squeeze(1)
    
class PairsModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(PairsModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.embedding_layer(x)
        #x = torch.nn.functional.normalize(x, dim=1)
        return x.squeeze(1)

class MultiTaskModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MultiTaskModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_triplet = nn.Linear(4*ssl_out_dim, 256)
        self.out_layer = nn.Linear(self.ssl_features + 256, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']

        # Features from 6 to 10 layers (where is phonetic informaton)
        middle_layers = torch.cat([ torch.mean(tr_layers[0], dim=0) for tr_layers in res['layer_results'][0:3]], 1)
        triplet_features = self.embedding_triplet(torch.nn.functional.relu(middle_layers)) 
        triplet_features = torch.nn.functional.normalize(triplet_features)

        tr_avg_pool = torch.mean(x, 1)
        out_features = torch.cat([tr_avg_pool, triplet_features], dim=1)
        x = self.out_layer(out_features)

        return x.squeeze(1), triplet_features

class MultiTaskPairModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MultiTaskPairModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.mos_layer = nn.Linear(self.ssl_features , 1)
        self.rank_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        #x = res['x']
        # Take first transformer layer for multi task learning
        x = res['layer_results'][1][0].permute(1, 0, 2)

        # Context representations 
        tr_avg_pool = torch.mean(x, 1)

        # Multi task output layers mos, rank
        mos_p = self.mos_layer(tr_avg_pool)
        rank_p = self.rank_layer(tr_avg_pool)
        return mos_p.squeeze(1), rank_p.squeeze(1)

class MultiTaskPairModelBce(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MultiTaskPairModelBce, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.out_layer = nn.Linear(self.ssl_features , 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']

        # Context representations 
        tr_avg_pool = torch.mean(x, 1)

        # Multi task output layers mos, rank
        mos_p = self.out_layer(tr_avg_pool)
        return mos_p.squeeze(1)


class CnnEncoder(nn.Module):
    def __init__(self, cnn_1=16, cnn_2=32, cnn_3=64):
        super().__init__()

        self.dropout = nn.Dropout2d(p=0.2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, cnn_1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=[24, 7])
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(cnn_1, cnn_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=[12, 5])

        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn_2, cnn_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_3),
            nn.ReLU(),
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(cnn_3, cnn_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=[6, 3])
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(cnn_3, cnn_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_3),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(cnn_3, cnn_3, kernel_size=3, stride=1, padding=(1, 0)),
            nn.BatchNorm2d(cnn_3),
            nn.ReLU()            
        )
        self.out_fmap_size = [6, 1]
        self.out_num_filters = cnn_3
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(self.layer3(x))
        x = self.dropout(self.layer4(x))
        x = self.dropout(self.layer5(x))
        x = self.layer6(x)

        # Flatten 
        x = x.view(-1, self.out_fmap_size[0] * self.out_fmap_size[1] * self.out_num_filters)
        return x

class FrameWiseMOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_enc = CnnEncoder()
        self.fc_mp = nn.Linear(384, 1)
    
    def forward(self, x, lengths):
        # Pack padded sequence
        batch_max_len = x.shape[1]
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Cnn encoder
        x = self.cnn_enc(x_packed.data)

        # Pad packed sequence
        x_packed = PackedSequence(x, batch_sizes=x_packed.batch_sizes, sorted_indices=x_packed.sorted_indices, unsorted_indices=x_packed.unsorted_indices)
        x, _ = pad_packed_sequence(x_packed, batch_first=True, total_length=batch_max_len)        

        # Extract one sequence (torch.max acts as a selector and backpropagation will only work on the extracted sequence)
        mp_features = torch.max(x, 1)[0]
        x = self.fc_mp(mp_features)
        return x.squeeze(1)

class FrameWiseMOSmtl(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_enc = CnnEncoder()
        self.mos_layer = nn.Linear(384, 1)
        self.rank_layer = nn.Linear(384, 1)
    
    def forward(self, x, lengths):
        # Pack padded sequence
        batch_max_len = x.shape[1]
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Cnn encoder
        x = self.cnn_enc(x_packed.data)

        # Pad packed sequence
        x_packed = PackedSequence(x, batch_sizes=x_packed.batch_sizes, sorted_indices=x_packed.sorted_indices, unsorted_indices=x_packed.unsorted_indices)
        x, _ = pad_packed_sequence(x_packed, batch_first=True, total_length=batch_max_len)        

        # Extract one sequence (torch.max acts as a selector and backpropagation will only work on the extracted sequence)
        mp_features = torch.max(x, 1)[0]
        
        mos = self.mos_layer(mp_features)
        rank = self.rank_layer(mp_features)
        return mos.squeeze(1), rank.squeeze(1)

class Transpose(nn.Module):
    def __init__(self, transpose_dim=-2):
        super().__init__()
        self.transpose_dim = transpose_dim
    
    def forward(self, x):
        return x.transpose(self.transpose_dim, -1)

class FeatureExtractor2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.num_layers = 20
        self.num_filters = 16
        self.filter_size = 15
        self.cnn_encoder = nn.ModuleList()
        self.batchnorm_encoder = nn.ModuleList()

        self.in_ch =  [1] + [(i + 1) * self.num_filters for i in range(self.num_layers-1)]
        self.out_ch = [(i + 1) * self.num_filters for i in range(self.num_layers)]
        
        nchan = self.num_filters
        for i in range(self.num_layers):
                    if i==0:
                        chin = 1
                    else:
                        chin = nchan
                    if (i+1)%4==0:
                        nchan = nchan*2
                    self.cnn_encoder.append(nn.Conv1d(chin,nchan,self.filter_size,padding=self.filter_size//2))
                    self.batchnorm_encoder.append(nn.BatchNorm1d(nchan))

    def forward(self, x):
        input = x
        
        for i in range(self.num_layers):
            x = self.cnn_encoder[i](x)
            x = self.batchnorm_encoder[i](x)
            x = torch.nn.functional.leaky_relu(x,0.1)
            if (i+1)%4==0:
                x = x[:,:,::2]
        
        x = torch.sum(x,dim=(2))/x.shape[2] # average by channel dimension
        
        return x

class FeatureExtractor(nn.Module):
    # conv_layers will be a list of tuples where each tuple includes (n_ch, k, stride)
    def __init__(self, conv_layers, dropout=0.2):
        super().__init__()

        def init_conv(n_in, n_out, k, stride):
            conv_layer = nn.Conv1d(n_in, n_out, k, stride=stride)
            nn.init.kaiming_normal_(conv_layer.weight)
            return conv_layer        

        def conv_block(n_in, n_out, k, stride, layer_norm=False):
            block = nn.Sequential(
                init_conv(n_in, n_out, k, stride),
                nn.Dropout(p=dropout),
                #nn.Sequential(
                #    Transpose(), 
                #    nn.LayerNorm(n_ch, elementwise_affine=True),
                #    Transpose()
                #),
                nn.ReLU()
            )
            return block
        
        # Create convolutional blocks
        self.conv_layers = nn.ModuleList()
        
        # First layer 1 channel
        n_in = 1
        
        for id_conv_layer, settings_conv_layer in enumerate(conv_layers):
            # Extract settings
            (n_ch, k, stride) = settings_conv_layer

            # Add blocks to ModuleList
            self.conv_layers.append(
                conv_block(n_in, n_ch, k, stride)
            )

            # Update input channels for next layer
            n_in = n_ch
    
    def forward(self, x):   
        # Add channel dimension to mono signal
        #x = x.unsqueeze(1)

        # Pass conv layers
        for conv in self.conv_layers:
            x = conv(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ContextEncoder(nn.Module):

    def __init__(self,  d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    #     self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.linear.bias.data.zero_()
    #     self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        #src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TripletModel2(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel2, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = QualityEmbeddings2(context_dim=ssl_out_dim, emb_dim=emb_dim)
    
    def forward(self, wav, lengths=None):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x_tr = torch.mean(x, 1)
        x = self.embedding_layer(x_tr)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

class MyModel(nn.Module):
    def __init__(self, conv_layers, d_model=512, nhead=8, d_hid=3072, nlayers=6, emb_dim=256):
        super().__init__()

        self.feature_encoder = FeatureExtractor(conv_layers)
        self.context_encoder = ContextEncoder(d_model, nhead, d_hid, nlayers)
        self.quality_embeddings = QualityEmbeddings(d_model, emb_dim)

    def forward(self, x):
        # Feature extraction CNN
        x = self.feature_encoder(x)

        # Context representations Transformer
        x = self.context_encoder(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        # Remove time dimension with the mean
        x = torch.mean(x, dim=2)

        # Embedding vec
        x = self.quality_embeddings(x)
        return x

# This architecture uses the backbone feature extractor of w2v and train a transformer from scratch with embedding layer
class SQUAD(nn.Module):
    def __init__(self, ssl_model, d_model=512, nhead=8, d_hid=3072, nlayers=4, dropout=0.5, emb_dim=256):
        super().__init__()
        self.ssl_model = ssl_model
        self.context_encoder = ContextEncoder(d_model, nhead, d_hid, nlayers, dropout)
        self.quality_embeddings = QualityEmbeddings(d_model, emb_dim)
    
    def forward(self, x):
        # Feature extraction pre-trained CNN
        x = x.squeeze(1)  ## [batches, audio_len]
        ssl_features = self.ssl_model(x, mask=False, features_only=True)
        x = ssl_features['features']        
        
        # Context encoder transformer (from scratch), take the mean to remove time dimension
        x = self.context_encoder(x)
        x = torch.mean(x, 1)

        # Quality embeddings
        x = self.quality_embeddings(x)

        return x

class SQUAD2(nn.Module):
    def __init__(self, conv_layers, d_model=512, emb_dim=256):
        super().__init__()
        #self.cnn_backbone = FeatureExtractor(conv_layers)
        self.cnn_backbone = FeatureExtractor2()
        self.quality_embeddings = QualityEmbeddings2(d_model, emb_dim)
    

    def forward(self, x):
        # Feature extraction pre-trained CNN
        #x = x.squeeze(1)  ## [batches, audio_len]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.cnn_backbone(x)       
        
        # Pooling over time dimension (get audio embeddings)
        # x = torch.mean(x, 2)

        # Quality embeddings net
        x = self.quality_embeddings(x)

        # Normalize row-wise each embedding vector
        x =  torch.nn.functional.normalize(x, dim=1)
        return x
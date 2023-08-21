import torch
import torch.nn as nn
import fairseq
import yaml
from src.dataloader.triplet_dataloader import TripletDataset, load_processing
from src.models.networks import TripletModel, MyModel, Origw2v, SQUAD, SQUAD2, TripletModel2, SpecgramModel
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import random
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import math
from scipy.optimize import curve_fit
from scipy.spatial.distance import cosine
import torch.optim.lr_scheduler as lr_scheduler
import torchaudio
from scipy.spatial.distance import cdist
sns.set_style('darkgrid')

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

class Training():
    def __init__(self, config_file):
    
        # Configuration file loading
        with open(config_file) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Find device
        if torch.cuda.is_available:
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'
        print(f'Device: {self.DEVICE}')
        #self.DEVICE = 'cpu'
        
        if self.config['architecture'] == 'w2v': 
            # Load SSL model if using wav2vec
            CHECKPOINT_PATH = self.config['checkpoint_path']
            SSL_OUT_DIM = self.config['ssl_out_dim']
            EMB_DIM = self.config['emb_dim']

            w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
            ssl_model = w2v_model[0] 
            ssl_model.remove_pretraining_modules()

            if self.config['eval_w2v']:
                self.model = Origw2v(ssl_model, SSL_OUT_DIM)
            else:
                self.model = TripletModel(ssl_model, SSL_OUT_DIM, EMB_DIM)
            self.model.to(self.DEVICE)

            # if self.config['pretrain']:
            #     self.model.load_state_dict(torch.load(self.config['pt_easy_model']))

            # Choose if you want to 1) Freeze only ConvNet 2) Freeze ConvNet + Transformer 3) Finetune the entire network (default behaviour)
            # Freeze only ConvNet
            if self.config['experiment_name'] == 'Training':
                if self.config['freeze_convnet']:
                    self.model.ssl_model.feature_extractor.requires_grad_(False)
                
                # Freeze both ConvNet and Transformer (no finetuning)
                if self.config['freeze_all']:
                    self.model.ssl_model.feature_extractor.requires_grad_(False)
                    self.model.ssl_model.encoder.requires_grad_(False)
        
        # elif self.config['architecture'] == 'mymodel':
        #     conv_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
        #     self.model = MyModel(conv_layers)
        #     self.model.to(self.DEVICE)
        
        # elif self.config['architecture'] == 'SQUAD':
        #     # Load SSL model if using wav2vec
        #     CHECKPOINT_PATH = self.config['checkpoint_path']
        #     SSL_OUT_DIM = self.config['ssl_out_dim']
        #     EMB_DIM = self.config['emb_dim']

        #     w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
        #     ssl_model = w2v_model[0]
        #     ssl_model.remove_pretraining_modules()     

        #     # Create SQUAD model (default parameters)
        #     self.model = SQUAD(ssl_model)
        #     self.model.to(self.DEVICE)
            
        #     # Freeze backbone
        #     self.model.ssl_model.feature_extractor.requires_grad_(False)

        # elif self.config['architecture'] == 'SQUAD2':
        #     EMB_DIM = self.config['emb_dim']

        #     # Create SQUAD2 model (default parameters)
        #     conv_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
        #     self.model = SQUAD2(conv_layers)
        #     self.model.to(self.DEVICE)
        
        elif self.config['architecture'] == 'SpecgramModel':
            self.model = SpecgramModel()
            self.model.to(self.DEVICE)
        

        if self.config['experiment_name'] == 'Training':
            # Create dataloaders (start with level 1)
            self.current_level = self.config['current_level']
            self.train_set = TripletDataset(self.config, data_mode='train_df', level=self.current_level)
            if self.config['architecture'] == 'w2v':
                collate_fn = self.train_set.collate_fn
            else:
                collate_fn = None
            self.train_loader = DataLoader(self.train_set, batch_size=self.config['train_bs'], shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn)
            self.valid_set = TripletDataset(self.config, data_mode='valid_df', level=self.current_level)
            self.valid_loader = DataLoader(self.valid_set, batch_size=self.config['val_bs'], shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn)

            # Create loss
            self.criterion = nn.TripletMarginLoss(margin=self.config['margin'])
            #self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(dim=1), margin=self.config['margin'])
            
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

            # Create optimizer
            # Set transformer learning rate small

            if self.config['architecture'] == 'w2v':
                if self.config['freeze_convnet']:
                    params_names_embeddings = [f'embeddings.{j}.weight' for j in range(7)] + [f'embeddings.{j}.bias' for j in range(7)]
                    params_pt = [param for name, param in self.model.named_parameters() if name not in params_names_embeddings]
                    params_embeddings = [param for name, param in self.model.named_parameters() if name in params_names_embeddings]                 
                    # Overwrite optimizer
                    self.optim = torch.optim.Adam([
                        {'params': params_pt, 'lr': 1e-5},
                        {'params': params_embeddings}
                    ], lr=self.config['lr'])
        
        # Create learning rate scheduler
            self.lr_scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=self.config['lr_decay_factor'])

        # Upload model if starting from a level greater than 1
        # if len(self.current_level) == 1:
        #     if self.current_level == 2:
        #         self.model.load_state_dict(torch.load(self.config['path_level_1']))
        #     elif self.current_level == 3:
        #         self.model.load_state_dict(torch.load(self.config['path_level_2']))
        #     elif self.current_level == 4:
        #         self.model.load_state_dict(torch.load(self.config['path_level_3']))

    def train(self, model, dataloader, optimizer, criterion):
        model.train()
        total_loss = 0.0

        for batch_index, (A, P, N, A_lengths, P_lengths, N_lengths) in enumerate(tqdm(dataloader)):

            A, P, N, A_lengths, P_lengths, N_lengths = A.to(self.DEVICE), P.to(self.DEVICE), N.to(self.DEVICE), A_lengths, P_lengths, N_lengths

            A_embs = model(A, A_lengths)
            P_embs = model(P, P_lengths)
            N_embs = model(N, N_lengths)

            loss = criterion(A_embs, P_embs, N_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def eval(self, model, dataloader, criterion):

        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_index, (A, P, N, A_lengths, P_lengths, N_lengths) in enumerate(tqdm(dataloader)):

                A, P, N, A_lengths, P_lengths, N_lengths = A.to(self.DEVICE), P.to(self.DEVICE), N.to(self.DEVICE), A_lengths, P_lengths, N_lengths

                A_embs = model(A, A_lengths)
                P_embs = model(P, P_lengths)
                N_embs = model(N, N_lengths)

                loss = criterion(A_embs, P_embs, N_embs)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def training_loop(self):
        # Start Training Loop
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.PATH_DIR = os.path.join('out_models', self.config['out_dir'], dt_string)
        if not os.path.isdir(self.PATH_DIR):
            os.makedirs(self.PATH_DIR)

        # Save configuration file used
        dump_path_config = os.path.join(self.PATH_DIR, 'config.yaml')
        with open(f"{dump_path_config}", 'w') as file:
            out_config_file = yaml.dump(self.config, file)

        best_valid_loss = np.Inf
        EPOCHS = self.config['num_epochs']

        counter = 0
        for i in range(EPOCHS):
            train_loss = self.train(self.model, self.train_loader, self.optim, self.criterion)
            valid_loss = self.eval(self.model, self.valid_loader, self.criterion)

            if valid_loss < best_valid_loss:
                path_best_model = os.path.join(self.PATH_DIR, 'best_model.pt')
                torch.save(self.model.state_dict(), path_best_model)
                best_valid_loss = valid_loss
                print("Saved Weights Success")
                counter = 0
            else:
                counter += 1
            
            # Check learning rate update
            if (counter + 1) % self.config['lr_decay_step'] == 0:
                self.lr_scheduler.step()

            print(f"COUNTER:  {counter}/{self.config['patience']}")
            print(f'LR: {self.lr_scheduler.get_last_lr()}')
            
            #print(f'CURRENT LEVEL: {self.current_level}')

            if counter > self.config['patience']:
                if self.current_level is not None:
                    if self.current_level < 4:
                        print(f"Change Level")
                        # Change dataloaders with new level
                        self.current_level += 1
                        self.train_set = TripletDataset(self.config, data_mode='train_df', level=self.current_level)
                        self.train_loader = DataLoader(self.train_set, batch_size=self.config['train_bs'], shuffle=True, num_workers=self.config['num_workers'], collate_fn=self.train_set.collate_fn)
                        self.valid_set = TripletDataset(self.config, data_mode='valid_df', level=self.current_level)
                        self.valid_loader = DataLoader(self.valid_set, batch_size=self.config['val_bs'], shuffle=False, num_workers=self.config['num_workers'], collate_fn=self.valid_set.collate_fn)
                        
                        # Reset
                        counter = 0
                        best_valid_loss = 99999

                        # Load best model of the previous level
                        self.model.load_state_dict(torch.load(path_best_model))

                    elif self.current_level == 4:
                        print('Stop Training, Level 4 not learning')
                        break
                else:
                    print('Stop training, counter greater than patience')
                    break
            
            print(f"EPOCHS: {i+1} train_loss : {train_loss}")
            print(f"EPOCHS: {i+1} valid_loss : {valid_loss}")
            print('\n')
    
    # Don't use dataloader since we don't need to load positive and negative samples
    def get_embeddings_csv(self, model, anc_names, root=False):
        anc_names_arr = np.array(anc_names)
        embeddings = []

        model.eval()
        with torch.no_grad():
            for i, filename_anchor in enumerate(tqdm(anc_names_arr)):
                if root:
                    filepath = os.path.join(root, filename_anchor)
                else:
                    filepath = filename_anchor
                
                anc_speech = load_processing(filepath, trim=False)
                lengths = None
                
                # Get log mel spectrograms
                if self.config['architecture'] == 'SpecgramModel':
                    anc_speech = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.config['sampling_rate'],
                    n_fft=self.config['n_fft'],
                    win_length=self.config['win_length'],
                    hop_length=self.config['hop_length'],
                    n_mels=self.config['n_mels']
                )(anc_speech)
                    anc_speech = torch.log10(anc_speech + np.finfo(float).eps)

                    # Split into patches
                    anc_speech = anc_speech.unfold(dimension=2, size=self.config['patch_length'], step=self.config['patch_length']//2).permute(2, 0, 1, 3)
                    lengths = anc_speech.shape[0]

                    # Store lengths into a tensor 
                    if not torch.is_tensor(lengths):
                        lengths = torch.Tensor([lengths])
                    
                    # Add batch size dimension (always equal to 1 during test)
                    anc_speech.unsqueeze_(0)
                
                anc_speech = anc_speech.to(self.DEVICE)
                anc_embeddings = model(anc_speech, lengths)
                embeddings.append(anc_embeddings.squeeze().cpu().detach().numpy())

            embeddings = np.array(embeddings)
            embeddings = pd.DataFrame(embeddings)
            df_emb = pd.concat([anc_names.reset_index(), embeddings], axis=1).drop('index', axis=1)
        
        return df_emb

    def euclidean_dist(self, embeddings, anc_emb_arr):
        dist = np.sqrt(np.dot(embeddings - anc_emb_arr, (embeddings - anc_emb_arr).T))
        return dist

    def order_three(self, x, a, b, c, d):
        return a*x + b*x**2 + c*x**3 + d

    def eval_anchor_embeddings(self, model_path):
        # Load model weights
        if self.config['eval_mos_pred']:
            pt_model = torch.load(self.config['mos_pred_model'], map_location=self.DEVICE)
            model_state_dict = self.model.state_dict()
            keys = list(model_state_dict.keys())
            layers = {k: v for k, v in pt_model.items() if k in keys}                      
            model_state_dict.update(layers)
            self.model.load_state_dict(model_state_dict)
        elif not self.config['eval_w2v']:
            self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()

        # Get dataframe of anchor embeddings of the full dataset
        test_data = pd.read_csv(self.config['test_db_file'])

        if self.config['db'] is not None:
            test_data = test_data[test_data['db'].isin(self.config['db'])]

        # PRINT LOG INFO
        db_test = self.config['db']
        if self.config['conds'] is not None:
            conds = self.config['conds']
            print(f'Testing DB: {db_test}, conds: {conds}')
            test_data = test_data[test_data['condition'].str.contains('|'.join(conds))]

        # Get reference files from another clean speech dataset
        ref_files = pd.DataFrame(os.listdir(self.config['non_match_dir']))
        ref_files.columns = ['reference']
        ref_files['reference'] = [os.path.join(self.config['non_match_dir'], x) for x in ref_files['reference']]
        
        # Loop over each database
        db_groups = test_data.groupby('db')
        
        ref_embeddings = self.get_embeddings_csv(self.model, ref_files)
        ref_embeddings.set_index('reference', inplace=True)

        for db_name, db in db_groups:
            print(db_name)
            df_emb = self.get_embeddings_csv(self.model, db['filepath'], root=self.config['test_db_wav'])
            
            test_embeddings = df_emb.set_index('filepath')
            test_names = df_emb.merge(db, on='filepath')[['filepath', 'condition', 'mos']]         
            
            euclid_dist = cdist(test_embeddings, ref_embeddings)
            avg_dist_nmr = np.mean(euclid_dist, axis=1)
            names = test_embeddings.index

            # Add distances 
            df_dist = pd.DataFrame({'filepath': names, 'Distance': avg_dist_nmr})
            df_dist = df_dist.merge(test_names, on='filepath')
            df_dist = df_dist.groupby('condition').mean()

            # Compute third order poly mapping
            #df_grouped = df_dist.groupby('condition')['mos', 'Distance'].mean()
            popt3, _ = curve_fit(self.order_three, df_dist['Distance'].values, df_dist['mos'].values)
            a3, b3, c3, d3 = popt3
            df_dist['Distance_map'] = df_dist['Distance'].apply(lambda x: self.order_three(x,a3,b3,c3,d3))

            # Scatter plot MOS - Embedding Distances
            sns.scatterplot(data=df_dist, x='mos', y='Distance_map')
            plt.xlabel('Actual MOS')
            plt.ylabel(f'Dist w.r.t. clean embeddings')
            plt.xlim([1, 5])
            plt.ylim([1, 5])
            plt.tight_layout()
            out_dir = '/'.join(self.config['rank_model_path'].split('/')[:-1])
            plt.savefig(os.path.join(out_dir, f'{db_name}_embeddings.png'))
            plt.close()
            
            # Performance Evaluation    
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['mos'])
            print(f'SRCC: {np.round(SRCC, 2)}')
            SRCC, _ = spearmanr(df_dist['Distance_map'], df_dist['mos'])
            print(f'SRCC 3rd map: {np.round(SRCC, 2)}')

            PCC, _ = pearsonr(df_dist['Distance'], df_dist['mos'])
            print(f'PCC: {np.round(PCC, 2)}')

            PCC, _ = pearsonr(df_dist['Distance_map'], df_dist['mos'])
            print(f'PCC 3rd map: {np.round(PCC, 2)}')
        return df_dist

    def eval_degr_level(self, model_path):
        # Load model weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Get dataframe of anchor embeddings of the full dataset
        valid_set = TripletDataset(self.config, data_mode='valid_df', level=self.config['current_level'])
        df_emb = self.get_embeddings_csv(self.model, valid_set.dataset['Anchor'], root=self.config['root'])

        # Get anchor embeddings of the query
        #query_name = df_emb['Anchor'].iloc[idx_query]
        #query_filepath = os.path.join(self.config['root'], query_name)
        # query_filepath = self.config['query_filepath']
        # query_name = 'Clean Speech.wav'
        # query_speech = self.valid_set.load_processing(query_filepath)

        # self.model.eval()
        # with torch.no_grad():
        #     query_speech = query_speech.to(self.DEVICE)
        #     query_embeddings = self.model(query_speech)
        #     query_embeddings = query_embeddings.detach().cpu().numpy()


        # Get random reference files
        ref_files = pd.DataFrame(os.listdir(self.config['non_match_dir']))
        ref_files.columns = ['reference']
        ref_files['reference'] = [os.path.join(self.config['non_match_dir'], x) for x in ref_files['reference']]
         
        ref_embeddings = self.get_embeddings_csv(self.model, ref_files)
        ref_avg_embeddings = ref_embeddings.iloc[:,1:].mean()

        anc_emb_arr = df_emb.iloc[:, 1:].to_numpy()
        anc_names = df_emb['Anchor']

        distance = []
        names = []

        for i in range(anc_emb_arr.shape[0]):
        #for i in range(200):
            test_embeddings = anc_emb_arr[i:i+1,:].T.squeeze(1)
            dist = self.euclidean_dist(ref_avg_embeddings, test_embeddings)
            #dist = cosine(query_embeddings, anc_emb_arr[i:i+1,:])
            distance = np.append(distance, dist)
            names.append(anc_names[i])

        df_dist = pd.DataFrame({'Anchor': names, 'Distance': distance})
        df_dist.sort_values(by='Distance', inplace=True)
        #df_dist['SNR'] = [float(x.split('_SNRdb_', 1)[1][:4].replace('_','')) for x in df_dist['Anchor']]
        #df_dist['SNR_dist'] = abs(df_dist.iloc[idx_query]['SNR'] - df_dist['SNR'])
        df_dist['condition'] = [x.split('_')[1] + ' ' + x.split('_')[2].split('.')[0] for x in df_dist['Anchor']]
        order = df_dist.groupby('condition')['Distance'].mean().sort_values().index
        sns.boxplot(df_dist, x='condition', y='Distance', order=order, showmeans=True)
        plt.xticks(rotation=90)
        #query_condition = query_name.split('_')[1] + query_name.split('_')[2].split('.')[0]
        plt.ylabel(f'Dist w.r.t non-matching reference')
        plt.xlabel('Condition')
        plt.tight_layout()
        plt.savefig('box_plot.pdf')
        plt.savefig('box_plot.png')
        return df_dist

    def eval_monotonocity(self, model_path):
        if not self.config['eval_w2v']:
            self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()

        # Get non matching references
        ref_files = pd.DataFrame(os.listdir(self.config['non_match_dir']))
        ref_files.columns = ['reference']
        ref_files['reference'] = [os.path.join(self.config['non_match_dir'], x) for x in ref_files['reference']]
         
        ref_embeddings = self.get_embeddings_csv(self.model, ref_files)
        ref_embeddings.set_index('reference', inplace=True)

        ref_avg_embeddings = ref_embeddings.iloc[:,1:].mean()

        # Get test data
        test_data = pd.read_csv(self.config['test_mono_data'])

        # Loop over each degradation
        test_data_group = test_data.groupby('Degradation')
        
        # Store for scatterplot
        degr_names = []
        degr_data = []
        srcc_scores = []

        for deg_name, deg_data in test_data_group:
            df_emb = self.get_embeddings_csv(self.model, deg_data['filepath_deg'], root=self.config['test_mono_wav'])
            #test_cond_embeddings = df_emb.merge(deg_data[['filepath_deg', 'Condition']], on='filepath_deg').iloc[:,1:].groupby('Condition').mean()
            #df_emb = df_emb.merge(deg_data[['filepath_deg', 'Condition']], on='filepath_deg').iloc[:,1:]
            test_cond_gt = deg_data.groupby('Condition').mean()     

            test_embeddings = df_emb.set_index('filepath_deg')
            test_names = df_emb.merge(deg_data, on='filepath_deg')[['filepath_deg', 'Condition']]         

            euclid_dist = cdist(test_embeddings, ref_embeddings)
            avg_dist_nmr = np.mean(euclid_dist, axis=1)
            names = test_embeddings.index

            # Add distances 
            df_dist = pd.DataFrame({'filepath_deg': names, 'Distance': avg_dist_nmr})
            df_dist = df_dist.merge(test_names, on='filepath_deg')
            df_dist = df_dist.groupby('Condition').mean().reset_index()

            # distance = []
            # names = []

            # for id_cond, test_emb in df_emb.iterrows():
            # #for i in range(200):
            #     dist = self.euclidean_dist(ref_avg_embeddings, test_emb.iloc[:-1].values)
            #     #dist = cosine(ref_avg_embeddings, test_emb.values)
            #     distance = np.append(distance, dist)
            #     names.append(test_emb.iloc[-1])
            
            # Add distances 
            # df_dist = pd.DataFrame({'Condition': names, 'Distance': avg_dist_nmr})
            # df_dist = df_dist.merge(test_cond_gt, on='Condition')
            # df_dist = df_dist.groupby('Condition').mean().reset_index()
            df_dist.sort_values(by='Distance', inplace=True)
            
            # Store for scatterplot
            degr_names.append([deg_name]*len(df_dist))
            degr_data.append(df_dist)

            # Performance Evaluation    
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['Condition'])
            srcc_scores.append([SRCC]*len(df_dist))
            print(f'Degradation: {deg_name}')
            print(f'SRCC: {np.round(SRCC, 2)}')

        pass

    def eval_anchor_embeddings2(self, model_path):
        # Load model weights
        if self.config['eval_mos_pred']:
            pt_model = torch.load(self.config['mos_pred_model'], map_location=self.DEVICE)
            model_state_dict = self.model.state_dict()
            keys = list(model_state_dict.keys())
            layers = {k: v for k, v in pt_model.items() if k in keys}                      
            model_state_dict.update(layers)
            self.model.load_state_dict(model_state_dict)
        elif not self.config['eval_w2v']:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Get dataframe of anchor embeddings of the full dataset
        test_data = pd.read_csv(self.config['test_db_file'])

        # Get random reference files
        df_ref = pd.read_csv(self.config['csv_clean'])
        ref_files = df_ref.groupby('reference').sample(n=1)['reference'].sample(2)
        
        # Loop over each database
        db_groups = test_data.groupby('db')
        
        ref_embeddings = self.get_embeddings_csv(self.model, ref_files)
        ref_avg_embeddings = ref_embeddings.iloc[:,1:].mean()
        
        for db_name, db in db_groups:
            print(db_name)
            df_emb = self.get_embeddings_csv(self.model, db['filepath'], root=self.config['test_db_wav'])
            
            test_embeddings = df_emb.set_index('filepath')
            test_names = df_emb.merge(db, on='filepath')[['filepath', 'condition', 'mos']]
            
            #test_cond_embeddings = df_emb.merge(db[['filepath', 'condition']], on='filepath').iloc[:,1:].groupby('condition').mean()
            #test_cond_gt = db.groupby('condition')['mos'].mean()            
            
            distance = []
            names = []

            for filename, test_emb in test_embeddings.iterrows():
            #for i in range(200):
                dist = self.euclidean_dist(ref_avg_embeddings, test_emb.values)
                distance = np.append(distance, dist)
                names.append(filename)
            
            # Add distances 
            df_dist = pd.DataFrame({'filepath': names, 'Distance': distance})
            df_dist = df_dist.merge(test_names, on='filepath')
            df_dist = df_dist.groupby('condition').mean()
            #df_dist.sort_values(by='Distance', inplace=True)

            # Compute third order poly mapping
            #df_grouped = df_dist.groupby('condition')['mos', 'Distance'].mean()
            popt3, _ = curve_fit(self.order_three, df_dist['Distance'].values, df_dist['mos'].values)
            a3, b3, c3, d3 = popt3
            df_dist['Distance_map'] = df_dist['Distance'].apply(lambda x: self.order_three(x,a3,b3,c3,d3))

            # Scatter plot MOS - Embedding Distances
            sns.scatterplot(data=df_dist, x='mos', y='Distance_map')
            plt.xlabel('Actual MOS')
            plt.ylabel(f'Dist w.r.t CLEAN ENGLISH')
            plt.xlim([1, 5])
            plt.ylim([1, 5])
            plt.tight_layout()
            out_dir = '/'.join(self.config['rank_model_path'].split('/')[:-1])
            plt.savefig(os.path.join(out_dir, f'{db_name}_embeddings.png'))
            plt.close()
            
            # Performance Evaluation    
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['mos'])
            print(f'SRCC: {np.round(SRCC, 3)}')
            SRCC, _ = spearmanr(df_dist['Distance_map'], df_dist['mos'])
            print(f'SRCC 3rd map: {np.round(SRCC, 3)}')

            PCC, _ = pearsonr(df_dist['Distance'], df_dist['mos'])
            print(f'PCC: {np.round(PCC, 3)}')

            PCC, _ = pearsonr(df_dist['Distance_map'], df_dist['mos'])
            print(f'PCC 3rd map: {np.round(PCC, 3)}')
        return df_dist

    def eval_crossref_embeddings(self, model_path):
        # Load model weights
        if self.config['eval_#mos_pred']:
            pt_model = torch.oad(self.config['mos_pred_model'], map_location=self.DEVICE)
            model_state_dict = self.model.state_dict()
            keys = list(model_state_dict.keys())
            layers = {k: v for k, v in pt_model.items() if k in keys}                      
            model_state_dict.update(layers)
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Compare cross reference data vs same reference data to see if the problem is there
        cross_ref_db = pd.read_csv(self.config['cross_ref_db'])
        
        # Loop over each database
        db_groups = cross_ref_db.groupby('ref_type')

        pos_distance = []
        neg_distance = []

        for db_name, db in db_groups:
            distance = []
            id_names = []
            sign_label = []
            concordant = 0
            total = 0
            
            anc_emb = self.get_embeddings_csv(self.model, db['Anchor'], root=self.config['cross_ref_wav'])
            pos_emb = self.get_embeddings_csv(self.model, db['Positive'], root=self.config['cross_ref_wav'])
            neg_emb = self.get_embeddings_csv(self.model, db['Negative'], root=self.config['cross_ref_wav'])

            for id_triplet, vals in anc_emb.iterrows():
                pos_dist = self.euclidean_dist(vals.iloc[1:].values, pos_emb.loc[id_triplet].iloc[1:].values)
                #pos_distance = np.append(pos_distance, pos_dist)
                sign_label.append('Positive')

                neg_dist = self.euclidean_dist(vals.iloc[1:].values, neg_emb.loc[id_triplet].iloc[1:].values)
                #neg_distance = np.append(neg_distance, neg_dist)
                sign_label.append('Negative')
                
                if (pos_dist - neg_dist) >= 0:
                    id_names.append(id_triplet)    
                    distance = np.append(distance, pos_dist - neg_dist)
                else:
                    concordant += 1
                    #id_names.append(id_triplet)          
                total += 1
                
            perf = np.round(concordant/ total, 2)
            print(f'{db_name}: {perf}')
            df_plot = pd.DataFrame({'Names': id_names, 'Distance': distance})
            sns.barplot(df_plot, x='Names', y='Distance')
            plt.savefig(f'{db_name}_mistakes.png')

            #test_embeddings = df_emb.iloc[:,1:].to_numpy()
            #test_names = df_emb.merge(db, on='filepath')['filename']
            
            # test_cond_embeddings = df_emb.merge(db[['filepath', 'condition']], on='filepath').iloc[:,1:].groupby('condition').mean()
            # test_cond_gt = db.groupby('condition')['mos'].mean()            
            
            # distance = []
            # names = []

            # for id_cond, test_emb in test_cond_embeddings.iterrows():
            # #for i in range(200):
            #     dist = self.euclidean_dist(ref_avg_embeddings, test_emb.values)
            #     distance = np.append(distance, dist)
            #     names.append(id_cond)
            
            # # Add distances 
            # df_dist = pd.DataFrame({'condition': names, 'Distance': distance})
            # df_dist = df_dist.merge(test_cond_gt, on='condition')
            # df_dist.sort_values(by='Distance', inplace=True)

            # # Compute third order poly mapping
            # #df_grouped = df_dist.groupby('condition')['mos', 'Distance'].mean()
            # popt3, _ = curve_fit(self.order_three, df_dist['Distance'].values, df_dist['mos'].values)
            # a3, b3, c3, d3 = popt3
            # df_dist['Distance_map'] = df_dist['Distance'].apply(lambda x: self.order_three(x,a3,b3,c3,d3))

            # # Scatter plot MOS - Embedding Distances
            # sns.scatterplot(data=df_dist, x='mos', y='Distance_map')
            # plt.xlabel('Actual MOS')
            # plt.ylabel(f'Dist w.r.t CLEAN ENGLISH')
            # plt.xlim([1, 5])
            # plt.ylim([1, 5])
            # plt.tight_layout()
            # plt.savefig(f'./img_avg_emb_hard/{db_name}_embeddings.png')
            # plt.close()
            
            # # Performance Evaluation
            # SRCC, _ = spearmanr(df_dist['Distance'], df_dist['mos'])
            # print(f'SRCC: {np.round(SRCC, 3)}')
            # SRCC, _ = spearmanr(df_dist['Distance_map'], df_dist['mos'])
            # print(f'SRCC 3rd map: {np.round(SRCC, 3)}')

            # PCC, _ = pearsonr(df_dist['Distance'], df_dist['mos'])
            # print(f'PCC: {np.round(PCC, 3)}')

            # PCC, _ = pearsonr(df_dist['Distance_map'], df_dist['mos'])
            # print(f'PCC 3rd map: {np.round(PCC, 3)}')
        return anc_emb


    def eval_full_reference(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Get dataframe of anchor embeddings of the full dataset
        test_data = pd.read_csv(self.config['test_db_file_fr'])
        
        # Loop over each database
        db_groups = test_data.groupby('db')
        
        for db_name, db in db_groups:
            print(db_name)
            df_emb_ref = self.get_embeddings_csv(self.model, db['Filepath Ref']).set_index('Filepath Ref')
            df_emb_test = self.get_embeddings_csv(self.model, db['Filepath Deg']).set_index('Filepath Deg')
            test_names = df_emb_test.merge(db, on='Filepath Deg')[['Filepath Deg', 'Condition', 'MOS']]

            ref_cond_embeddings = df_emb_ref.merge(db[['Filepath Ref', 'Condition']], on='Filepath Ref').iloc[:,1:].groupby('Condition').mean()
            test_cond_embeddings = df_emb_test.merge(db[['Filepath Deg', 'Condition']], on='Filepath Deg').iloc[:,1:].groupby('Condition').mean()
            test_cond_gt = db.groupby('Condition')['MOS'].mean()            
            
            distance = []
            names = []

            for id_file, (filepath, test_emb) in enumerate(df_emb_test.iterrows()):
            #for i in range(200):
                dist = self.euclidean_dist(df_emb_ref.iloc[id_file].values, test_emb.values)
                #dist = cosine(ref_avg_embeddings, test_emb.values)
                distance = np.append(distance, dist)
                names.append(filepath)
            
            # # Add distances 
            df_dist = pd.DataFrame({'Filepath Deg': names, 'Distance': distance})
            df_dist = df_dist.merge(test_names, on='Filepath Deg')
            df_dist = df_dist.groupby('Condition').mean()

            # Compute third order poly mapping
            #df_grouped = df_dist.groupby('condition')['mos', 'Distance'].mean()
            popt3, _ = curve_fit(self.order_three, df_dist['Distance'].values, df_dist['MOS'].values)
            a3, b3, c3, d3 = popt3
            df_dist['Distance_map'] = df_dist['Distance'].apply(lambda x: self.order_three(x,a3,b3,c3,d3))

            # Scatter plot MOS - Embedding Distances
            sns.scatterplot(data=df_dist, x='MOS', y='Distance_map')
            plt.xlabel('Actual MOS')
            plt.ylabel(f'Dist w.r.t Reference')
            plt.xlim([1, 5])
            plt.ylim([1, 5])
            plt.tight_layout()
            out_dir = '/'.join(self.config['rank_model_path'].split('/')[:-1])
            plt.savefig(os.path.join(out_dir, f'fr_{db_name}_embeddings.png'))
            plt.close()
            
            # Performance Evaluation    
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['MOS'])
            print(f'SRCC: {np.round(SRCC, 2)}')
            SRCC, _ = spearmanr(df_dist['Distance_map'], df_dist['MOS'])
            print(f'SRCC 3rd map: {np.round(SRCC, 2)}')

            PCC, _ = pearsonr(df_dist['Distance'], df_dist['MOS'])
            print(f'PCC: {np.round(PCC, 2)}')

            PCC, _ = pearsonr(df_dist['Distance_map'], df_dist['MOS'])
            print(f'PCC 3rd map: {np.round(PCC, 2)}')
        return 3    
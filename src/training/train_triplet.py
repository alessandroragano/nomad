import torch
import torch.nn as nn
import fairseq
import yaml
from src.dataloader.triplet_dataloader import TripletDataset, load_processing
from src.models.networks import TripletModel, Origw2v
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
from scipy.optimize import curve_fit
import torch.optim.lr_scheduler as lr_scheduler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
sns.set_style('darkgrid')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.04, point['y'], str(point['val']))

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
        if torch.cuda.is_available():
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'
        print(f'Device: {self.DEVICE}')

        # Load SSL model if using wav2vec
        CHECKPOINT_PATH = self.config['checkpoint_path']
        SSL_OUT_DIM = self.config['ssl_out_dim']
        EMB_DIM = self.config['emb_dim']

        w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
        ssl_model = w2v_model[0] 
        ssl_model.remove_pretraining_modules()
        
        self.model = TripletModel(ssl_model, SSL_OUT_DIM, EMB_DIM)
        if self.config['eval_w2v']:
            self.model = Origw2v(ssl_model, SSL_OUT_DIM)
        self.model.to(self.DEVICE)

        # Choose if you want to 1) Freeze only ConvNet 2) Freeze ConvNet + Transformer 3) Finetune the entire network
        # Freeze only ConvNet
        if self.config['experiment_name'] == 'Training':
            if self.config['freeze_convnet']:
                self.model.ssl_model.feature_extractor.requires_grad_(False)
            
            # Freeze both ConvNet and Transformer (no finetuning)
            if self.config['freeze_all']:
                self.model.ssl_model.feature_extractor.requires_grad_(False)
                self.model.ssl_model.encoder.requires_grad_(False)

        if self.config['experiment_name'] == 'Training':
            # Create dataloaders
            self.current_level = self.config['current_level']
            self.train_set = TripletDataset(self.config, data_mode='train_df', level=self.current_level)
            collate_fn = self.train_set.collate_fn
            self.train_loader = DataLoader(self.train_set, batch_size=self.config['train_bs'], shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn)
            self.valid_set = TripletDataset(self.config, data_mode='valid_df', level=self.current_level)
            self.valid_loader = DataLoader(self.valid_set, batch_size=self.config['val_bs'], shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn)

            # Create loss
            self.criterion = nn.TripletMarginLoss(margin=self.config['margin'])
            
            # Create optimizer
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

            # Create optimizer with adaptive learning rate
            if self.config['freeze_convnet']:
                # params_names_embeddings = [f'embeddings.{j}.weight' for j in range(7)] + [f'embeddings.{j}.bias' for j in range(7)]
                params_names_embeddings = ['embedding_layer.1.weight', 'embedding_layer.1.bias']
                params_pt = [param for name, param in self.model.named_parameters() if name not in params_names_embeddings]
                params_embeddings = [param for name, param in self.model.named_parameters() if name in params_names_embeddings]                 
                # Overwrite optimizer
                self.optim = torch.optim.Adam([
                    {'params': params_pt, 'lr': 1e-5},
                    {'params': params_embeddings}
                ], lr=self.config['lr'])
        
            # Create learning rate scheduler
            self.lr_scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=self.config['lr_decay_factor'])

    def train(self, model, dataloader, optimizer, criterion):
        model.train()
        total_loss = 0.0

        # Batch data (Anchor, Positive, Negative)
        for batch_index, (A, P, N) in enumerate(tqdm(dataloader)):

            A, P, N = A.to(self.DEVICE), P.to(self.DEVICE), N.to(self.DEVICE)

            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)

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
            for batch_index, (A, P, N) in enumerate(tqdm(dataloader)):

                A, P, N = A.to(self.DEVICE), P.to(self.DEVICE), N.to(self.DEVICE)

                A_embs = model(A)
                P_embs = model(P)
                N_embs = model(N)

                loss = criterion(A_embs, P_embs, N_embs)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def training_loop(self):
        # Start Training Loop
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.PATH_DIR = os.path.join('out-models', self.config['out_dir'], dt_string)
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
                print('Stop training, counter greater than patience')
                break
            
            print(f"EPOCHS: {i+1} train_loss : {train_loss}")
            print(f"EPOCHS: {i+1} valid_loss : {valid_loss}")
            print('\n')

    # *** BELOW FUNCTIONS TO EVALUATE NOMAD ***
    # Function that extract NOMAD embeddings of a db and store them in a csv
    def get_embeddings_csv(self, model, file_names, root=False):
        file_names_arr = np.array(file_names)
        embeddings = []

        model.eval()
        with torch.no_grad():
            for i, filename_anchor in enumerate(tqdm(file_names_arr)):
                if root:
                    filepath = os.path.join(root, filename_anchor)
                else:
                    filepath = filename_anchor
                
                test_speech = load_processing(filepath, trim=False)
                
                test_speech = test_speech.to(self.DEVICE)
                anc_embeddings = model(test_speech)
                embeddings.append(anc_embeddings.squeeze().cpu().detach().numpy())

            embeddings = np.array(embeddings)
            embeddings = pd.DataFrame(embeddings)
            df_emb = pd.concat([file_names.reset_index(), embeddings], axis=1).drop('index', axis=1)
        
        return df_emb

    def order_three(self, x, a, b, c, d):
        return a*x + b*x**2 + c*x**3 + d

    # Evaluate audio quality with non-matching references
    def eval_audio_quality(self, model_path):
        # Load model weights
        if not self.config['eval_w2v']:
            self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()

        # Get dataframe of the test set
        test_data = pd.read_csv(self.config['test_db_file'])

        # Filter db you want to test, use None to use every db
        if self.config['db'] is not None:
            test_data = test_data[test_data['db'].isin(self.config['db'])]

        # Filter conditions you want to test, use None to use every condition
        db_test = self.config['db']
        if self.config['conds'] is not None:
            conds = self.config['conds']
            print(f'Testing DB: {db_test}, conds: {conds}')
            test_data = test_data[test_data['condition'].str.contains('|'.join(conds))]

        # Get reference files from a clean set (paper uses TSP speech database)
        ref_embeddings = self.get_nmr_embeddings()
        ref_embeddings.set_index('reference', inplace=True)

        # Loop over each database
        db_groups = test_data.groupby('db')

        for db_name, db in db_groups:
            print(db_name)

            # Get test embeddings
            df_emb = self.get_embeddings_csv(self.model, db['filepath_deg'], root=self.config['test_root_wav'])
            test_embeddings = df_emb.set_index('filepath_deg')
            test_names = df_emb.merge(db, on='filepath_deg')[['filepath_deg', 'condition', 'mos']]         
            
            # Calculate distance of each pair then take the average over all the non-matching references for each test audio
            euclid_dist = cdist(test_embeddings, ref_embeddings)
            avg_dist_nmr = np.mean(euclid_dist, axis=1)
            names = test_embeddings.index

            # Add distances 
            df_dist = pd.DataFrame({'filepath_deg': names, 'Distance': avg_dist_nmr})
            df_dist = df_dist.merge(test_names, on='filepath_deg').set_index('filepath_deg')
            df_dist = df_dist.groupby('condition').mean()

            # Compute third order poly mapping (performance in the paper are reported without mapping)
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
            out_dir = '/'.join(self.config['nomad_model_path'].split('/')[:-1])
            plt.savefig(os.path.join(out_dir, f'{db_name}_embeddings.png'))
            plt.close()
            
            # Performance Evaluation Spearman
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['mos'])
            print(f'SRCC: {np.round(SRCC, 2)}')
            SRCC, _ = spearmanr(df_dist['Distance_map'], df_dist['mos'])
            print(f'SRCC 3rd map: {np.round(SRCC, 2)}')

            # Performance Evaluation Pearson
            PCC, _ = pearsonr(df_dist['Distance'], df_dist['mos'])
            print(f'PCC: {np.round(PCC, 2)}')
            PCC, _ = pearsonr(df_dist['Distance_map'], df_dist['mos'])
            print(f'PCC 3rd map: {np.round(PCC, 2)}')
        
    def eval_degr_level(self, model_path):
        # Load model weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Get validation set embeddings
        valid_set = TripletDataset(self.config, data_mode='valid_df', level=self.config['current_level'])
        df_emb = self.get_embeddings_csv(self.model, valid_set.dataset['Anchor'], root=self.config['root'])

        # Get non-matching reference embeddings
        ref_embeddings = self.get_nmr_embeddings()

        # Get validation set anchors - test embeddings
        test_embeddings = df_emb.iloc[:, 1:].to_numpy()
        anc_names = df_emb['Anchor']

        # Take pairwise euclidean distance and then average over non-matching references
        euclid_dist = cdist(test_embeddings, ref_embeddings.iloc[:,1:].to_numpy())
        avg_dist_nmr = np.mean(euclid_dist, axis=1)

        # DF Store 
        df_dist = pd.DataFrame({'Anchor': anc_names, 'Distance': avg_dist_nmr})
        df_dist.sort_values(by='Distance', inplace=True)

        # Boxplot validation set sorted 
        plt.figure(figsize=(50, 20))
        sns.set(font_scale=8.0)
        df_dist['condition'] = [x.split('_')[1] + ' ' + x.split('_')[2].split('.')[0] for x in df_dist['Anchor']]
        order = df_dist.groupby('condition')['Distance'].mean().sort_values().index
        sns.boxplot(df_dist, x='condition', y='Distance', order=order, showmeans=True, meanprops={"markerfacecolor":"white", "markeredgecolor":"blue", "markersize": "50"}, width=0.8, linewidth=6)
        
        # Figure settings
        plt.xticks(rotation=65)
        plt.ylabel(f'NOMAD')
        plt.xlabel('Condition')
        plt.tight_layout()
        out_dir = '/'.join(self.config['nomad_model_path'].split('/')[:-1])
        plt.savefig(os.path.join(out_dir, f'validset_embeddings.png'))
    
    def eval_degradation_intensity(self, model_path):
        if not self.config['eval_w2v']:
            self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()

        # Get non matching references
        ref_embeddings = self.get_nmr_embeddings()
        ref_embeddings.set_index('reference', inplace=True)

        # Get test data
        test_data = pd.read_csv(self.config['test_mono_data'])

        # Loop over each degradation
        test_data_group = test_data.groupby('Degradation')
        
        # Store for scatterplot
        degr_names = []
        degr_data = []
        srcc_scores = []

        degradations = [len(ref_embeddings) * ['Unpaired Clean']]
        conditions_pca = [len(ref_embeddings) * ['Unpaired Clean']]
        test_db = [ref_embeddings]

        for deg_name, deg_data in test_data_group:
            df_emb = self.get_embeddings_csv(self.model, deg_data['filepath_deg'], root=self.config['test_mono_wav']) 

            test_embeddings = df_emb.set_index('filepath_deg')
            test_names = df_emb.merge(deg_data, on='filepath_deg')[['filepath_deg', 'Condition']]         

            euclid_dist = cdist(test_embeddings, ref_embeddings)
            avg_dist_nmr = np.mean(euclid_dist, axis=1)
            names = test_embeddings.index

            # Add distances 
            df_dist = pd.DataFrame({'filepath_deg': names, 'Distance': avg_dist_nmr})
            df_dist = df_dist.merge(test_names, on='filepath_deg')
            df_dist.set_index('filepath_deg', inplace=True)
            df_dist = df_dist.groupby('Condition').mean().reset_index()
            df_dist.sort_values(by='Distance', inplace=True)
            
            # Store for scatterplot
            degr_names.append([deg_name]*len(df_dist))
            degr_data.append(df_dist)

            # Performance Evaluation    
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['Condition'])
            srcc_scores.append([SRCC]*len(df_dist))
            print(f'Degradation: {deg_name}')
            print(f'SRCC: {np.round(SRCC, 2)}')

            # Store for PCA
            degradations.append([deg_name] * len(test_embeddings))
            conditions = ['_'.join(x.split('/')[1].split('_')[-2:]).split('.')[0] for x in names]
            if len(test_embeddings) != len(conditions):
                print('dd')
            conditions_pca.append(conditions)
            test_db.append(test_embeddings)
        
        # Compute PCA
        pca_plot = False
        if pca_plot:
            df_pca = pd.concat(test_db)
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(df_pca)
            pca_features = pd.DataFrame(pca_features)
            pca_features['Condition'] = np.concatenate(conditions_pca)
            pca_features = pca_features.groupby('Condition').mean().reset_index()
            p1 = sns.scatterplot(data=pca_features, x=0, y=1, hue='Condition', legend=False)
            #label_point(pca_features[0], pca_features[1], pca_features['Condition'], plt.gca())  
            for line in range(0,pca_features.shape[0]):
                if line % 10 == 0:
                    p1.text(pca_features[0][line]-0.02, pca_features[1][line] + 0.01, 
                    pca_features['Condition'][line], horizontalalignment='left', 
                    size='medium', color='black', weight='semibold')
            plt.savefig('pca_figs/nomad_embeddings_pca.png')

    def eval_full_reference(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Get dataframe test set
        test_data = pd.read_csv(self.config['test_db_file_fr'])

        # Loop over each database
        db_groups = test_data.groupby('db')
        
        for db_name, db in db_groups:
            print(db_name)
            df_emb_ref = self.get_embeddings_csv(self.model, db['filepath_ref'], root=self.config['test_root_wav']).set_index('filepath_ref')
            df_emb_test = self.get_embeddings_csv(self.model, db['filepath_deg'], root=self.config['test_root_wav']).set_index('filepath_deg')
            test_names = df_emb_test.merge(db, on='filepath_deg')[['filepath_deg', 'condition', 'mos']]   
            
            # Eucl distance only with matching reference (take diagonal)
            euclid_dist = cdist(df_emb_test, df_emb_ref)
            fr_distance = np.diag(euclid_dist)
            names = df_emb_test.index

            # # Add distances 
            df_dist = pd.DataFrame({'filepath_deg': names, 'Distance': fr_distance})
            df_dist = df_dist.merge(test_names, on='filepath_deg')
            df_dist = df_dist.groupby('condition').mean()

            # Compute third order poly mapping (paper results are reported without mapping)
            #df_grouped = df_dist.groupby('condition')['mos', 'Distance'].mean()
            popt3, _ = curve_fit(self.order_three, df_dist['Distance'].values, df_dist['mos'].values)
            a3, b3, c3, d3 = popt3
            df_dist['Distance_map'] = df_dist['Distance'].apply(lambda x: self.order_three(x,a3,b3,c3,d3))

            # Scatter plot MOS - Embedding Distances
            sns.scatterplot(data=df_dist, x='mos', y='Distance_map')
            plt.xlabel('Actual MOS')
            plt.ylabel(f'Dist w.r.t Reference')
            plt.xlim([1, 5])
            plt.ylim([1, 5])
            plt.tight_layout()
            out_dir = '/'.join(self.config['nomad_model_path'].split('/')[:-1])
            plt.savefig(os.path.join(out_dir, f'fr_{db_name}_embeddings.png'))
            plt.close()
            
            # Performance Evaluation Spearman
            SRCC, _ = spearmanr(df_dist['Distance'], df_dist['mos'])
            print(f'SRCC: {np.round(SRCC, 2)}')
            SRCC, _ = spearmanr(df_dist['Distance_map'], df_dist['mos'])
            print(f'SRCC 3rd map: {np.round(SRCC, 2)}')

            # Performance Evaluation Pearson
            PCC, _ = pearsonr(df_dist['Distance'], df_dist['mos'])
            print(f'PCC: {np.round(PCC, 2)}')
            PCC, _ = pearsonr(df_dist['Distance_map'], df_dist['mos'])
            print(f'PCC 3rd map: {np.round(PCC, 2)}')
    
    def get_nmr_embeddings(self):
        # Dataframe
        ref_files = pd.DataFrame(os.listdir(self.config['non_match_dir']))
        ref_files.columns = ['reference']
        ref_files['reference'] = [os.path.join(self.config['non_match_dir'], x) for x in ref_files['reference']]
        
        # Calculate embeddings
        ref_embeddings = self.get_embeddings_csv(self.model, ref_files)
        return ref_embeddings
    
    # Use this function to check if cdist gives the same result
    def euclidean_dist(self, emb_a, emb_b):
        dist = np.sqrt(np.dot(emb_a - emb_b, (emb_a - emb_b).T))
        return dist

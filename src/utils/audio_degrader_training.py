import yaml
import os
import subprocess
from degradations import mp3, opus, noise, clip_signal
import pandas as pd
import random
random.seed(0)

with open('src/config/config_audio_degrader.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

sr = config['sr']
in_dir = os.path.join(config['root'], config['in_dir_train'])
in_dir_wav = os.path.join(config['root'], config['in_dir_train_wav'])
out_dir = os.path.join(config['root'], config['out_dir_train'])

# Specify the total number of conditions (5 degradations x 4 intensity levels )
n_cond = 20

filepath_deg, filename_deg, filepath_ref, filename_ref = [], [], [], []
abs_filepath_deg, abs_filepath_ref = [], []
for root, dirs, files in os.walk(in_dir):
    for filename in files:
        if (filename.endswith('.flac') | (filename.endswith('.wav'))):  
            in_filepath = os.path.join(root, filename)
            filename = os.path.splitext(filename)[0] + '.wav'
            
            # Store filename ref 
            filename_ref += n_cond * [filename]

            # Store filepath ref (relative path)
            sub_dir = root.split('/')[-2:]
            sub_dir = '/'.join(sub_dir)

            # Make subdirs
            if not os.path.isdir(os.path.join(in_dir_wav, sub_dir)):
                os.makedirs(os.path.join(in_dir_wav, sub_dir))
            in_filepath_wav = os.path.join(in_dir_wav, sub_dir, filename)
            
            # Convert to wav for visqol
            if not in_filepath.endswith('.wav'):
                wav_convert = f'ffmpeg -y -i {in_filepath} -ar {sr} {in_filepath_wav}'.split(' ')
                subprocess.call(wav_convert)

            filepath_ref_relative = os.path.join(sub_dir, filename)
            filepath_ref += n_cond * [filepath_ref_relative]

            # Store absolute filepath ref
            abs_filepath_ref += n_cond * [in_filepath_wav]

            # *** MP3 ****
            # Create out subdir
            subdir_name = 'MP3'
            out_dir_mp3 = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_mp3):
                os.makedirs(out_dir_mp3)
            
            # Call mp3 codec for all conditions
            for cond in config['mp3_train']:
                # Build filename deg file
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{cond}.wav'
                filename_deg.append(filename_deg_file)
                
                # Build filepath deg file
                out_filepath = os.path.join(out_dir_mp3, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)

                mp3(in_filepath_wav, out_filepath, bitrate=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)


            # *** OPUS ***
            # Create out subdir
            subdir_name = 'OPUS'
            out_dir_opus = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_opus):
                os.makedirs(out_dir_opus)
                    
            # Call mp3 codec for all conditions
            for cond in config['opus_train']:
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_opus, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)

                opus(in_filepath_wav, out_filepath, bitrate=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

            # *** NOISE ***
            # Create out subdir
            subdir_name = 'NOISE'
            out_dir_noise = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_noise):
                os.makedirs(out_dir_noise)
                
            # Pick one noise file
            noise_dir = os.path.join(config['root_noise'], config['noise_dir_train'])
            list_noise_files = [nx for nx in os.listdir(noise_dir) if nx.endswith('.wav')]
            noise_path = os.path.join(noise_dir, random.choice(list_noise_files))
            for cond in config['noise_train']:
                str_cond = str(cond)
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_noise, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)
                
                noise(in_filepath_wav, noise_path, out_filepath, snr_db=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

            # *** CLIP ***
            # Create out subdir
            subdir_name = 'CLIP'
            out_dir_clip = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_clip):
                os.makedirs(out_dir_clip)
            
            # Clip
            for cond in config['clip_train']:
                str_cond = str(cond)
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_clip, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)            

                clip_signal(in_filepath_wav, out_filepath, clip_factor=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

# Store data
df = pd.DataFrame({'filename_ref': filename_ref, 'filepath_ref': filepath_ref, 'filename_deg': filename_deg, 'filepath_deg': filepath_deg})
df.to_csv('degraded_data.csv', index=False)

# Store data for ViSQOL
df_visqol = pd.DataFrame({'filepath_ref': abs_filepath_ref, 'filepath_deg': abs_filepath_deg})
df_visqol.to_csv('degraded_data_visqol_format.csv', index=False)
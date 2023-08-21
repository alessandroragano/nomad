import yaml
import os
import subprocess
from degradations import mp3, opus, noise, clip_signal, vorbis, reverb
import degradations
import pandas as pd
import random
random.seed(0)

# Function that picks a random file
def getRandomFile(path):
    randomDir = random.choice([(x) for x in list(os.scandir(path)) if x.is_dir()]).name
    randomDir2 = random.choice([f for f in list(os.scandir(os.path.join(path, randomDir)))]).name
    listOfFiles = os.listdir(os.path.join(path, randomDir, randomDir2))
    listOfFiles = [x for x in listOfFiles if x.endswith('.flac')]
    randomFile = random.choice(listOfFiles)
    path_randomFile = os.path.join(path, randomDir, randomDir2, randomFile)
    return path_randomFile, randomFile, os.path.join(randomDir, randomDir2)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

sr = config['sr']
in_dir = config['in_dir']
out_dir = config['out_dir']


n_cond = 60
num_files = 4

degradations = ['MP3', 'OPUS', 'CLIP', 'NOISE', 'VORBIS', 'REVERB']

filepath_deg, filename_deg, filepath_ref, filename_ref = [], [], [], []
abs_filepath_deg, abs_filepath_ref = [], []
id_clean_file = 0

def pick_clean_file():
    # Pick random clean file
    in_filepath, filename, root = getRandomFile(in_dir)
    filename_ref = os.path.splitext(filename)[0] + '.wav'

    # Store filepath ref (relative path)
    sub_dir = root.split('/')[-2:]
    sub_dir = '/'.join(sub_dir)

    # Convert to wav for visqol
    if not os.path.isdir(os.path.join(config['librispeech_wav'], sub_dir)):
        os.makedirs(os.path.join(config['librispeech_wav'], sub_dir))
    
    in_filepath_wav = os.path.join(config['librispeech_wav'], sub_dir, filename_ref)
    wav_convert = f'ffmpeg -y -i {in_filepath} -ar {sr} {in_filepath_wav}'.split(' ')
    subprocess.call(wav_convert)

    filepath_ref = os.path.join(sub_dir, filename)

    return filename_ref, filepath_ref, filename, in_filepath_wav

for nfil in range(0, num_files):

    for d in degradations:
        # *** MP3 ***
        # Create out subdir
        if d == 'MP3':
            subdir_name = 'MP3'
            out_dir_mp3 = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_mp3):
                os.makedirs(out_dir_mp3)
            
            # Call mp3 codec for all conditions
            for cond in config[d]:
                filename_ref, filepath_ref, filename, in_filepath_wav = pick_clean_file()
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

        elif d == 'OPUS':
            # *** OPUS ***
            # Create out subdir
            subdir_name = 'OPUS'
            out_dir_opus = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_opus):
                os.makedirs(out_dir_opus)
                    
            # Call mp3 codec for all conditions
            for cond in config[d]:
                filename_ref, filepath_ref, filename, in_filepath_wav = pick_clean_file()

                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_opus, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)

                opus(in_filepath_wav, out_filepath, bitrate=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

        elif d == 'NOISE':
            # *** NOISE ***
            # Create out subdir
            subdir_name = 'NOISE'
            out_dir_noise = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_noise):
                os.makedirs(out_dir_noise)
                
            # Pick one noise file
            list_noise_files = [nx for nx in os.listdir(config['noise_dir_test']) if nx.endswith('.wav')]
            noise_path = os.path.join(config['noise_dir_test'], random.choice(list_noise_files))
            for cond in config[d]:
                filename_ref, filepath_ref, filename, in_filepath_wav = pick_clean_file()

                str_cond = str(cond)
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_noise, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)
                
                noise(in_filepath_wav, noise_path, out_filepath, snr_db=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

        elif d == 'CLIP':
            # *** CLIP ***
            # Create out subdir
            subdir_name = 'CLIP'
            out_dir_clip = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_clip):
                os.makedirs(out_dir_clip)
            
            # Clip
            for cond in config[d]:
                filename_ref, filepath_ref, filename, in_filepath_wav = pick_clean_file()

                str_cond = str(cond)
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_clip, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)            

                clip_signal(in_filepath_wav, out_filepath, clip_factor=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)
        
        elif d == 'VORBIS':
            # VOrbis
            subdir_name = 'VORBIS'
            out_dir_vorbis = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_vorbis):
                os.makedirs(out_dir_vorbis)

            for cond in config[d]:
                filename_ref, filepath_ref, filename, in_filepath_wav = pick_clean_file()

                str_cond = str(cond)
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_vorbis, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)            

                vorbis(in_filepath_wav, out_filepath, quality=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

        elif d == 'REVERB':
            # Reverb
            subdir_name = 'REVERB'
            out_dir_reverb = os.path.join(out_dir, subdir_name)
            if not os.path.isdir(out_dir_reverb):
                os.makedirs(out_dir_reverb)

            for cond in config[d]:
                filename_ref, filepath_ref, filename, in_filepath_wav = pick_clean_file()

                str_cond = str(cond)
                filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
                filename_deg.append(filename_deg_file)

                out_filepath = os.path.join(out_dir_reverb, filename_deg_file)
                filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
                abs_filepath_deg.append(out_filepath)            

                reverb(in_filepath_wav, out_filepath, p=cond, sr=config['sr'])
                cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
                subprocess.call(cmd_normalization)

df = pd.DataFrame({'filename_ref': filename_ref, 'filepath_ref': filepath_ref, 'filename_deg': filename_deg, 'filepath_deg': filepath_deg})
df.to_csv('degraded_data_test_new.csv', index=False)
df_visqol = pd.DataFrame({'filepath_ref': abs_filepath_ref, 'filepath_deg': abs_filepath_deg})
df_visqol.to_csv('degraded_data_visqol_format_test_new.csv', index=False)


# for root, dirs, files in os.walk(in_dir):
#     for filename in files:
#         if filename.endswith('.flac'):
#             in_filepath = os.path.join(root, filename)
#             filename = os.path.splitext(filename)[0] + '.wav'
            
#             # Store filename ref 
#             filename_ref += n_cond * [filename]

#             # Store filepath ref (relative path)
#             sub_dir = root.split('/')[-2:]
#             sub_dir = '/'.join(sub_dir)

#             # Convert to wav for visqol
#             if not os.path.isdir(os.path.join(config['librispeech_wav'], sub_dir)):
#                 os.makedirs(os.path.join(config['librispeech_wav'], sub_dir))
            
#             in_filepath_wav = os.path.join(config['librispeech_wav'], sub_dir, filename)
#             wav_convert = f'ffmpeg -y -i {in_filepath} -ar {sr} {in_filepath_wav}'.split(' ')
#             subprocess.call(wav_convert)

#             filepath_ref_relative = os.path.join(sub_dir, filename)
#             filepath_ref += n_cond * [filepath_ref_relative]

#             # Store absolute filepath ref
#             #abs_filepath_ref.append([in_filepath]*n_cond)
#             abs_filepath_ref += n_cond * [in_filepath_wav]

#             # *** MP3 ***
#             # Create out subdir
#             subdir_name = 'MP3'
#             out_dir_mp3 = os.path.join(out_dir, subdir_name)
#             if not os.path.isdir(out_dir_mp3):
#                 os.makedirs(out_dir_mp3)
            
#             # Call mp3 codec for all conditions
#             for cond in config['mp3_params_test']:
#                 # Build filename deg file
#                 filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{cond}.wav'
#                 filename_deg.append(filename_deg_file)
                
#                 # Build filepath deg file
#                 out_filepath = os.path.join(out_dir_mp3, filename_deg_file)
#                 filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
#                 abs_filepath_deg.append(out_filepath)

#                 mp3(in_filepath_wav, out_filepath, bitrate=cond, sr=config['sr'])
#                 cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
#                 subprocess.call(cmd_normalization)

#             # *** OPUS ***
#             # Create out subdir
#             subdir_name = 'OPUS'
#             out_dir_opus = os.path.join(out_dir, subdir_name)
#             if not os.path.isdir(out_dir_opus):
#                 os.makedirs(out_dir_opus)
                    
#             # Call mp3 codec for all conditions
#             for cond in config['opus_params_test']:
#                 filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{cond}.wav'
#                 filename_deg.append(filename_deg_file)

#                 out_filepath = os.path.join(out_dir_opus, filename_deg_file)
#                 filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
#                 abs_filepath_deg.append(out_filepath)

#                 opus(in_filepath_wav, out_filepath, bitrate=cond, sr=config['sr'])
#                 cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
#                 subprocess.call(cmd_normalization)

#             # *** NOISE ***
#             # Create out subdir
#             subdir_name = 'NOISE'
#             out_dir_noise = os.path.join(out_dir, subdir_name)
#             if not os.path.isdir(out_dir_noise):
#                 os.makedirs(out_dir_noise)
                
#             # Pick one noise file
#             list_noise_files = [nx for nx in os.listdir(config['noise_dir_test']) if nx.endswith('.wav')]
#             noise_path = os.path.join(config['noise_dir_test'], random.choice(list_noise_files))
#             for cond in config['noise_params_test']:
#                 str_cond = str(cond)
#                 filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
#                 filename_deg.append(filename_deg_file)

#                 out_filepath = os.path.join(out_dir_noise, filename_deg_file)
#                 filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
#                 abs_filepath_deg.append(out_filepath)
                
#                 noise(in_filepath_wav, noise_path, out_filepath, snr_db=cond, sr=config['sr'])
#                 cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
#                 subprocess.call(cmd_normalization)

#             # *** CLIP ***
#             # Create out subdir
#             subdir_name = 'CLIP'
#             out_dir_clip = os.path.join(out_dir, subdir_name)
#             if not os.path.isdir(out_dir_clip):
#                 os.makedirs(out_dir_clip)
            
#             # Clip
#             for cond in config['clip_params_test']:
#                 str_cond = str(cond)
#                 filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
#                 filename_deg.append(filename_deg_file)

#                 out_filepath = os.path.join(out_dir_clip, filename_deg_file)
#                 filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
#                 abs_filepath_deg.append(out_filepath)            

#                 clip_signal(in_filepath_wav, out_filepath, clip_factor=cond, sr=config['sr'])
#                 cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
#                 subprocess.call(cmd_normalization)
            
#             # VOrbis
#             subdir_name = 'VORBIS'
#             out_dir_vorbis = os.path.join(out_dir, subdir_name)
#             if not os.path.isdir(out_dir_vorbis):
#                 os.makedirs(out_dir_vorbis)

#             for cond in config['vorbis_params']:
#                 str_cond = str(cond)
#                 filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
#                 filename_deg.append(filename_deg_file)

#                 out_filepath = os.path.join(out_dir_vorbis, filename_deg_file)
#                 filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
#                 abs_filepath_deg.append(out_filepath)            

#                 vorbis(in_filepath_wav, out_filepath, quality=cond, sr=config['sr'])
#                 cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
#                 subprocess.call(cmd_normalization)

#             # Reverb
#             subdir_name = 'REVERB'
#             out_dir_reverb = os.path.join(out_dir, subdir_name)
#             if not os.path.isdir(out_dir_reverb):
#                 os.makedirs(out_dir_reverb)

#             for cond in config['reverb_params']:
#                 str_cond = str(cond)
#                 filename_deg_file = filename.split('.')[0] + f'_{subdir_name}_{str_cond}.wav'
#                 filename_deg.append(filename_deg_file)

#                 out_filepath = os.path.join(out_dir_reverb, filename_deg_file)
#                 filepath_deg.append(os.path.join(subdir_name, filename_deg_file))
#                 abs_filepath_deg.append(out_filepath)            

#                 reverb(in_filepath_wav, out_filepath, p=cond, sr=config['sr'])
#                 cmd_normalization = f'ffmpeg-normalize {out_filepath} -o {out_filepath} -f -ar {16000}'.split(' ')
#                 subprocess.call(cmd_normalization)              

#         id_clean_file += 1
#         break
#     if id_clean_file == num_files:
#         break
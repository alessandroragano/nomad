import math
import subprocess
import numpy as np
import soundfile as sf
import os
import torchaudio

def mp3(in_filepath, out_filepath, bitrate='320k', sr=16000):
    # Compress mp3
    mp3_outpath = os.path.splitext(out_filepath)[0] + '.mp3'
    cmd = f'ffmpeg -y -i {in_filepath} -ar {sr} -b:a {bitrate} {mp3_outpath}'.split(' ')
    subprocess.call(cmd)

    # Convert to wav 
    wav_convert = f'ffmpeg -y -i {mp3_outpath} -ar {sr} {out_filepath}'.split(' ')
    subprocess.call(wav_convert)
    os.remove(mp3_outpath)
    
def opus(in_filepath, out_filepath, bitrate='320k', sr=16000):
    # Compress opus
    opus_outpath = os.path.splitext(out_filepath)[0] + '.opus'
    cmd = f'ffmpeg -y -i {in_filepath} -c:a libopus -b:a {bitrate} -vbr on {opus_outpath}'.split(' ')
    subprocess.call(cmd)

    # Convert to wav
    wav_convert = f'ffmpeg -y -i {opus_outpath} -ar {sr} {out_filepath}'.split(' ')
    subprocess.call(wav_convert)
    os.remove(opus_outpath)
    
def noise(clean_path, noise_path, out_filepath, snr_db=0, sr=16000):
    x, sr = sf.read(clean_path)
    s, sr = sf.read(noise_path)

    # First let's match the length of the two signals
    x_len = x.shape[0]
    s_len = s.shape[0]

    if x_len > s_len:
        # Calculate how many times we need to repeat the signal and round up to the nearest integer
        reps = math.ceil(x_len/s_len)

        # Use the function np.tile to repeat an array
        s = np.tile(s, reps)

    # Truncate the background signal  
    s = s[:x_len]

    # Check if the lengths are the same
    assert x_len == s.shape[0]

    # Convert SNRdb to linear
    snr = 10**(snr_db/10)
    
    # Estimate noise and signal power
    sp = np.sqrt(np.mean(s**2))
    xp = np.sqrt(np.mean(x**2))

    # Calculate desired noise power based on the SNR value
    sp_target = xp/snr

    # Scale factor noise signal
    alpha = sp_target/sp

    # Sum speech and background noise
    y = x + alpha*s

    # Save file
    sf.write(out_filepath, data=y, samplerate=sr)

def clip_signal(in_filepath, out_filepath, clip_factor='10', sr=16000):
    # Read signal
    x, sr = sf.read(in_filepath)
    
    # Calculate percentiles
    lower_percentile = clip_factor/2
    higher_percentile = 100 - lower_percentile
    percentile_values = np.percentile(x, [lower_percentile, higher_percentile])
    
    # Clip signal
    x[x>percentile_values[1]], x[x<percentile_values[0]] = percentile_values[1], percentile_values[0]
    
    # Store clipped signal
    sf.write(out_filepath, data=x, samplerate=sr)

# UNSEEN degradations
def vorbis(in_filepath, out_filepath, quality='3', sr=16000):
    # Compress vorbis
    vorbis_outpath = os.path.splitext(out_filepath)[0] + '.ogg'
    cmd = f'ffmpeg -y -i {in_filepath} -c:a libvorbis -qscale:a {quality} {vorbis_outpath}'.split(' ')
    subprocess.call(cmd)

    # Convert to wav
    wav_convert = f'ffmpeg -y -i {vorbis_outpath} -ar {sr} {out_filepath}'.split(' ')
    subprocess.call(wav_convert)
    os.remove(vorbis_outpath)

def reverb(in_filepath, out_filepath, p=50, sr=16000):
    wave, _  = torchaudio.load(in_filepath)
    d_wave, _ = torchaudio.sox_effects.apply_effects_tensor(wave, sample_rate=sr, effects=[['reverb', f'{p}']])
    d_wave = (d_wave[0,:] + d_wave[1,:])/2
    sf.write(out_filepath, data=d_wave, samplerate=sr)
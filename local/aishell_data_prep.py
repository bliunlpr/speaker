import os
import sys
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc


def make_feature(wav_path, feature_path):
    if not os.path.exists(feature_path):
        rate, sig = wav.read(wav_path)
        feature = mfcc(sig, rate)
        np.save(feature_path, feature)
    return

def main():
    
    wavdir = sys.argv[1]
    datadir = sys.argv[2]
    featdir = sys.argv[3]
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(featdir):
        os.makedirs(featdir)
        
    num = 0
    print(wavdir, datadir, featdir)
    wav_scp_write = open(os.path.join(datadir, 'wav.scp'), 'w')
    utt2spk_write = open(os.path.join(datadir, 'utt2spk'), 'w')
    feats_scp_write = open(os.path.join(datadir, 'feats.scp'), 'w')
    for speaker in os.listdir(wavdir):
        wav_subdir = os.path.join(wavdir, speaker)
        if os.path.isdir(wav_subdir):
            feature_subdir = os.path.join(featdir, speaker)
            if not os.path.exists(feature_subdir):
                os.makedirs(feature_subdir)
            for wav_file in os.listdir(wav_subdir):
                if wav_file[0] != '.':
                    try:
                        make_feature(os.path.join(wav_subdir, wav_file),
                                     os.path.join(feature_subdir, wav_file).replace('.wav', '.npy'))
                        num += 1
                        utt_id = wav_file.replace('.wav', '')
                        wav_scp_write.write(utt_id + ' ' + os.path.join(wav_subdir, wav_file) + '\n')
                        utt2spk_write.write(utt_id + ' ' + speaker + '\n')
                        feats_scp_write.write(utt_id + ' ' + os.path.join(feature_subdir, wav_file).replace('.wav', '.npy') + '\n')
                    except:
                        print(os.path.join(wav_subdir, wav_file), ' has error')
    print('finish, wav file is ', num)                
    wav_scp_write.close()
    utt2spk_write.close()
    feats_scp_write.close()

if __name__ == '__main__':
    main()

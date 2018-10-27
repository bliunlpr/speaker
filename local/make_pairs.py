import os
import sys
import random
import numpy as np

def main():
    
    datadir = sys.argv[1]
    pos_num = int(sys.argv[2])
    neg_num = int(sys.argv[3])
    
    speakers = []
    utt2spk_dict = {}
    fread = open(os.path.join(datadir, 'utt2spk'), 'r')
    for line in fread.readlines():
        line = line.strip()
        splits = line.split(' ')  
        uttid = splits[0]  
        speaker = splits[1]
        utt2spk_dict[uttid] = speaker 
        speakers += [speaker]     
    speakers = list(set(speakers))
    speakers.sort()
    spk2ids = {v: i for i, v in enumerate(speakers)}
    print('spk2ids', len(spk2ids), 'len(utt2spk) ', len(utt2spk_dict))
    
    inds = dict()
    fread = open(os.path.join(datadir, 'feats.scp'), 'r')
    for line in fread.readlines():
        line = line.strip()
        splits = line.split(' ')  
        uttid = splits[0]  
        feat_path = splits[1]
        spk_id = spk2ids[utt2spk_dict[uttid]]
        if spk_id not in inds:
            inds[spk_id] = []
        inds[spk_id].append((uttid, feat_path))    
    print('inds', len(inds), len(inds[0]))

    num = 0
    fwrite = open(os.path.join(datadir, 'pairs.txt'), 'w')
    utt_ids = {}
    while num < pos_num:
        flag = True
        while flag:
            rand_idx = np.random.permutation(len(inds))[0]
            selected_file = inds[rand_idx]
            utters = random.sample(selected_file, 2)
            utt_id0, audio_path0 = utters[0]
            utt_id1, audio_path1 = utters[1]
            if utt_id0 in utt_ids and utt_id1 in utt_ids:
                flag = True
            else:
                flag = False
        utt_ids[utt_id0] = 0
        utt_ids[utt_id1] = 0
        out_line = utt_id0 + ' ' + utt_id1 + ' ' + audio_path0 + ' ' + audio_path1 + ' 1' + '\n'
        fwrite.write(out_line)
        num += 1
    
    num = 0
    while num < neg_num:
        flag = True
        while flag:
            rand_idx = np.random.permutation(len(inds))[:2]
            selected_file0 = inds[rand_idx[0]]
            utters0 = random.sample(selected_file0, 1)
            utt_id0, audio_path0 = utters0[0]
    
            selected_file1 = inds[rand_idx[1]]
            utters1 = random.sample(selected_file1, 1)
            utt_id1, audio_path1 = utters1[0]
            if utt_id0 in utt_ids and utt_id1 in utt_ids:
                flag = True
            else:
                flag = False
        utt_ids[utt_id0] = 0
        utt_ids[utt_id1] = 0
        out_line = utt_id0 + ' ' + utt_id1 + ' ' + audio_path0 + ' ' + audio_path1 + ' 0' + '\n'
        fwrite.write(out_line)
        num += 1
    fwrite.close()
    print(len(utt_ids))

if __name__ == '__main__':
    main()

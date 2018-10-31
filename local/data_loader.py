import os
import random
import gzip
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def dump_to_text(spk2ids, out_file): 
    fwrite = open(out_file, 'w')     
    for key, value in spk2ids.items():
        fwrite.write('{} {}'.format(value, key) + '\n') 
    fwrite.close()

          
def make_utt2spk(utt2spk_file, utt_ids):
    utt2spk = {}        
    fread = open(utt2spk_file, 'r')
    for line in fread.readlines():
        line = line.replace('\n','').strip()
        splits = line.split(' ')     
        utt2spk[splits[0]] = splits[1]

    speakers = []
    for sample in utt_ids:
        utt_id = sample[0]
        if utt_id in utt2spk:
            speakers += [utt2spk[utt_id]]
    speakers = list(set(speakers))
    speakers.sort()
    spk2ids = {v: i for i, v in enumerate(speakers)}
    
    utt2spk_ids = {}
    for sample in utt_ids:
        utt_id = sample[0]
        if utt_id in utt2spk:
            speaker = spk2ids[utt2spk[utt_id]]
            utt2spk_ids[utt_id] = speaker
        else:
            utt2spk_ids[utt_id] = 0
            print ('{} has no utt2spk '.format(utt_id))
    return utt2spk_ids, spk2ids
    
    
def create_indices(wav_utt_ids, utt2spk_ids):
    inds = dict()
    for sample in wav_utt_ids:
        utt_id, audio_path = sample[0], sample[1]
        spk_id = utt2spk_ids[utt_id]
        if spk_id not in inds:        
            inds[spk_id] = []
        inds[spk_id].append((utt_id, audio_path))
    return inds


def create_pair_indices(pair_txt):   
    pairID = []
    inds = dict()
    pair_num = 0
    with open(pair_txt) as f:
        pairs = f.readlines()
        for pair in pairs:
            pair_splits = pair.split(' ')
            if len(pair_splits) < 5:
                continue
            pairID.append(pair_num)
            inds[pair_num] = (pair_splits[0].strip(), pair_splits[1].strip(), pair_splits[2].strip(), pair_splits[3].strip(), int(pair_splits[4]))
            pair_num += 1
    return pairID, inds, len(pairID) 


class BaseDataset(Dataset):
    def __init__(self, opt, data_dir):
        
        self.opt = opt
        self.exp_path = opt.expr_dir
        self.num_utt_cmvn = opt.num_utt_cmvn
        self.normalize_type = opt.normalize_type 
        self.speaker_num = opt.speaker_num
        self.utter_num = opt.utter_num
        self.cmvn = None
        self.data_type = opt.data_type
        self.lb = opt.lb
        self.ub = opt.ub
        
        self.delta_order         = opt.delta_order
        self.left_context_width  = opt.left_context_width
        self.right_context_width = opt.right_context_width
        
        if self.data_type == 'train': 
            self.wav_scp = os.path.join(data_dir, 'feats.scp')
            self.utt2spk = os.path.join(data_dir, 'utt2spk')
            with open(self.wav_scp) as f:
                wav_utt_ids = f.readlines()
            self.wav_utt_ids = [x.strip().split(' ') for x in wav_utt_ids]
            self.wav_utt_size = len(wav_utt_ids)
                
            in_feat = self.parse_audio(self.wav_utt_ids[0][1])
            self.nfeatures = in_feat.shape[1]
        
            self.utt2spk_ids, self.spk2ids = make_utt2spk(self.utt2spk, self.wav_utt_ids)   
            dump_to_text(self.spk2ids, os.path.join(self.exp_path, 'spk2ids'))      
            self.indices = create_indices(self.wav_utt_ids, self.utt2spk_ids)        
            print('have {} speakers'.format(len(self.indices)))
        else:
            self.pairID, self.indices, self.wav_utt_size = create_pair_indices(os.path.join(data_dir, 'pairs.txt'))        
            print('have {} speakers'.format(len(self.indices)))
                    
        if self.normalize_type == 1:
            self.cmvn = self.loading_cmvn()
        super(BaseDataset, self).__init__()
        
    def name(self):
        return 'BaseDataset'
    
    def parse_audio(self, wav_path):
        if wav_path is None:
            return None
        try:
            feature = np.load(wav_path)
            if self.left_context_width > 0 or self.right_context_width > 0:
                feature = splice(feature, self.left_context_width, self.right_context_width)
        except:
            print('{} has error'.format(wav_path))
            feature = None
        return feature
        
    def load_norm(self, feature_mat, frame_slice):
        if feature_mat is None:
            return None
        if self.cmvn is not None and self.normalize_type == 1:
            feature_mat = (feature_mat + self.cmvn[0, :]) * self.cmvn[1, :]
        if frame_slice is not None and feature_mat is not None: 
            if feature_mat.shape[0] - frame_slice >= 0:
                index = random.randint(0, (feature_mat.shape[0] - frame_slice))
                feature_mat = feature_mat[index:index + frame_slice, :]
            else:
                feature_mat = None            
        return feature_mat
            
    def compute_cmvn_method(self, cmvn_num, utt_ids, frame_count, sum, sum_sq):         
        print(">> compute cmvn using {0} utterance ".format(cmvn_num))        
        cmvn_rand_idx = np.random.permutation(len(utt_ids))
        for n in tqdm(range(cmvn_num)):  
            audio_path = utt_ids[cmvn_rand_idx[n]][1]
            feature_mat = self.parse_audio(audio_path)                    
            if feature_mat is None:
                continue
            if isinstance(feature_mat, list):
                feature_mat = feature_mat[0] 
            sum_1utt = np.sum(feature_mat, axis=0)
            sum = np.add(sum, sum_1utt)
            feature_mat_square = np.square(feature_mat)
            sum_sq_1utt = np.sum(feature_mat_square, axis=0)
            sum_sq = np.add(sum_sq, sum_sq_1utt)
            frame_count += feature_mat.shape[0]            
        return frame_count, sum, sum_sq
               
    def compute_cmvn(self):
        if self.data_type == 'train': 
            audio_path = self.indices[0][0][1]
        else:
            for key in self.indices.keys():
                audio_path = self.indices[key][2]
        in_feat = self.parse_audio(audio_path)
        nfeatures = in_feat.shape[1]
        sum = np.zeros(shape=[1, nfeatures], dtype=np.float32)
        sum_sq = np.zeros(shape=[1, nfeatures], dtype=np.float32)
        cmvn = np.zeros(shape=[2, nfeatures], dtype=np.float32)
        frame_count = 0
               
        cmvn_num = min(self.wav_utt_size, self.num_utt_cmvn)        
        frame_count, sum, sum_sq = self.compute_cmvn_method(cmvn_num, self.wav_utt_ids, frame_count, sum, sum_sq)
                
        mean = sum / frame_count
        var = sum_sq / frame_count - np.square(mean)
        print (frame_count)
        print (mean)
        print (var)
        cmvn[0, :] = -mean
        cmvn[1, :] = 1 / np.sqrt(var)
        return cmvn

    def loading_cmvn(self):
        if not os.path.isdir(self.exp_path):
            raise Exception(self.exp_path + ' isn.t a path!')
        cmvn_file = os.path.join(self.exp_path, 'cmvn.npy')
        if self.data_type == 'train': 
            audio_path = self.indices[0][0][1]
        else:
            for key in self.indices.keys():
                audio_path = self.indices[key][2]
        in_feat = self.parse_audio(audio_path)
        if in_feat is None:
            raise Exception('Wav file {} is not exist!'.format(audio_path))  
        in_size = in_feat.shape[1]  # count nfeatures
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            if cmvn.shape[1] == in_size:
                print ('load cmvn from {}'.format(cmvn_file))
            else:
                cmvn = self.compute_cmvn()
                np.save(cmvn_file, cmvn)
                print ('original cmvn is wrong, so save new cmvn to {}'.
                        format(cmvn_file))
        else:
            cmvn = self.compute_cmvn()
            np.save(cmvn_file, cmvn)
            print ('save cmvn to {}'.format(cmvn_file))

        return cmvn
                                
class DeepSpeakerDataset(BaseDataset):
    def __init__(self, opt, data_dir):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        utt_id /path/to/audio.wav
        ...
        :param data_scp: Path to scp as describe above
        :param label_file : Dictionary containing the delta_order, context_width, normalize_type and max_num_utt_cmvn
        :param audio_conf: Dictionary containing the sample_rate, num_channel, window_size window_shift
        """            
        super(DeepSpeakerDataset, self).__init__(opt, data_dir)

    def __getitem__(self, index):
    
        utter_batch = []        
        utter_spk_ids_batch = []        
        if self.data_type == 'train': 
            frame_slice = np.random.randint(self.lb, self.ub)
        else:
            frame_slice = None
            
        if self.data_type != 'train':  
            pair = self.pairID[index]
            utt_id0, utt_id1, audio_path0, audio_path1, label = self.indices[pair]
            
            feature_mat0 = self.parse_audio(audio_path0)            
            feature_mat0 = self.load_norm(feature_mat0, frame_slice)      
            feature_mat1 = self.parse_audio(audio_path1)
            feature_mat0 = self.load_norm(feature_mat0, frame_slice)                                  
            return self.pairID[index], torch.FloatTensor(feature_mat0), torch.FloatTensor(feature_mat1), torch.IntTensor([label])                       
        else:  
            rand_idx = np.random.permutation(len(self.indices))[:self.speaker_num]            
            for x in rand_idx:
                selected_file = self.indices[x]                  
                utter_num = self.utter_num    
                feature_mats = np.zeros(shape=[utter_num, frame_slice, self.nfeatures], dtype=np.float32)
                spk_id_mats = np.zeros(shape=[utter_num, frame_slice], dtype=np.int64)                
                for num in range(utter_num):
                    feature_mat_slice = None
                    while feature_mat_slice is None:
                        index = random.randint(0, len(selected_file) - 1)
                        utt_id, audio_path = selected_file[index] 
                        feature_mat = self.parse_audio(audio_path)                                       
                        feature_mat_slice = self.load_norm(feature_mat, frame_slice) 
                        feature_mats[num, :, :] = feature_mat_slice   # each speakers utterance [M, frames, n_mels]
                        spk_id_mats[num, :] = np.array([self.utt2spk_ids[utt_id]] * frame_slice, dtype=np.int64)                           
                       
                utter_batch.append(feature_mats)
                utter_spk_ids_batch.append(spk_id_mats)
                del feature_mats
                    
            utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), frames, n_mels]
            utter_batch = np.transpose(utter_batch, axes=(1,0,2))     # transpose [frames, batch, n_mels] 
            utter_spk_ids_batch = np.concatenate(utter_spk_ids_batch, axis=0)                               
            return torch.FloatTensor(utter_batch), torch.LongTensor(utter_spk_ids_batch)
    
    def __len__(self):
        return self.wav_utt_size
        
                                
class DeepSpeakerDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DeepSpeakerDataLoader, self).__init__(*args, **kwargs)
        

class DeepSpeakerSeqDataset(BaseDataset):
    def __init__(self, opt, data_dir):
        
        self.frame_slice_steps = opt.frame_slice_steps                            
        super(DeepSpeakerSeqDataset, self).__init__(opt, data_dir)
        
    def __getitem__(self, index):
    
        utter_batch = []
        utter_spk_ids_batch = []
        utter_segment_num_batch = []
        if self.data_type == 'train': 
            frame_slice = np.random.randint(self.lb, self.ub)
        else:
            frame_slice = None
            
        if self.data_type != 'train':  
            pair = self.pairID[index]
            utt_id0, utt_id1, audio_path0, audio_path1, label = self.indices[pair]    
            feature_mat0 = self.parse_audio(audio_path0)
            feature_mat0 = self.load_norm(feature_mat0, frame_slice)                        
            feature_mat1 = self.parse_audio(audio_path1)
            feature_mat1 = self.load_norm(feature_mat1, frame_slice)                
            return self.pairID[index], torch.FloatTensor(feature_mat0), torch.FloatTensor(feature_mat1), torch.IntTensor([label])      
                    
        else:  
            rand_idx = np.random.permutation(len(self.indices))[:self.speaker_num]            
            for x in rand_idx:
                selected_file = self.indices[x]                 
                utter_num = self.utter_num    
                feature_mats = np.zeros(shape=[0, frame_slice, self.nfeatures], dtype=np.float32)
                spk_id_mats = np.zeros(shape=[0, frame_slice], dtype=np.int64)
                for num in range(utter_num):
                    feature_mat_slice = None
                    while feature_mat_slice is None:
                        index = random.randint(0, len(selected_file) - 1)
                        utt_id, audio_path = selected_file[index] 
                        feature_mat = self.parse_audio(audio_path)
                        feature_mat_slice = self.load_norm(feature_mat, frame_slice=self.frame_slice_steps)             
                    segment_num = 0  
                    feature_mat = self.load_norm(feature_mat, frame_slice=None)   
                    for start in range(0, feature_mat.shape[0], self.frame_slice_steps):
                        end = start + frame_slice
                        if end < feature_mat.shape[0] and segment_num < 25: 
                            feature_mat_slice = feature_mat[start:end, :]
                            feature_mat_slice = feature_mat_slice[np.newaxis, :]
                            feature_mats = np.concatenate((feature_mats, feature_mat_slice), axis=0)
                            spk_id_mat = np.array([self.utt2spk_ids[utt_id]] * frame_slice, dtype=np.int64)
                            spk_id_mat = spk_id_mat[np.newaxis, :]
                            spk_id_mats = np.concatenate((spk_id_mats, spk_id_mat), axis=0)
                            segment_num += 1
                    utter_segment_num_batch.append(segment_num)   
                                            
                utter_batch.append(feature_mats)
                utter_spk_ids_batch.append(spk_id_mats)
                del feature_mats, spk_id_mats              
            utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), frames, n_mels]
            utter_batch = np.transpose(utter_batch, axes=(1,0,2))     # transpose [frames, batch, n_mels]  
            utter_segment_num_batch = np.array(utter_segment_num_batch, dtype=np.int64)
            utter_spk_ids_batch = np.concatenate(utter_spk_ids_batch, axis=0)                           
            return torch.FloatTensor(utter_batch), torch.LongTensor(utter_segment_num_batch), torch.LongTensor(utter_spk_ids_batch)
            
    def __len__(self):
        return self.wav_utt_size
        
                            
class DeepSpeakerSeqDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DeepSpeakerSeqDataLoader, self).__init__(*args, **kwargs)
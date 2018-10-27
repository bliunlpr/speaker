from __future__ import print_function
import argparse
import os
import random
import shutil
import psutil
import time 
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import DeepSpeakerModel, DeepSpeakerSeqModel, similarity, loss_cal, normalize, penalty_loss_cal, similarity_segment, loss_cal_segment, penalty_seq_loss_cal
from config import TrainOptions
from data_loader import DeepSpeakerDataset, DeepSpeakerDataLoader, DeepSpeakerSeqDataset, DeepSpeakerSeqDataLoader
import utils 

opt = TrainOptions().parse()
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)    
logging = utils.create_output_dir(opt)

print (opt.gpu_ids)
device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")

# data
logging.info("Building dataset.")
if opt.seq_training == 'true':
    opt.data_type = 'train'
    train_dataset = DeepSpeakerSeqDataset(opt, os.path.join(opt.dataroot, 'train'))                         
    train_loader = DeepSpeakerSeqDataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers,
                                         shuffle=True, pin_memory=True)
    opt.data_type = 'dev'                                               
    val_dataset = DeepSpeakerSeqDataset(opt, os.path.join(opt.dataroot, 'dev'))
    val_loader = DeepSpeakerSeqDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers,
                                          shuffle=False, pin_memory=True)
else:
    opt.data_type = 'train'
    train_dataset = DeepSpeakerDataset(opt, os.path.join(opt.dataroot, 'train'))                         
    train_loader = DeepSpeakerDataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers,
                                         shuffle=True, pin_memory=True)
    opt.data_type = 'dev'                                               
    val_dataset = DeepSpeakerDataset(opt, os.path.join(opt.dataroot, 'dev'))
    val_loader = DeepSpeakerDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers,
                                       shuffle=False, pin_memory=True)
logging.info("Dataset ready!")

def train(opt, model, optimizer):      
    model.train()
    total_steps = opt.total_steps
    losses = utils.AverageMeter()
    embedding_losses = utils.AverageMeter()
    penalty_losses = utils.AverageMeter()
    lr = opt.lr
        
    for i, (data) in enumerate(train_loader, start=0): 
        if opt.seq_training == 'true':
            feature_input, seq_len, spk_ids = data
            feature_input = feature_input.squeeze(0).to(device)
            seq_len = seq_len.squeeze(0).to(device)
            spk_ids = spk_ids.squeeze(0).to(device)
            if opt.train_type == 'multi_attention':
                outputs, attention_matrix = model(feature_input, seq_len)
                sim_matrix = similarity(outputs, model.w, model.b, opt)
                embedding_loss = opt.embedding_loss_lamda * loss_cal(sim_matrix, opt)
                penalty_loss = opt.penalty_loss_lamda * penalty_seq_loss_cal(attention_matrix, device)
                loss = embedding_loss + penalty_loss
                penalty_losses.update(penalty_loss.item())
            else:
                outputs = model(feature_input, seq_len)
                sim_matrix = similarity(outputs, model.w, model.b, opt)
                embedding_loss = opt.embedding_loss_lamda * loss_cal(sim_matrix, opt)
                loss = embedding_loss 
        else:
            feature_input, spk_ids = data   
            spk_ids = spk_ids.squeeze(0).to(device)
            feature_input = feature_input.squeeze(0).to(device)

            if opt.train_type == 'multi_attention':
                outputs, attention_matrix = model(feature_input)
                sim_matrix  = similarity(outputs, model.w, model.b, opt)
                embedding_loss = opt.embedding_loss_lamda * loss_cal(sim_matrix, opt)
                penalty_loss = opt.penalty_loss_lamda * penalty_seq_loss_cal(attention_matrix, device)
                loss = embedding_loss + penalty_loss
                penalty_losses.update(penalty_loss.item())
            else:
                outputs = model(feature_input)
                sim_matrix = similarity(outputs, model.w, model.b, opt)
                embedding_loss = opt.embedding_loss_lamda * loss_cal(sim_matrix, opt)
                loss = embedding_loss 
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if utils.check_grad(model.parameters(), opt.clip_grad, opt.ignore_grad):
            logging.info('Not a finite gradient or too big, ignoring.')
            optimizer.zero_grad()
            continue
        optimizer.step()
        
        losses.update(loss.item())
        embedding_losses.update(embedding_loss.item())
              
                     
        if total_steps % opt.print_freq == 0:
            logging.info('  ==> Train set steps {} lr: {:.6f}, loss: {:.4f} [ embedding: {:.4f}, penalty_loss {:.4f}]'
                         .format(total_steps, lr, losses.avg, embedding_losses.avg, penalty_losses.avg))     
            losses.reset()
            embedding_losses.reset()
            penalty_losses.reset()
            state = {'state_dict': model.state_dict(), 'opt': opt,                                             
                     'learning_rate': lr, 'total_steps': total_steps}
            filename = 'latest'
            utils.save_checkpoint(state, opt.expr_dir, filename=filename)  
            
        if total_steps % opt.validate_freq == 0:
            EER = evaluate(opt, model)     
            lr = utils.adjust_learning_rate_by_factor(optimizer, lr, opt.lr_reduce_factor)            
            state = {'state_dict': model.state_dict(), 'opt': opt,                                             
                     'learning_rate': lr, 'total_steps': total_steps}
            filename='steps-{}_lr-{:.6f}_EER-{:.4f}.pth'.format(total_steps, lr, EER)
            utils.save_checkpoint(state, opt.expr_dir, filename=filename)
            model.train() 
        total_steps += 1        
        
        if total_steps > opt.training_total_steps:
            logging.info('finish training, total_steps is  {}'.format(total_steps))
            break          
                    
def evaluate(opt, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()
    frame_slice = int((opt.lb + opt.ub) / 4)
    valid_enum = tqdm(val_loader, desc='Valid')
    model.eval()
    embedding_mean_probs = []
    score_mean_probs = []
    probs = []
    labels = []
    for i, (data) in enumerate(valid_enum, start=0):  
        with torch.no_grad():
            pair, data_a, data_b, label = data
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            data_a = data_a.transpose(0, 1)
            data_b = data_b.transpose(0, 1)
            input_a = None
            seq_a = 0
            length = data_a.size(0)
            for x in range(0, length, frame_slice):
                end = x + frame_slice * 2
                if end < length:
                    feature_mat = data_a[x:end, :, :]
                else:
                    if x == 0:
                        input_a = data_a
                        seq_a += 1
                    break
                seq_a += 1
                if input_a is None:
                    input_a = feature_mat
                else:
                    input_a = torch.cat((input_a, feature_mat), 1)
            input_a = input_a.to(device)
            seq_a = torch.LongTensor([seq_a]).to(device)
            if opt.seq_training == 'true':
                if opt.train_type == 'multi_attention':
                    out_a, _ = model(input_a, seq_a)
                else:
                    out_a = model(input_a, seq_a)
            else:
                if opt.train_type == 'multi_attention':
                    out_a, _ = model(input_a)
                else:
                    out_a = model(input_a)

            input_b = None
            seq_b = 0
            length = data_b.size(0)
            for x in range(0, length, frame_slice):
                end = x + frame_slice * 2
                if end < length:
                    feature_mat = data_b[x:end, :, :]
                else:
                    if x == 0:
                        input_b = data_b
                        seq_b += 1
                    break
                seq_b += 1
                if input_b is None:
                    input_b = feature_mat
                else:
                    input_b = torch.cat((input_b, feature_mat), 1)
            input_b = input_b.to(device)
            seq_b = torch.LongTensor([seq_b]).to(device)
            if opt.seq_training == 'true':
                if opt.train_type == 'multi_attention':
                    out_b, _ = model(input_b, seq_b)
                else:
                    out_b = model(input_b, seq_b)
            else:
                if opt.train_type == 'multi_attention':
                    out_b, _ = model(input_b)
                else:
                    out_b = model(input_b)
            
            out_a = torch.mean(out_a, 0)
            out_b = torch.mean(out_b, 0)
            prob = out_a * out_b
            prob = torch.sum(prob)
            prob = torch.abs(model.w) * prob + model.b
            result = float(F.sigmoid(prob).detach().cpu().numpy())
            embedding_mean_probs.append(result)
            labels.append(int(label.detach().cpu().numpy()))

    embedding_mean_eer, embedding_mean_thresh = utils.processDataTable2(np.array(labels), np.array(embedding_mean_probs))
    logging.info("embedding_mean_EER : %0.4f (thres:%0.4f)" % (embedding_mean_eer, embedding_mean_thresh))
    eer = embedding_mean_eer
    return eer

def main():
    
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if not os.path.isfile(model_path):
            raise Exception("no checkpoint found at {}".format(model_path))
        
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        opt.lr = package.get('learning_rate', opt.lr)
        opt.total_steps = int(package.get('total_steps', 0)) - 1 
        print('total_steps is {}'.format(opt.total_steps))
        
        if opt.seq_training == 'true':
            model = DeepSpeakerSeqModel.load_model(model_path, 'state_dict')
        else:
            model = DeepSpeakerModel.load_model(model_path, 'state_dict') 
        logging.info('Loading model {}'.format(model_path))
    else:
        if opt.seq_training == 'true':
            model = DeepSpeakerSeqModel(opt)
        else:
            model = DeepSpeakerModel(opt)
    
    print (model)
    for k, v in model.state_dict().items():
       print(k, v.shape)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)              
    train(opt, model, optimizer)

if __name__ == '__main__':
    main()

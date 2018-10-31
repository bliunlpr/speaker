import argparse
import os
import utils
import torch


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--works_dir', help='path to work', default='.')
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, dev, test)')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        self.parser.add_argument('--num_features', type=int, default=13, help='num_features of a frame')
        self.parser.add_argument('--left_context_width', type=int, default=0, help='input left_context_width-width')
        self.parser.add_argument('--right_context_width', type=int, default=0, help='input right_context_width')
        self.parser.add_argument('--delta_order', type=int, default=0, help='input delta-order')
        self.parser.add_argument('--normalize_type', type=int, default=1, help='normalize_type')
        
        self.parser.add_argument('--num_utt_cmvn', type=int, help='the number of utterances for cmvn', default=20000)
        self.parser.add_argument('--num_utt_per_loading', type=int, help='the number of utterances one loading', default=200)
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
        self.parser.add_argument('--batch_size', type=int, default=256, help='the batch size for one training')              
        self.parser.add_argument('--speaker_num', type=int, default=64, help='speaker_num')        
        self.parser.add_argument('--utter_num', type=int, default=10, help='utter_num')
        self.parser.add_argument('--frame_slice_steps', type=int, default=50, help='frame_slice_steps')                                 

        self.parser.add_argument('--train_type', type=str, default='last_state', help='train_type, last_state|average_state|base_attention|multi_attention')   
        self.parser.add_argument('--segment_type', type=str, default='none', help='train_type, average|all|none')    
        self.parser.add_argument('--model_type', type=str, default='lstm', help='model_type, lstm|cnn')         
        self.parser.add_argument('--rnn_hidden_size', type=int, default=128, help='rnn_hidden_size')
        self.parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size')       
        self.parser.add_argument('--nb_layers', type=int, default=3, help='dnn_num_layer')        
        self.parser.add_argument('--dropout', type=float, default=0, help='ignore grad before clipping')
        self.parser.add_argument('--rnn_type', default='lstm', help='Type of the functions. relu|sigmoid|tanh are supported')
        self.parser.add_argument('--bidirectional', default=False, action='store_true', help='bidirectional to dnn')
        self.parser.add_argument('--attention_dim', type=int, default=100, help='attention_dim')        
        self.parser.add_argument('--attention_head_num', type=int, default=3, help='attention_head_num')
        self.parser.add_argument('--embedding_loss_lamda', type=float, default=1., help='class_loss_lamda')
        self.parser.add_argument('--penalty_loss_lamda', type=float, default=0.001, help='penalty_loss_lamda') 
        self.parser.add_argument('--segment_loss_lamda', type=float, default=0.2, help='segment_loss_lamda')  
        self.parser.add_argument('--seq_training', type=str, default='False', help='seq_training to dnn')
                        
        self.parser.add_argument('--name', type=str, default='speaker', help='name of the experiment.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')  
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')    
        self.parser.add_argument('--total_steps', default=0, type=int, metavar='N', help='manual hours number (useful on restarts)')
        self.parser.add_argument('--training_total_steps', default=10000, type=int, metavar='N', help='manual hours number (useful on restarts)')                
        self.parser.add_argument('--validate_freq', type=int, default=1000, help='how many batches to validate the trained model')   
        self.parser.add_argument('--print_freq', type=int, default=50, help='how many batches to print the trained model') 
        self.parser.add_argument('--data_type', type=str, default='train', help='data_type')  
        self.parser.add_argument('--lb', type=int, default=90, help='how many batches to print the trained model') 
        self.parser.add_argument('--ub', type=int, default=120, help='how many batches to print the trained model') 
          
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
        self.parser.add_argument('--loss_type', type=str, default='softmax', help='train_type, softmax|contrast|margin')  
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
        self.parser.add_argument('--clip-grad', type=float, default=3.0, help='maximum norm of gradient clipping')
        self.parser.add_argument('--ignore-grad', type=float, default=100000.0, help='ignore grad before clipping')
        self.parser.add_argument('--lr_reduce_factor', default=0.9, type=float, help='lr_reduce_factor')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        ##if len(self.opt.gpu_ids) > 0:
        ##    torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(expr_dir)
        self.opt.expr_dir = expr_dir
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


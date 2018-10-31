import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from collections import OrderedDict
import math
import os
import numpy as np
import torch.utils.model_zoo as model_zoo

device = torch.device("cuda")
supported_acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(), 'tanh': nn.Tanh(), 
                  'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU()}

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

class ModelBase(nn.Module):
    """
    ModelBase class for sharing code among various model.
    """
    def forward(self, x):
        raise NotImplementedError
    
    @classmethod
    def load_model(cls, path, state_dict):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(opt=package['opt'])
        if package[state_dict] is not None:
            model.load_state_dict(package[state_dict])    
        return model
        
    @staticmethod
    def serialize(model, state_dict, optimizer=None, optim_dict=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'opt': model.opt,
            'state_dict': model.state_dict()            
        }
        if optimizer is not None:
            package[optim_dict] = optimizer.state_dict()
        return package
        
    @staticmethod    
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
        
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
            
class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, dropout=0, batch_norm=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        ##self.rnn.flatten_parameters()
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DeepSpeakerModel(ModelBase):
    def __init__(self, opt):
        super(DeepSpeakerModel, self).__init__()

        self._version = '0.0.1'
        self._hidden_size = opt.rnn_hidden_size
        self._embedding_size = opt.embedding_size
        self._hidden_layers = opt.nb_layers
        self._rnn_type = supported_rnns[opt.rnn_type]
        self._bidirectional = opt.bidirectional
        self._dropout = opt.dropout
        self.train_type = opt.train_type
        self.w = nn.Parameter(torch.FloatTensor(np.array([10])))
        self.b = nn.Parameter(torch.FloatTensor(np.array([-5])))
        rnn_input_size = opt.num_features * (opt.delta_order + 1) * (opt.left_context_width + opt.right_context_width + 1)

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=self._hidden_size, rnn_type=self._rnn_type,
                       bidirectional=self._bidirectional, dropout=self._dropout, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self._hidden_layers - 2):
            rnn = BatchRNN(input_size=self._hidden_size, hidden_size=self._hidden_size, rnn_type=self._rnn_type,
                           bidirectional=self._bidirectional, dropout=self._dropout)
            rnns.append(('%d' % (x + 1), rnn))
        if self.train_type == 'divide_attention':
            rnn = BatchRNN(input_size=self._hidden_size, hidden_size=2 * self._hidden_size, rnn_type=self._rnn_type,
                           bidirectional=self._bidirectional, dropout=0)
        else:
            rnn = BatchRNN(input_size=self._hidden_size, hidden_size=self._hidden_size, rnn_type=self._rnn_type,
                           bidirectional=self._bidirectional, dropout=0)
        rnns.append(('%d' % (self._hidden_layers - 1), rnn))
            
        self.rnns = nn.Sequential(OrderedDict(rnns))                
        self.fc = nn.Linear(self._hidden_size, self._embedding_size, bias=False)
        
        if self.train_type == 'base_attention':
            self.query = nn.Conv1d(self._hidden_size, 1, 1)
                    
        elif self.train_type == 'multi_attention' or self.train_type == 'divide_attention':
            self._attention_dim = opt.attention_dim
            self._attention_head_num = opt.attention_head_num
            self.w1 = SequenceWise(nn.Sequential(nn.Linear(self._hidden_size, self._attention_dim, bias=False),
                                                 nn.ReLU(),
                                                 nn.Linear(self._attention_dim, self._attention_head_num, bias=False)))
         
    def forward(self, x):
        attn = None
        x = self.rnns(x)
        if self.train_type == 'last_state':
            x = x[-1]
            x = self.fc(x)            
        elif self.train_type == 'average_state':
            x = torch.mean(x, dim=0)
            x = self.fc(x)
        elif self.train_type == 'base_attention':
            x = x.transpose(0, 1).transpose(1, 2)
            query = self.query(x).permute(0, 2, 1)
            query = F.softmax(query, 1)
            query = torch.bmm(x, query)        
            x = self.fc(query.squeeze())
        elif self.train_type == 'multi_attention':
            attn = self.w1(x)
            attn = F.softmax(attn, 0).transpose(0, 1)
            x = torch.bmm(x.transpose(0, 1).transpose(1, 2), attn) 
            x = x.view(x.size(0), -1)      
        elif self.train_type == 'divide_attention':
            x_a = x[:, :, :self._hidden_size]
            x_b = x[:, :, self._hidden_size:]
            attn = self.w1(x_b)
            attn = F.softmax(attn, 0).transpose(0, 1)
            x = torch.bmm(x_a.transpose(0, 1).transpose(1, 2), attn) 
            x = x.view(x.size(0), -1)      
        x = normalize(x)
        return x, attn        
            
            
class DeepSpeakerSeqModel(ModelBase):
    def __init__(self, opt):
        super(DeepSpeakerSeqModel, self).__init__()
        self._version = '0.0.1'
        self._hidden_size = opt.rnn_hidden_size
        self._embedding_size = opt.embedding_size
        self._hidden_layers = opt.nb_layers
        self._rnn_type = supported_rnns[opt.rnn_type]
        self._bidirectional = opt.bidirectional
        self._dropout = opt.dropout
        self.train_type = opt.train_type
        self.segment_type = opt.segment_type
        self.w = nn.Parameter(torch.FloatTensor(np.array([10])))
        self.b = nn.Parameter(torch.FloatTensor(np.array([-5])))
        rnn_input_size = opt.num_features * (opt.delta_order + 1) * (opt.left_context_width + opt.right_context_width + 1)

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=self._hidden_size, rnn_type=self._rnn_type,
                       bidirectional=self._bidirectional, dropout=self._dropout, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self._hidden_layers - 2):
            rnn = BatchRNN(input_size=self._hidden_size, hidden_size=self._hidden_size, rnn_type=self._rnn_type,
                           bidirectional=self._bidirectional, dropout=self._dropout)
            rnns.append(('%d' % (x + 1), rnn))
        if self.train_type == 'divide_attention':
            rnn = BatchRNN(input_size=self._hidden_size, hidden_size=2 * self._hidden_size, rnn_type=self._rnn_type,
                           bidirectional=self._bidirectional, dropout=0)
        else:
            rnn = BatchRNN(input_size=self._hidden_size, hidden_size=self._hidden_size, rnn_type=self._rnn_type,
                           bidirectional=self._bidirectional, dropout=0)
        rnns.append(('%d' % (self._hidden_layers - 1), rnn))
            
        self.rnns = nn.Sequential(OrderedDict(rnns))                
        self.fc = nn.Linear(self._hidden_size, self._embedding_size, bias=False)
        
        if self.train_type == 'base_attention':
            self.query = nn.Sequential(nn.Linear(self._hidden_size, 1))
        
        if self.train_type == 'multi_attention' or self.train_type == 'divide_attention':
            self._attention_dim = opt.attention_dim
            self._attention_head_num = opt.attention_head_num
            self.w1 = nn.Sequential(nn.Linear(self._hidden_size, self._attention_dim, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(self._attention_dim, self._attention_head_num, bias=False))
         
    def forward(self, x, segment_num):
        x = self.rnns(x)
        assert x.size(1) == torch.sum(segment_num)   
        out = None
        attn_i = None
        attn = None
        out_segment = None
        start = 0
        for i in range(segment_num.size(0)):
            end = start + int(segment_num[i])
            out_i = x[-1, start:end, :].squeeze(0)
            start += int(segment_num[i])    
            if len(out_i.shape) == 1:
                out_i = out_i.unsqueeze(0) 
                 
            if self.segment_type == 'average':
                out_segment_i = torch.mean(out_i, dim=0, keepdim=True)
            elif self.segment_type == 'all':
                out_segment_i = out_i
            else:
                out_segment_i = None 
                            
            if self.train_type == 'last_state':
                out_i = out_i[-1, :]
                out_i = self.fc(out_i)
            elif self.train_type == 'average_state':
                out_i = torch.mean(out_i, dim=0)
                out_i = self.fc(out_i)
            elif self.train_type == 'base_attention':
                query = self.query(out_i).squeeze(1) 
                query = F.softmax(query, 0)
                out_i = out_i.transpose(0, 1)
                out_i = torch.mv(out_i, query)      
                out_i = self.fc(out_i)
            elif self.train_type == 'multi_attention':
                attn_i = self.w1(out_i)
                attn_i = F.softmax(attn_i, 0)
                out_i = torch.mm(out_i.transpose(0, 1), attn_i) 
                out_i = out_i.view(-1)     
            elif self.train_type == 'divide_attention':
                x_a = out_i[:, :self._hidden_size]
                x_b = out_i[:, self._hidden_size:]
                attn_i = self.w1(x_b)
                attn_i = F.softmax(attn_i, 0)
                out_i = torch.mm(x_a.transpose(0, 1), attn_i) 
                out_i = out_i.view(-1)      
                
            out_i = out_i.unsqueeze(0)
            if out is None:
                out = out_i
            else:
                out = torch.cat((out, out_i), 0)
                                 
            if attn_i is not None:
                attn_i = torch.mm(attn_i.transpose(0, 1), attn_i).unsqueeze(0)
                if attn is None:
                    attn = attn_i
                else:
                    attn = torch.cat((attn, attn_i), 0) 
                    
            if out_segment_i is not None:
                if out_segment is None:
                    out_segment = out_segment_i
                else:
                    out_segment = torch.cat((out_segment, out_segment_i), 0)
                                                   
        out = normalize(out)
        if out_segment is not None:
            out_segment = normalize(out_segment)        
        return out, attn, out_segment
        
        
class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
                
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual        
        return out


class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet, self).__init__()
        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        
        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d([4,1])
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepSpeakerCnnModel(ModelBase):

    def __init__(self, opt):

        super(DeepSpeakerCnnModel, self).__init__()        
        self.embedding_size = opt.embedding_size
        self.model = myResNet(BasicBlock, [2, 2, 2, 2], self.embedding_size)
        self.w = nn.Parameter(torch.FloatTensor(np.array([10])))
        self.b = nn.Parameter(torch.FloatTensor(np.array([-5])))
        
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = normalize(x)

        return self.features

        
class DeepSpeakerCnnSeqModel(ModelBase):

    def __init__(self, opt):

        super(DeepSpeakerCnnSeqModel, self).__init__()        
        self.embedding_size = opt.embedding_size
        self.train_type = opt.train_type
        self.segment_type = opt.segment_type
        self.model = myResNet(BasicBlock, [2, 2, 2, 2], self.embedding_size)
        self.w = nn.Parameter(torch.FloatTensor(np.array([10])))
        self.b = nn.Parameter(torch.FloatTensor(np.array([-5])))
        
        if self.train_type == 'base_attention':
            self.query = nn.Sequential(nn.Linear(self.embedding_size, 1))

        if self.train_type == 'multi_attention' or self.train_type == 'divide_attention':
            self.attention_dim = opt.attention_dim
            self.attention_head_num = opt.attention_head_num
            self.w1 = nn.Sequential(nn.Linear(self.embedding_size, self.attention_dim, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(self.attention_dim, self.attention_head_num, bias=False))
            

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
    
    def forward(self, x, segment_num):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        
        assert x.size(0) == torch.sum(segment_num), print(x.size(), segment_num)
        out = None
        out_segment = None
        attn_i = None
        attn = None
        start = 0
        for i in range(segment_num.size(0)):
            end = start + int(segment_num[i])
            out_i = x[start:end, :]
            start += int(segment_num[i])
            if len(out_i.shape) == 1:
                out_i = out_i.unsqueeze(0)

            if self.segment_type == 'average':
                out_segment_i = torch.mean(out_i, dim=0, keepdim=True)
            elif self.segment_type == 'all':
                out_segment_i = out_i
            else:
                out_segment_i = None

            if self.train_type == 'average_state':
                out_i = torch.mean(out_i, dim=0)
            elif self.train_type == 'base_attention':
                query = self.query(out_i).squeeze(1)
                query = F.softmax(query, 0)
                out_i = out_i.transpose(0, 1)
                out_i = torch.mv(out_i, query)
            elif self.train_type == 'multi_attention':
                attn_i = self.w1(out_i)
                attn_i = F.softmax(attn_i, 0)
                out_i = torch.mm(out_i.transpose(0, 1), attn_i) 
                out_i = out_i.view(-1)     
            elif self.train_type == 'divide_attention':
                x_a = out_i[:, :self._hidden_size]
                x_b = out_i[:, self._hidden_size:]
                attn_i = self.w1(x_b)
                attn_i = F.softmax(attn_i, 0)
                out_i = torch.mm(x_a.transpose(0, 1), attn_i)
                out_i = out_i.view(out_i.size(0), -1)

            out_i = out_i.unsqueeze(0)
            if out is None:
                out = out_i
            else:
                out = torch.cat((out, out_i), 0)

            if out_segment_i is not None:
                if out_segment is None:
                    out_segment = out_segment_i
                else:
                    out_segment = torch.cat((out_segment, out_segment_i), 0)

            if attn_i is not None:
                attn_i = torch.mm(attn_i.transpose(0, 1), attn_i).unsqueeze(0)
                if attn is None:
                    attn = attn_i
                else:
                    attn = torch.cat((attn, attn_i), 0)

        out = normalize(out)
        if out_segment is not None:
            out_segment = normalize(out_segment)
            
        return out, attn, out_segment
          
                                
def normalize(x):
    """ normalize the last dimension vector of the input matrix .unsqueeze(0)
    :return: normalized input
    """
    return x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-6)     

def similarity(embedded, w, b, opt, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    N = opt.speaker_num
    M = opt.utter_num 
    ##S = opt.segment_num
    if opt.train_type == 'multi_attention' or opt.train_type == 'divide_attention':
        P = opt.embedding_size * opt.attention_head_num   
    else: 
        P = opt.embedding_size
    ##embedded_mean = torch.cat([torch.mean(embedded[i*S:(i+1)*S,:], dim=0, keepdim=True) for i in range(N*M)], dim=0)
    embedded_split = torch.reshape(embedded, (N, M, P))

    if center is None:
        center = normalize(torch.mean(embedded_split, dim=1))              # [N,P] normalized center vectors eq.(1)
        center_except = normalize(torch.reshape(torch.sum(embedded_split, dim=1, keepdim=True)
                                             - embedded_split, (N*M,P)))  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = torch.cat(
            [torch.cat([torch.sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], dim=1, keepdim=True) if i==j
                        else torch.sum(center[i:(i+1),:]*embedded_split[j,:,:], dim=1, keepdim=True) for i in range(N)],
                       dim=1) for j in range(N)], dim=0)
    else :
        # If center(enrollment) exist, use it.
        S = torch.cat(
            [torch.cat([torch.sum(center[i:(i + 1), :] * embedded_split[j, :, :], dim=1, keepdim=True) for i
                        in range(N)], dim=1) for j in range(N)], dim=0)

    S = torch.abs(w)*S + b   # rescaling

    return S

def loss_cal(S, opt):
    """ calculate loss with similarity matrix(S) eq.(6) (7) 
    :type: "softmax" or "contrast"
    :return: loss
    """
    N = opt.speaker_num
    M = opt.utter_num 
    loss_type = opt.loss_type
    S_correct = torch.cat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], dim=0)  # colored entries in Fig.1

    if loss_type == "softmax":
        total = -torch.sum(S_correct-torch.log(torch.sum(torch.exp(S), dim=1, keepdim=True) + 1e-6))
    elif loss_type == "contrast":
        S_sig = torch.sigmoid(S)
        S_sig = torch.cat([torch.cat([0*S_sig[i*M:(i+1)*M, j:(j+1)] if i==j
                              else S_sig[i*M:(i+1)*M, j:(j+1)] for j in range(N)], dim=1)
                             for i in range(N)], dim=0)
        total = torch.sum(1-torch.sigmoid(S_correct)+torch.max(S_sig, dim=1, keepdim=True)[0])
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total

def similarity_segment(embedded, seq_len, w, b, opt, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N) normalize(
    """
    N = opt.speaker_num
    M = opt.utter_num
    assert embedded.size(0) == torch.sum(seq_len), print(embedded.size(), seq_len)

    s, seq_index, seq_utter_index = 0, [], []
    for i in range(seq_len.size(0)):
        seq_index.append(s)
        if i % M == 0:
            seq_utter_index.append(s)
        s += int(seq_len[i])
    seq_index.append(s)
    seq_utter_index.append(s)

    center = normalize(
        torch.cat([torch.mean(embedded[seq_utter_index[i]:seq_utter_index[i + 1], :], dim=0, keepdim=True) for i
                   in range(len(seq_utter_index) - 1)], dim=0))

    center_except = normalize(torch.cat([(torch.sum(embedded[seq_utter_index[i]:seq_utter_index[i + 1], :], dim=0,keepdim=True)
                                          - embedded[seq_utter_index[i]:seq_utter_index[i + 1],:]) / (seq_utter_index[i + 1] - seq_utter_index[i] - 1)
                                         for i in range(len(seq_utter_index) - 1)], dim=0))

    S = torch.cat(
        [torch.cat(
            [torch.sum(center_except[seq_utter_index[i]:seq_utter_index[i + 1], :] * embedded[seq_utter_index[j]:seq_utter_index[j + 1], :],
                       dim=1, keepdim=True) if i == j else torch.sum(center[i:(i + 1), :] * embedded[seq_utter_index[j]:seq_utter_index[j + 1], :],
                       dim=1, keepdim=True) for i in range(N)], dim=1) for j in range(N)], dim=0)

    S = torch.abs(w) * S + b  # rescaling

    return S
    
def loss_cal_segment(S, seq_len, opt):
    """ calculate loss with similarity matrix(S) eq.(6) (7)
    :type: "softmax" or "contrast"
    :return: loss
    """
    N = opt.speaker_num
    M = opt.utter_num
    loss_type = opt.loss_type

    assert S.size(0) == torch.sum(seq_len), print(S.size(), seq_len)
    assert N * M == seq_len.size(0), print(N, M, seq_len)

    s, seq_index, seq_utter_index = 0, [], []
    for i in range(seq_len.size(0)):
        seq_index.append(s)
        if i % M == 0:
            seq_utter_index.append(s)
        s += int(seq_len[i])
    seq_index.append(s)
    seq_utter_index.append(s)

    S_correct = torch.cat([S[seq_utter_index[i]:seq_utter_index[i + 1], i:(i + 1)] for i in range(N)],
                          dim=0)  # colored entries in Fig.1

    if loss_type == "softmax":
        total = -torch.sum(S_correct - torch.log(torch.sum(torch.exp(S), dim=1, keepdim=True) + 1e-6))
    elif loss_type == "contrast":
        S_sig = torch.sigmoid(S)
        S_sig = torch.cat([torch.cat([0 * S_sig[i * M:(i + 1) * M, j:(j + 1)] if i == j
                                      else S_sig[i * M:(i + 1) * M, j:(j + 1)] for j in range(N)], dim=1)
                           for i in range(N)], dim=0)
        total = torch.sum(1 - torch.sigmoid(S_correct) + torch.max(S_sig, dim=1, keepdim=True)[0])
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    total = total * int(seq_len.size(0)) / float(torch.sum(seq_len))
    return total
    
def penalty_loss_cal(A, device):
    loss_call = torch.nn.MSELoss(size_average=False)
    I = torch.eye(A.size(2)).to(device)
    out = torch.bmm(A.transpose(1, 2), A)
    return loss_call(out, I.expand_as(out))
    
def penalty_seq_loss_cal(A, device):
    print(A.shape)
    loss_call = torch.nn.MSELoss(size_average=False)
    I = torch.eye(A.size(1)).to(device)
    print(I.expand_as(A).shape)
    return loss_call(A, I.expand_as(A))

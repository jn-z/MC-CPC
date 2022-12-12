from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb
## PyTorch implementation of CDCK2, CDCK3, CDCK6, CDCK_FUSION
# CDCK5: oroginal CPC model
# CDCK2: lightweight encoder with Representation Learning with Contrastive Predictive Coding
# CDCK3: 3 channel decom signal with three channel CDCK2
# CDCK6: original forward signal and its reverse signal with double channel CDCK2
# CDCK_FUSION: original time domain signal, the frequency domain signal, and features extracted from shallow networks with three channel CDCK2.

class CDCK6(nn.Module):
    ''' original forward signal and its reverse signal with double channel CDCK2 '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK6, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru1 = nn.GRU(512, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.gru2 = nn.GRU(512, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru1 and gru2
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden1(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 128).cuda()
        else: return torch.zeros(1, batch_size, 128)

    def init_hidden2(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 128).cuda()
        else: return torch.zeros(1, batch_size, 128)

    def forward(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]
        nce = 0
        t_samples = torch.randint(int(self.seq_len/50-self.timestep), size=(1,)).long()
        # first gru
        z = self.encoder(x)
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512)
        forward_seq = z[:,:t_samples+1,:]
        output1, hidden1 = self.gru1(forward_seq, hidden1)
        c_t = output1[:,t_samples,:].view(batch, 128)
        pred = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk1[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        # second gru
        z = self.encoder(x_reverse)
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512)
        forward_seq = z[:,:t_samples+1,:]
        output2, hidden2 = self.gru2(forward_seq, hidden2)
        c_t = output2[:,t_samples,:].view(batch, 128)
        pred = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk2[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

        nce /= -1.*batch*self.timestep
        nce /= 2.
        accuracy = 1.*(correct1.item()+correct2.item())/(batch*2) # accuracy over batch and two grus

        return accuracy, nce, hidden1, hidden2

    def predict(self, x, x_reverse, hidden1, hidden2):
        # first gru
        z1 = self.encoder(x)
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1) # output size e.g. 8*128*256

        # second gru
        z2 = self.encoder(x_reverse)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)

        return torch.cat((output1, output2), dim=2)


class CDCK5(nn.Module):
    ''' Original CDCK'''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK5, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gru = nn.GRU(512, 40, num_layers=2, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(40, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size):
        return torch.zeros(2*1, batch_size, 40)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(self.seq_len/50-self.timestep, size=(1,)).long()
        z = self.encoder(x)
        z = z.transpose(1,2)
        nce = 0
        encode_samples = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512)
        forward_seq = z[:,:t_samples+1,:]
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:,t_samples,:].view(batch, 40)
        pred = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(0, self.timestep):
            decoder = self.Wk[i]
            pred[i] = decoder(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        z = self.encoder(x)
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden)

        return output, hidden


class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK2, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, 256).cuda()
        else: return torch.zeros(1, batch_size, 256)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.seq_len/20-self.timestep), size=(1,)).long() # randomly pick time stamps

        z = self.encoder(x)
        z = z.transpose(1,2)
        nce = 0
        encode_samples = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512)
        forward_seq = z[:,:t_samples+1,:]
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:,t_samples,:].view(batch, 256)
        pred = torch.empty((self.timestep,batch,512)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        z = self.encoder(x)
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden)

        return output

class CDCK3(nn.Module):
    ''' 3 channel decom signal with three channel CDCK2 '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK3, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 256, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.gru1 = nn.GRU(256, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(64, 256) for i in range(timestep)])
        self.gru2 = nn.GRU(256, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(64, 256) for i in range(timestep)])
        self.gru3 = nn.GRU(256, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk3 = nn.ModuleList([nn.Linear(64, 256) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru1 and gru2
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru3._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru3.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden1(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
        else: return torch.zeros(1, batch_size, 64)

    def init_hidden2(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
        else: return torch.zeros(1, batch_size, 64)

    def init_hidden3(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
        else: return torch.zeros(1, batch_size, 64)

    def forward(self, x1, x2, x3, hidden1, hidden2, hidden3):
        batch = x1.size()[0]
        nce = 0
        t_samples = torch.randint(int(self.seq_len/50-self.timestep), size=(1,)).long() # randomly pick time stamps. ONLY DO THIS ONCE FOR BOTH GRU.

        # first gru
        z = self.encoder(x1)
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256)
        forward_seq = z[:,:t_samples+1,:]
        output1, hidden1 = self.gru1(forward_seq, hidden1)
        c_t = output1[:,t_samples,:].view(batch, 64)
        pred = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk1[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        # second gru
        z = self.encoder(x2)
        z = z.transpose(1, 2)
        encode_samples = torch.empty((self.timestep, batch, 256)).float()
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 256)
        forward_seq = z[:, :t_samples + 1, :]
        output2, hidden2 = self.gru2(forward_seq, hidden2)
        c_t = output2[:, t_samples, :].view(batch, 64)
        pred = torch.empty((self.timestep, batch, 256)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk2[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct2 = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        # second gru
        # input sequence is N*C*L
        z3 = self.encoder(x3)
        # encoded sequence is N*C*L
        # reshape to N*L*C for GRU
        z3 = z3.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z3[:,t_samples+i,:].view(batch,256)
        forward_seq = z3[:,:t_samples+1,:]
        output3, hidden3 = self.gru2(forward_seq, hidden3)
        c_t = output3[:,t_samples,:].view(batch, 64)
        pred = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk3[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1.*batch*self.timestep
        nce /= 3. # over three grus
        accuracy = 1.*(correct1.item()+correct3.item()+correct2.item())/(batch*2)

        return accuracy, nce, hidden1, hidden2,hidden3

    def predict(self, x1, x2,x3, hidden1, hidden2,hidden3):

        # first gru
        z1 = self.encoder(x1)
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1)
        # second gru
        z2 = self.encoder(x2)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)
        # three gru
        z3 = self.encoder(x3)
        z3 = z3.transpose(1, 2)
        output3, hidden3 = self.gru3(z3, hidden3)
        return torch.cat((output1, output2, output3), dim=1)

class CDCK_FUSION(nn.Module):
    '''original time domain signal, the frequency domain signal, and features extracted from shallow networks with three channel CDCK2.'''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK_FUSION, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.encoder_domain = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.gru1 = nn.GRU(256, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(64, 256) for i in range(timestep)])
        self.gru2 = nn.GRU(256, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(64, 256) for i in range(timestep)])
        self.gru3 = nn.GRU(256, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk3 = nn.ModuleList([nn.Linear(64, 256) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru1 and gru2
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru3._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru3.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden1(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
        else: return torch.zeros(1, batch_size, 64)

    def init_hidden2(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
        else: return torch.zeros(1, batch_size, 64)

    def init_hidden3(self, batch_size,use_gpu = True): # initialize gru1
        if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
        else: return torch.zeros(1, batch_size, 64)

    def forward(self, x1, x2, x3, hidden1, hidden2, hidden3):
        #
        batch = x1.size()[0]
        nce = 0 # average over timestep and batch and gpus
        t_samples = torch.randint(int(self.seq_len/50-self.timestep), size=(1,)).long() # randomly pick time stamps. ONLY DO THIS ONCE FOR BOTH GRU.

        # first gru
        z = self.encoder(x1)
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256)
        forward_seq = z[:,:t_samples+1,:]
        output1, hidden1 = self.gru1(forward_seq, hidden1)
        c_t = output1[:,t_samples,:].view(batch, 64)
        pred = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk1[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        # second gru
        z = self.encoder(x2)
        z = z.transpose(1, 2)
        encode_samples = torch.empty((self.timestep, batch, 256)).float()
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 256)
        forward_seq = z[:, :t_samples + 1, :]
        output2, hidden2 = self.gru2(forward_seq, hidden2)
        c_t = output2[:, t_samples, :].view(batch, 64)
        pred = torch.empty((self.timestep, batch, 256)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk2[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct2 = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        # second gru
        z3 = self.encoder_domain(x3)
        z3 = z3.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z3[:,t_samples+i,:].view(batch,256)
        forward_seq = z3[:,:t_samples+1,:]
        output3, hidden3 = self.gru2(forward_seq, hidden3)
        c_t = output3[:,t_samples,:].view(batch, 64)
        pred = torch.empty((self.timestep,batch,256)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk3[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        nce /= 3. # over three grus
        accuracy = 1.*(correct1.item()+correct3.item()+correct2.item())/(batch*2)

        return accuracy, nce, hidden1, hidden2,hidden3

    def predict(self, x1, x2, x3, hidden1, hidden2,hidden3):
        # first gru
        z1 = self.encoder(x1)
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1) # output size e.g. 8*128*256
        # second gru
        z2 = self.encoder(x2)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)
        # three gru
        z3 = self.encoder_domain(x3)
        z3 = z3.transpose(1, 2)
        output3, hidden3 = self.gru3(z3, hidden3)
        return torch.cat((output1, output2, output3), dim=1)
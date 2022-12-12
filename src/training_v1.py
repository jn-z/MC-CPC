import torch
import logging
import os
import torch.nn.functional as F
import pdb
import numpy as np
## Get the same logger from main"
logger = logging.getLogger("cdc")

def get_features(sig , window_length=2000, window_step=2000, NFFT=446, max_frames=256):
    feat_mat = []
    for i in range(max_frames):
        start = window_step * i
        end = start + window_length
        slice_sig = sig[start:end]
        feature = STFT(slice_sig, NFFT)
        feat_mat.append(feature)
    feat_mat = np.array(feat_mat, dtype=float)
    return feat_mat
def STFT(frames, NFFT):
    complex_spectrum=np.fft.rfft(frames,NFFT)
    complex_spectrum=np.absolute(complex_spectrum)
    return 1.0/NFFT * np.square(complex_spectrum)
def trainXXreverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, data_r] in enumerate(train_loader):
        data   = data.float().unsqueeze(1).to(device) # add channel dimension
        data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
        optimizer.zero_grad()
        hidden1 = model.init_hidden1(len(data))
        hidden2 = model.init_hidden2(len(data))
        acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def train_spk(args, cdc_model, spk_model, device, train_loader2, optimizer, epoch, batch_size, frame_window):
    cdc_model.eval()  # not training cdc model
    spk_model.train()
    for batch_idx, [data, target, utti] in enumerate(train_loader2):
        b, f_total = data.size()
        tar_total = torch.zeros([b, args.spk_num])
        optimizer.zero_grad()
        if args.CPCType == 'CDCK3':
            data_1 = data
            all_num = data_1.shape[1]
            data1 = get_array(data_1, all_num, 0).unsqueeze(1).to(device)
            data2 = get_array(data_1, all_num, 1).unsqueeze(1).to(device)
            data3 = get_array(data_1, all_num, 2).unsqueeze(1).to(device)
            hidden1 = cdc_model.init_hidden1(len(data1), use_gpu=True)
            hidden2 = cdc_model.init_hidden2(len(data2), use_gpu=True)
            hidden3 = cdc_model.init_hidden3(len(data3), use_gpu=True)
            output = cdc_model.predict(data1, data2, data3, hidden1, hidden2, hidden3)
            data1 = output.contiguous().view((-1, 64))
            src = data1.view(-1, 123, 64)
        if args.CPCType == 'CDCK2':
            data1 = data
            data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension
            hidden = cdc_model.init_hidden(len(data1), use_gpu=True)
            output = cdc_model.predict(data1,hidden)
            data1 = output.contiguous().view((-1, 256))
            src = data1.view(-1, 125, 256)
        if args.CPCType == 'CDCK6':
            data_rev = torch.flip(data, dims=[1])
            data_1 = data.float().unsqueeze(1).to(device)  # add channel dimension
            data_r = data_rev.float().unsqueeze(1).to(device)  # add channel dimension
            hidden1 = cdc_model.init_hidden1(len(data_1))
            hidden2 = cdc_model.init_hidden2(len(data_r))
            output = cdc_model.predict(data_1, data_r, hidden1, hidden2)
            data1 = output.contiguous().view((-1, 256))
            src = data1.view(-1, 125, 256)
        if args.CPCType == 'CDCK_FUSION':
            data1 = data
            data2 = torch.flip(data1, dims=[1])
            data3 = np.concatenate(data1.numpy())
            data3 = get_features(data3, window_length=2000, window_step=2000, NFFT=128 * 8 - 1, max_frames=b)
            data3 = torch.tensor(data3)
            data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension
            data2 = data2.float().unsqueeze(1).to(device)
            data3 = data3.float().unsqueeze(1).to(device)
            hidden1 = cdc_model.init_hidden1(len(data1), use_gpu=True)
            hidden2 = cdc_model.init_hidden2(len(data2), use_gpu=True)
            hidden3 = cdc_model.init_hidden3(len(data3), use_gpu=True)
            output = cdc_model.predict(data1, data2, data3, hidden1, hidden2, hidden3)
            data1 = output.contiguous().view((-1, 314))
            src = data1.view(-1, 64, 314)
        batch_size, fea_frames, fea_dim = src.size()
        state_lab = target[:,0].view(-1,1)
        state_lab = [int(ss) for ss in state_lab]
        state_lab = torch.Tensor(state_lab).long().view(-1,1).to(device)
        anchor_sv, predict, ce_loss,tar = spk_model(src,state_lab, fea_frames,0)
        loss = (ce_loss.sum() / (batch_size))
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        tar_total += tar.cpu()
        tar_total = tar_total
        predict = tar_total.max(dim=1)[1]
        acc = 1.*predict.eq(state_lab.view_as(predict).cpu()).sum().item()/batch_size
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * b / frame_window, len(train_loader2.dataset),
                100. * batch_idx / len(train_loader2), lr, acc, loss.item()))

def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, target, utti] in enumerate(train_loader):
        b, f_total = data.size()
        optimizer.zero_grad()
        if args.CPCType == 'CDCK3':
           data_1 = data
           all_num = data_1.shape[1]
           data1 = get_array(data_1, all_num, 0).unsqueeze(1).to(device)
           data2 = get_array(data_1, all_num, 1).unsqueeze(1).to(device)
           data3 = get_array(data_1, all_num, 2).unsqueeze(1).to(device)
       
           hidden1 = model.init_hidden1(len(data1), use_gpu=True)
           hidden2 = model.init_hidden2(len(data2), use_gpu=True)
           hidden3 = model.init_hidden3(len(data3), use_gpu=True)
           acc, loss, hidden1, hidden2, hidden3 = model(data1, data2, data3, hidden1, hidden2, hidden3)
        if args.CPCType == 'CDCK2':
           data1 = data
           data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension
           hidden = model.init_hidden(len(data1), use_gpu=True)
           acc, loss, hidden = model(data1, hidden)
        if args.CPCType == 'CDCK6':
           data_rev = torch.flip(data, dims=[1])
           data_1 = data.float().unsqueeze(1).to(device) # add channel dimension
           data_r = data_rev.float().unsqueeze(1).to(device) # add channel dimension
           hidden1 = model.init_hidden1(len(data_1))
           hidden2 = model.init_hidden2(len(data_r))
           acc, loss, hidden1, hidden2 = model(data_1, data_r, hidden1, hidden2)
        if args.CPCType == 'CDCK_FUSION':
            data1 = data
            data2 = torch.flip(data1, dims=[1])
            data3 = np.concatenate(data1.numpy())

            data3 = get_features(data3, window_length=2000, window_step=2000, NFFT=128 * 8 - 1, max_frames=b)

            data3 = torch.tensor(data3)
            data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension
            data2 = data2.float().unsqueeze(1).to(device)
            data3 = data3.float().unsqueeze(1).to(device)
            hidden1 = model.init_hidden1(len(data1), use_gpu=True)
            hidden2 = model.init_hidden2(len(data2), use_gpu=True)
            hidden3 = model.init_hidden3(len(data3), use_gpu=True)
            acc, loss, hidden1, hidden2, hidden3 = model(data1, data2, data3, hidden1, hidden2, hidden3)
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx, lr, acc, loss.item()))
        batch_idx += 1
def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
def get_array(ori_array,all_num,fist_index):
    #pdb.set_trace()
    size_h = ori_array.shape[0]
    size_w = ori_array.shape[1]

    new_array = torch.zeros(size_h, int(size_w / 3))
    index = range(fist_index, int(size_w / 3)*3, 3)
    for index_j in range(ori_array.shape[0]):
        new_array[index_j,:] = ori_array[index_j,index].unsqueeze(0)
    return new_array
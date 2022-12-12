import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb
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

def validation_spk(args, cdc_model, spk_model, device, data_loader2, batch_size, frame_window):
    logger.info("Starting Validation")
    cdc_model.eval()  # not training cdc model
    spk_model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for [data, target, utti] in data_loader2:
            b, f_total = data.size()
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
                output = cdc_model.predict(data1, hidden)
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
            tar_total = tar.cpu()
            loss = ce_loss.sum()
            tar_total = tar_total
            total_loss += (loss / f_total)
            predict = tar_total.max(dim=1)[1]
            total_acc += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
    total_loss /= len(data_loader2.dataset)*frame_window # average loss
    total_acc  /= 1.*len(data_loader2.dataset)*frame_window

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tdev_num: {:.4f}\n'.format(
                total_loss, total_acc,1.0*len(data_loader2.dataset)))

    return total_acc, total_loss

def validation(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch_idx, [data, target, utti] in enumerate(data_loader):
            b, f_total = data.size()
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
                data1 = data.float().unsqueeze(1).to(device)  # add channel dimension
                data_r = data_rev.float().unsqueeze(1).to(device)  # add channel dimension
                hidden1 = model.init_hidden1(len(data1))
                hidden2 = model.init_hidden2(len(data_r))
                acc, loss, hidden1, hidden2 = model(data1, data_r, hidden1, hidden2)
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
            tmp_loss = loss
            tmp_acc = acc
            total_loss += len(data1) * tmp_loss
            total_acc += len(data1) * tmp_acc
    total_loss /= len(data_loader.dataset)  # average loss
    total_acc /= len(data_loader.dataset)  # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))

    return total_acc, total_loss

def get_array(ori_array,all_num,fist_index):
    size_h = ori_array.shape[0]
    size_w = ori_array.shape[1]

    new_array = torch.zeros(size_h, int(size_w / 3))
    index = range(fist_index, int(size_w / 3)*3, 3)
    for index_j in range(ori_array.shape[0]):
        new_array[index_j,:] = ori_array[index_j,index].unsqueeze(0)
    return new_array

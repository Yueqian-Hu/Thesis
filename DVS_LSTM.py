# -*- coding: utf-8 -*-
# Install dcll from https://github.com/nmi-lab/dcll

from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
from tqdm import tqdm

import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


n_iters = 60
n_iters_test = 60 
n_test_interval = 20
batch_size = 72
dt = 10000 
ds = 4
target_size = 11 
n_epochs = 3000 
in_channels = 2 
thresh = 0.3
lens = 0.25
decay = 0.3
learning_rate = 1e-4
time_window = 60
im_dims = im_width, im_height = (128//ds, 128//ds)
names = 'dvsGesture_lstm_fcorig_count'

parser = argparse.ArgumentParser(description='STDP for DVS gestures')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)

def generate_test(gen_test, n_test:int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test,int(np.ceil(input_test.shape[0]/batch_size)))
    for i in range(n_test):
        input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
    return n_test, input_tests, labels1h_tests

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predicted, target):
        mse_loss = torch.mean((predicted - target)**2)
        mae_loss = torch.mean(torch.abs(predicted - target))
        hybrid_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
        return hybrid_loss

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predicted, target):
        absolute_error = torch.abs(predicted - target)
        huber_loss = torch.where(absolute_error < self.delta, 0.5 * absolute_error**2, self.delta * (absolute_error - 0.5 * self.delta))
        mean_huber_loss = torch.mean(huber_loss)
        return mean_huber_loss
    
class CustomLoss(nn.Module):
    def __init__(self, mse_weight=0.5, huber_delta=2.0):
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.huber_delta = huber_delta

    def forward(self, predicted, target):
        mse_loss = torch.mean((predicted - target)**2)
        absolute_error = torch.abs(predicted - target)
        huber_loss = torch.where(absolute_error < self.huber_delta, 0.5 * absolute_error**2, self.huber_delta * (absolute_error - 0.5 * self.huber_delta))
        huber_loss = torch.mean(huber_loss)
        custom_loss = self.mse_weight * mse_loss + (1 - self.mse_weight) * huber_loss
        return custom_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        loss = (1 - pt) ** self.gamma * F.cross_entropy(inputs, targets, reduction='none')
        if self.reduction == 'mean':
            return (self.alpha * loss).mean()
        elif self.reduction == 'sum':
            return (self.alpha * loss).sum()
        else:
            return self.alpha * loss


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)


cfg_fc = [512, 512, 11]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


best_acc = 0
acc = 0
acc_record = list([])


class LSTM_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=32*32*2,
                          hidden_size=cfg_fc[0],
                          num_layers=2)
        
        self.readout = nn.Linear(cfg_fc[0],cfg_fc[-1])

    def zeros_hidden_state(self):
        h_state = []
        for i in range(2):
            h_state.append(torch.zeros(cfg_fc[0],cfg_fc[0],device=device))

        return h_state

    def forward(self, input, h_state, win = 60):

        outs = []
        x = input[:, :, :, :, :win].view(batch_size, -1, win).permute([0, 2, 1])

        r_out, (h_n, h_c) = self.lstm(x)
        out = self.readout(r_out[:,-1,:])
        return out,(h_n, h_c)

    def compute_loss(self, outputs, labels):
        loss = HuberLoss(delta=2.0)(outputs, labels)
        #loss = CustomLoss(mse_weight=0.5, huber_delta=2.0)(outputs, labels)
        return loss


rnn_model = LSTM_Model()
rnn_model.to(device)
#criterion = nn.MSELoss()
#criterion = HuberLoss(delta=2.0)
#optimizer = optim.SGD(rnn_model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=1e-4)
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)
#optimizer = torch.optim.NAdam(rnn_model.parameters(), lr=learning_rate)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(rnn_model))
act_fun = ActFun.apply
print('Generating test...')
n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)
print('n_test %d' % (n_test))

for epoch in range(n_epochs):
    rnn_model.zero_grad()
    optimizer.zero_grad()
    
    running_loss = 0
    start_time = time.time()

    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    input = input.float().to(device)
    input = input.permute([1,2,3,4,0])
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]
    outputs, h_state = rnn_model(input, time_window)

    #loss = criterion(outputs.cpu(), labels)
    #loss = rnn_model.compute_loss(outputs.cpu(), labels)
    #loss = F.cross_entropy(outputs.cpu(), labels)
    loss = FocalLoss()(outputs.cpu(), labels)
    #running_loss += loss
    running_loss = running_loss + loss.item()
    #loss.backward(retain_graph=True)
    loss.backward()
    for name, parms in rnn_model.named_parameters():
        print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad.shape)
    #torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 0.5)
    optimizer.step()
    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, running_loss))

    if (epoch + 1) % n_test_interval == 0:
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 4000)
    
        for i in range(len(input_tests)):
            inputTest = input_tests[1].float().to(device)
            inputTest = inputTest.permute([1,2,3,4,0])
            outputs, h_state = rnn_model(inputTest, time_window)

            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)
            correct = correct + (predicted.cpu() == labelTest).sum()

        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct.float() / total))
        
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

        print(acc)
        print('Saving..')
        state = {
            'net': rnn_model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc
print(max(acc_record))

    



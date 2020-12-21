import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from data_loader import *
from network import *
from config import*

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    target_exp=target.view(1, -1).expand_as(pred)

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_num=correct.view(-1).float().sum(0)
    acc=correct_num.mul_(100.0 / batch_size)
    return acc

class init_all():
    def __init__(self):
        self._init_network()
        self._init_dataloader()
        self._init_optimizer()
        self._init_Criterion()

    def _init_network(self):
        self.network=create_model()

    def _init_dataloader(self):
        self.dloader=DataLoader(batch_size=config["BatchSize"])

    def _init_optimizer(self):
        optim_opts = config['Optimizer']
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        parameters = self.network.parameters()
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                                              betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            self.optimizer = torch.optim.SGD(parameters,  # 优化的参数
                                             lr=learning_rate,
                                             momentum=optim_opts['momentum'],
                                             nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                                             weight_decay=optim_opts['weight_decay'])

    def _init_Criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def adjust_learning_rates(self, epoch):
        optim_opts = config['Optimizer']
        LUT = optim_opts['LUT_lr']
        # lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
        L_R=0
        for (max_epoch, lr) in LUT:
            if max_epoch>epoch:
                L_R=lr
                break
        print('==> Set to optimizer lr = %.10f' % (L_R))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = L_R

class train_net(init_all):
    def __init__(self):
        init_all.__init__(self)
        self.epoch_num = config['epoch']
        self.model_path = config['model_path']

    def save_net(self,epoch):
        state = {'net': self.network.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state,self.model_path)
    def load_net(self):
        print('loading model:',self.model_path)
        model_state=torch.load(self.model_path)
        self.network.load_state_dict(model_state['net'])
        self.optimizer.load_state_dict(model_state['optimizer'])

    def train_step(self,batch):
        dataX=batch[0]
        labels=batch[1]
        dataX_var =dataX
        labels_var =labels
        self.optimizer.zero_grad()
        pred_var = self.network(dataX_var)
        record = {}
        loss_total = self.criterion(pred_var, labels_var)
        record['prec1'] = accuracy(pred_var.data, labels)

        record['loss'] = loss_total
        loss_total.backward()
        self.optimizer.step()

        return record

    def train_process(self):
        if os.path.exists(self.model_path):
            self.load_net()
        else:
            pass

        for epoch in range(self.epoch_num):
            if epoch%20==0:
                # self.adjust_learning_rates(epoch)
                self.save_net(epoch)
            for batch_index in range(self.dloader.batch_num):
                batch=self.dloader.load_batch()
                record = self.train_step(batch)

                # info='| epoch:{} |**| accuracy:{:.4f} |**| loss:{:.4f} |**| load_time:{:.4f} |**| process_time:{:.4f} |**| total_time:{}时{}分{:.2f}秒'.format(epoch,record['prec1'].item(),record['loss'].item(),record['load_time'],record['process_time'],h,m,s)
                info = '| epoch:{} |**| batch:{}/{} |**| accuracy:{:.4f} |**| loss:{:.4f}'.format(epoch,batch_index,self.dloader.batch_num, record['prec1'].item(), record['loss'].item())
                print(info)

            self.test_process()

        self.save_net(epoch)

    def test_step(self,test_data):
        batch_x=test_data[0]
        batch_y=test_data[1]
        pred = self.network(batch_x)
        acc = accuracy(pred.data, batch_y)
        return acc

    def test_process(self):
        acc=0
        for test_batch_index in range(self.dloader.test_batch_num):
            acc+=self.test_step(self.dloader.load_test_batch())

        print("The accuracy on test set is {} !".format(acc/self.dloader.test_batch_num))

if __name__ == '__main__':
    train_net().train_process()
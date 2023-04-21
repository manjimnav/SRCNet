import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .callback import EarlyStopping
from .dataset import TimeDataset
from .metrics import metric

class Experimenter:
    def __init__(self, args, model):
        super(Experimenter, self)
        self.args = args
        self.device = self._acquire_device()
        self.model = model.to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, split, scaler=None):
        args = self.args

        if split == 'test':
            shuffle_flag = False
        else:
            shuffle_flag = True

        data_set = TimeDataset(
            data_path=args.data_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            split=split,
            step=args.step,
            target=args.target,
            scaler=scaler,
            reduce_data=args.reduce_data
        )
        print(split, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            drop_last=False)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        
        criterion = nn.MSELoss()

        return criterion

    def validate(self, vali_loader, criterion, train_data, split='valid'):
        self.model.eval()
        total_loss = []
        #mean, std = train_data.target_scale_factors()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)

            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            outputs = self.model(batch_x)

            loss = self.model.compute_loss(batch_y.detach(), outputs.detach(), criterion)
            if type(loss) != torch.Tensor:
                loss = loss[0]#+loss[1]
                
            total_loss.append(loss.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(split='train')
        _, vali_loader = self._get_data(
            split='valid', scaler=train_data.scaler)
        _, test_loader = self._get_data(split='test', scaler=train_data.scaler)

        path = './checkpoints/' + setting
        if not os.path.exists(path):
            print(path)
            os.makedirs(path+"/0")


        time_now = time.time()
        training_time = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_epochs = []
        valid_loss_epochs = []
        test_loss_epochs = []
        print('Starting training...')
        for epoch in range(self.args.train_epochs):
            self.current_epoch = epoch
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)

                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                outputs = self.model(batch_x)

                loss = self.model.compute_loss(batch_y, outputs, criterion)
                

                if (i + 1) % 1000 == 0:
                    train_loss_val = loss
                    if type(loss) != torch.Tensor:
                        train_loss_val = loss[0]+loss[1]
                    
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, train_loss_val))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * \
                        ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.censor_grads:
                    self.model = censor_grads(self.model, loss, self.args.lamb, self.args.censor_type)
                    train_loss.append(loss[0].item())
                else:
                    if type(loss) != torch.Tensor:
                        loss = loss[0]+loss[1]

                    train_loss.append(loss.item())

                    loss.backward()

                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.validate(vali_loader, criterion, train_data)
            test_loss = self.validate(test_loader, criterion, train_data, 'test')

            train_loss_epochs.append(train_loss)
            valid_loss_epochs.append(vali_loss)
            test_loss_epochs.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                train_loss_epochs = np.array(train_loss_epochs)
                valid_loss_epochs = np.array(valid_loss_epochs)
                test_loss_epochs = np.array(test_loss_epochs)

                """np.save(path + '/train_history.npy', train_loss_epochs)
                np.save(path + '/valid_history.npy', valid_loss_epochs)
                np.save(path + '/test_history.npy', test_loss_epochs)"""

                break
        
        training_time = (time.time()-training_time)/60
        best_model_path = path + '/' + 'checkpoint.pth'.format(epoch)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, training_time

    def test(self, setting, split='test'):
        train_data, _ = self._get_data(split='train')
        _, test_loader = self._get_data(
            split=split, scaler=train_data.scaler)

        self.model.eval()

        preds = []
        trues = []
        times = []

        mean, std = train_data.target_scale_factors()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device).detach().cpu().numpy()

            outputs = self.model(batch_x)

           
            pred = outputs.detach().cpu().numpy()
            #pred = train_data.scaler.transform(pred, groups=batch_y_mark[:, -1, -1] if len(batch_y_mark.shape)>2 else batch_y_mark[:, -1],target_index= train_data.target_index) if self.args.scale_results else pred
            pred = pred#*std + mean

            if pred.shape[-1] < 2:
                pred = pred.squeeze(axis=2)
                true = batch_y.detach().cpu().numpy().squeeze(axis=2)
            else:
                pred = pred
                true = batch_y.detach().cpu().numpy()
                
            preds.append(pred)
            #true = train_data.scaler.transform(true, groups=batch_y_mark[:, -1, -1] if len(batch_y_mark.shape)>2 else batch_y_mark[:, -1],target_index= train_data.target_index) if self.args.scale_results else true
            true = true#*std + mean
            trues.append(true)

            times.append(batch_y_mark)
        
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        times = np.concatenate(times)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print(preds)
        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)
        print('mse:{}, mae:{}, wape:{}'.format(mse, mae, wape))

        np.save(folder_path + 'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe, wape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'time.npy', times)
        
        return mae, mse, rmse, mape, mspe, wape

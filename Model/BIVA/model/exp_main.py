
from model import BIVA

import matplotlib.pyplot as plt
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_model, load_model
from data_provider.data_factory import data_provider
from exp_basic import Exp_Basic
from torch.utils.tensorboard import SummaryWriter
# import pytorch_model_summary as pms
# from torchinfo import summary
from torch import optim
import torch.nn as nn
import torch

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # print(pms.summary(self.model, torch.zeros(self.args.batch_size, self.args.seq_len, self.args.channels),
        #                     max_depth=None, show_parent_layers=True, show_input=True))    

    def _build_model(self):
        model_dict = {
            'BIVA': BIVA,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
                
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        
        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'BIVA' in self.args.model:
                            states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                        else:
                            pass

                else:
                    if 'BIVA' in self.args.model:
                       states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                    else:
                        pass

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:,f_dim:].to(self.device)
                
                #input = batch_x.detach().cpu()
                imputed_loss = imputed_loss.detach().cpu()
                VAE_loss = VAE_loss.detach().cpu()
                # recon = recon_output.detach().cpu()
                pred = states.detach().cpu()
                true = batch_y.detach().cpu()
                
                # calc loss
                # loss_recon = criterion(recon,imputed)
                loss_states = criterion(pred,true)
                loss = VAE_loss + imputed_loss + loss_states
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.last_chkpt:
            self.model.load_state_dict(torch.load(os.path.join(path + self.args.last_chkpt)))
            print(f"=========> load_last chkpt: {self.args.last_chkpt}")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            state_loss = []

            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'BIVA' in self.args.model:
                            states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                        else:
                            pass

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # loss_recon = criterion(recon_output,imputed)
                        loss_states = criterion(states,batch_y)
                        loss = VAE_loss + imputed_loss + loss_states
                        # loss = (loss_recon*self.args.recon_loss_w) + (imputed_loss*self.args.imputed_loss_w) + (loss_states*self.args.state_loss_w)

                        train_loss.append(loss.item())

                else:
                    if 'BIVA' in self.args.model:
                        states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)

                    else:
                        pass

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # loss_recon = criterion(recon_output,imputed)
                    loss_states = criterion(states,batch_y)
                    loss = VAE_loss + imputed_loss + loss_states
                    
                    # state_loss.append(loss_states.item())
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * \
                        ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            # train_loss_s = train_loss.copy()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            torch.save(self.model.state_dict(), path + self.args.model_id +'_last_checkpoint.pth')
            
            # loss save
            save_path = '/content/drive/MyDrive/ZZ/Code_02/exp/(BIVA)_result/' + self.args.model_id + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            train_loss = np.array(train_loss)
            vali_loss = np.array(vali_loss)
            test_loss = np.array(test_loss)

            np.save(save_path + 'train_loss.npy',train_loss)
            np.save(save_path + 'vali_loss.npy', vali_loss)
            np.save(save_path + 'test_loss.npy', test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, self.args.model_id)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + self.args.model_id + '_best_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(
                path + self.args.model_id + '_best_checkpoint.pth')))

        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join(
        #         path + self.args.model_id, 'last_checkpoint.pth')))


        preds = []
        trues = []
        recon_xs = []
        imputed_xs = []
        imputation = []
        inputx = []
        # './test_results/' + setting + '/'
        folder_path = '/content/drive/MyDrive/ZZ/Code_02/exp/(BIVA)_plot/' + self.args.model_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'BIVA' in self.args.model:
                            states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                        else:
                            pass
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                else:
                    if 'BIVA' in self.args.model:
                        states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                    else:
                        pass

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = states.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                recon = recon_output.detach().cpu().numpy()
                imputed = imputed_x.detach().cpu().numpy()
                imputed_s = seasonal_init.detach().cpu().numpy()
                recon_output = recon_output.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                imputed_xs.append(imputed_s)
                recon_xs.append(recon)
                imputation.append(imputed)
                inputx.append(batch_x.detach().cpu().numpy())

                # if i % 10 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate(
                #         (input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate(
                #         (input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        imputed_xs = np.array(imputed_xs)
        recon_xs = np.array(recon_xs)
        imputation = np.array(imputation)
        inputx = np.array(inputx)
        
        # x_mark = np.array(batch_x_mark.cpu().numpy())
        # y_mark = np.array(batch_y_mark.cpu().numpy())

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        imputed_xs = imputed_xs.reshape(-1, imputed_xs.shape[-2], imputed_xs.shape[-1])
        recon_xs = recon_xs.reshape(-1, recon_xs.shape[-2], recon_xs.shape[-1])
        imputation = imputation.reshape(-1, imputation.shape[-2], imputation.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        # x_mark = x_mark.reshape(-1, x_mark.shape[-2], x_mark.shape[-1])
        # y_mark = y_mark.reshape(-1, y_mark.shape[-2], y_mark.shape[-1])

        # result save
        save_path = '/content/drive/MyDrive/ZZ/Code_02/exp/(BIVA)_result/' + self.args.model_id + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(save_path + 'BIVA_metrics.npy',
                np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        np.save(save_path + 'BIVA_pred.npy', preds)
        np.save(save_path + 'BIVA_trues.npy', trues)
        np.save(save_path + 'BIVA_recons.npy',recon_xs)
        np.save(save_path + 'BIVA_imputation.npy',imputation)
        np.save(save_path + 'BIVA_inputs.npy', inputx)
        np.save(save_path + 'BIVA_imputed_xs.npy', imputed_xs)
        # np.save(save_path + 'x_mark.npy', x_mark)
        # np.save(save_path + 'y_mark.npy', y_mark)
        return preds, trues, inputx, mae, mse, rmse, mape, mspe, rse, corr

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints)
            best_model_path = path + self.args.model_id, '_checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        imputation = []

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'BIVA' in self.args.model:
                            states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                        else:
                            pass
                else:
                    if 'BIVA' in self.args.model:
                        states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x = self.model(batch_x)
                    else:
                        pass

                pred = states.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                imputation.append(imputed_x)

        preds = np.array(preds)
        imputation = np.array(imputation)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        imputation = imputation.reshape(-1, imputation.shape[-2], imputation.shape[-1])

        # result save
        folder_path = '/content/drive/MyDrive/ZZ/Code_02/result/(BIVA)_predict/' + self.args.model_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

# -*- encoding: utf-8 -*-
'''
@File    :   diffusion.py
@Time    :   2022/04/09 16:17:51
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''


# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
from processor.process import Process
from processor.base_method import import_class
import time
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class main(Process):

    def load_model(self):
        # 加载模型
        Model = import_class(self.args.model)
        self.log_print(f"import model form: {self.args.model}")
        if self.args.model_config_method == 1:
            self.model = Model(self.args.model_args).cuda(self.device)
        else:
            self.model = Model(**self.args.model_args).cuda(self.device)
        self.load_weights() # 在这里加载权重 避免多卡单卡保存的权重名称不一问题

        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                if dist.is_initialized():
                    self.model = DDP(self.model, device_ids=[self.device],
                                    output_device=self.device)
                else:
                    raise RuntimeError('Distributed not initialized')
        self.log_print("The Model is {} had loaded".format(self.model))

        # 加载Loss函数
        if len(self.args.device) > 1:
            self.loss = self.model.module.loss
        else:
            self.loss = self.model.loss

    def train(self):
        self.model.train()
        epoch = self.meta_info['epoch']

        loader = self.data_loader['train']

        loss_value = []

        self.record_time()  # 记录时间
        if self.rank == 0:
            process = tqdm(loader, ncols=90)
        else:
            process = loader

        for batch_idx, (data, cls) in enumerate(process):
            # load data
            data, cls = data.cuda(self.device), cls.cuda(self.device)

            self.optim.zero_grad()
            # forward
            start = time.time()

            # NotImplemented hook
            # if self.args.hook:
            #     encoded, decoded = self.hook(data)
            # else:
            #     encoded, decoded = self.model(data)

            loss = self.loss(data)


            forward_time = time.time() - start

            # backward
            loss.backward()
            self.optim.step()

            # record
            self.iter_info['loss'] = loss.data.item()
            if self.rank == 0:
                self.writer.add_scalar('train_iter_loss', self.iter_info['loss'], self.meta_info['iter'])
                self.writer.flush()

            self.iter_info['lr'] = self.optim.param_groups[0]['lr']
            
            if self.rank == 0:
                loss_value.append(self.iter_info['loss'])
                self.show_iter_info()

            self.meta_info['iter'] += 1

            if batch_idx % self.args.log_interval == 0 and self.rank == 0:
                process.set_description(
                    '\tLoss: {:.4f}  lr:{:.6f}'.format(self.iter_info['loss'], self.iter_info['lr']))

            # if self.meta_info['iter'] % self.args.log_interval == 0:
            #     # print(data[:8].shape, decoded[:8].shape)
            #     grid = torch.cat([data[:8], decoded[:8]],dim=0)
            #     grid = torchvision.utils.make_grid(grid, nrow=8)
            #     self.writer.add_image('recon_iter', grid, self.meta_info['iter'])
            #     self.writer.flush()


        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['lr'] = self.optim.param_groups[0]['lr']
        if self.rank == 0:
            self.writer.add_scalar('train_epoch_loss', self.epoch_info['mean_loss'], self.meta_info['epoch'])
            # self.writer.flush()
            self.show_epoch_info()
            self.writer.add_scalar('lr', self.epoch_info['lr'], self.meta_info['epoch'])
            self.writer.flush()

        epoch_time = self.split_time()
        self.log_print("The train epoch {} time:{}, loss: {}".format(
            epoch, epoch_time, self.epoch_info['mean_loss']))

        if self.args.scheduler != "None":
            self.scheduler.step()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        loader = self.data_loader['test']
        self.record_time()  # 记录时间

        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        # TODO: n_sample
        n_sample = self.args.n_samples
        x = torch.randn([n_sample, self.args.model_args['eps_args']['image_channels'], 
                        self.args.train_feeder_args['image_size'], self.args.train_feeder_args['image_size']],
                        device=self.device)

        # Remove noise for $T$ steps
        if self.rank == 0:
            process = tqdm(range(self.args.model_args['n_steps']), ncols=90)
        else:
            process = range(self.args.model_args['n_steps'])
        for t_ in process:
            # $t$
            t = self.args.model_args['n_steps'] - t_ - 1
            # Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$
            x = self.model.module.p_sample(x, x.new_full((n_sample,), t, dtype=torch.long))

        if self.rank == 0:
            x = torch.clamp(x, 0, 1)
            grid = torchvision.utils.make_grid(x, nrow=8)
            self.writer.add_image('recon_val', grid, self.meta_info['epoch'])
            self.writer.flush()


        epoch_time = self.split_time()
        self.logger.eval_log("The test epoch {} time:{}".format(
            self.meta_info['epoch'], epoch_time))
        print("The test epoch {} time:{}".format(
            self.meta_info['epoch'], epoch_time))

        return None, -1
        
        





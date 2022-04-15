from processor.IO import IO
import torch
from processor.base_method import import_class, str2bool
import torch.optim  as optim 
import yaml
import numpy as np
import time
import os
import pickle
import shutil
from torch.utils.tensorboard import SummaryWriter


class Process(IO):
    def __init__(self, args=None, rank=0):
        self.load_args(args)
        self.init_log()
        self.init_env(rank)
        self.init_tb()
        self.load_model()
        # self.load_weights()
        self.save_args()
        self.cp_model()
        self.load_data()
        self.load_optim()

        if self.args.hook:
            self.register_hook()

    def init_env(self, rank):
        super().init_env(rank)
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)   

    def init_tb(self):
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=self.log_path)

    def save_args(self):
        arg_dict = vars(self.args)
        with open('{}/config.yaml'.format(self.log_path), 'w') as f:
            yaml.dump(arg_dict, f)

    def cp_model(self):
        if os.path.exists(os.path.join(self.log_path, 'model')):
            shutil.rmtree(os.path.join(self.log_path, 'model'))
        shutil.copytree('model', os.path.join(self.log_path, 'model'), ignore=shutil.ignore_patterns('pyc'))

    def load_data(self):
        Feeder = import_class(self.args.feeder)
        # if "debug" not in self.args.test_feeder_args:
        #     self.args.test_feeder_args["debug"] = self.args.debug
        self.data_loader = {}
        
        if self.args.phase == "train":
            train_set = Feeder(**self.args.train_feeder_args)
            train_sample = torch.utils.data.distributed.DistributedSampler(train_set,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True)
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=train_set,
                sampler=train_sample,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=True
            )
        test_set = Feeder(**self.args.test_feeder_args)
        test_sample = torch.utils.data.distributed.DistributedSampler(test_set,
            num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False)
        self.data_loader["test"] = torch.utils.data.DataLoader(
                dataset=test_set,
                sampler=test_sample,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=True
            )
        self.log_print(f"Successfully load dataset: {self.args.feeder}")

    def load_optim(self):
        if self.args.optim == "Adam":
            self.optim = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        elif self.args.optim == "SGD":
            self.optim = optim.SGD(self.model.parameters(),
                                   lr=self.args.lr,
                                   momentum=0.9,
                                   weight_decay=self.args.weight_decay,
                                   nesterov=self.args.nesterov)
        else:
            raise ValueError("No match optim to use")
        
        if self.args.scheduler == "StepLR" or self.args.scheduler == "step": # StepLR, need step_size and gamma parameters
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=self.args.step_size, gamma=self.args.gamma)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, **self.args.scheduler_args)
        elif self.args.scheduler == "MultiStepLR" or self.args.scheduler == 'multiStep': # MultiStepLR, need milestones and gamma parameters
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.args.milestones, gamma=self.args.gamma)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, **self.args.scheduler_args)
        elif self.args.scheduler == "CosineAnnealingLR" or self.args.scheduler == 'cosine': # CosineAnnealingLR, need T_max and eta_min parameters
                                                        # T_max: the maximum number of epochs to reset the learning rate
                                                        # eta_min: the minimum learning rate
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.args.T_max, eta_min=self.args.eta_min)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, **self.args.scheduler_args)
        elif self.args.scheduler == "None":
            self.scheduler = lambda x: None
        else:
            raise ValueError("NotImplementedError, No match scheduler to use, please check your parameters")

    def adjust_learning_rate(self, epoch): # 功能废弃，由scheduler自动调整
        if self.args.optim == 'SGD' or self.args.optim == 'Adam':
            if epoch < self.args.warm_up_epoch:
                lr = self.args.lr * (epoch + 1) / self.args.warm_up_epoch
            else:
                lr = self.args.lr * (
                        0.1 ** np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.logger.train_log('\t{}: {}'.format(k, v))

    def show_iter_info(self):
        if self.meta_info['iter'] % self.args.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.logger.train_log(info)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def only_train_part(self, key_words, epoch):
        if epoch >= self.args.only_train_epoch:
            self.logger.log('only train part, require grad')
            for key, value in self.model.named_parameters():
                if key_words in key:
                    value.requires_grad = True
                    # self.logger.log(key + '-require grad')
        else:
            self.logger.log('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if key_words in key:
                    value.requires_grad = False
                    self.logger.log(key + '-not require grad')

    def save_results(self, results, filename): # 可用于保存logits, preds, labels等
        results_path = os.path.join(self.log_path, "results")
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        with open('{}/{}'.format(results_path, filename), 'wb') as f:
            pickle.dump(results, f)


    def train(self):    # The train template
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

        return self.epoch_info['mean loss'] # 重构时这里需要返回result和acc

    def register_hook(self): # hook template
        Hook = import_class(self.args.hook_method)
        self.hook = Hook(self.model) # hook 函数需要符合设置，其输入需为要检测的model

    def start(self):  #此处执行的程序
        self.logger.log('Parameters:\n{}\n'.format(str(vars(self.args))))
        self.best_score = 0

        # training phase
        if self.args.phase == 'train':
            for epoch in range(self.args.epochs):# TODO: check point
                self.meta_info['epoch'] = epoch
                # training
                self.logger.train_log('Training epoch: {}'.format(epoch))
                self.train()
                self.logger.train_log('Done.\n')
                # save model
                if ((epoch + 1) % self.args.save_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    self.save_model(epoch)
                # evaluation
                self.logger.eval_log('Eval epoch: {}'.format(epoch))
                result, acc = self.test()
                self.logger.eval_log('Done.\n')
                # save best_model
                if acc > self.best_score:
                    self.best_score = acc
                    self.save_best_model()

        # test phase
        elif self.args.phase == 'test':
            # the path of weights must be appointed
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.logger.log('Model:   {}.'.format(self.args.model))
            self.logger.log('Weights: {}.'.format(self.args.weights))
            # evaluation
            self.logger.eval_log('Evaluation Start:')
            result, acc = self.test()
            self.logger.eval_log('Done.\n')
            # save the output of model
            result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name, result))
            self.save_results(result_dict, 'result_{}_{}.pkl'.format(epoch, acc))
            # save best results
            # if acc > self.best_score:
            #     self.save_results(result_dict, 'result_best_{}.pkl'.format(acc))
        


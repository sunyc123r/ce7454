import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm




def warm_up_and_cosine_anneal(current_step,total_epoch,warm_up_step,max,min):
    if current_step<warm_up_step:
        return current_step/warm_up_step
    else:
        return (min+0.5*(max-min)*(1+np.cos((current_step-warm_up_step)/(total_epoch-warm_up_step)*np.pi)))/max


class BaseTrainer:
    def __init__(self,
                 net: nn.Module,
                 train_loader: DataLoader,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100) -> None:
        self.net = net
        self.train_loader = train_loader

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warm_up_and_cosine_anneal(
                step,
                epochs * len(train_loader),
                0.1*epochs * len(train_loader),
                learning_rate, 
                1e-5),)

    def train_epoch(self,args):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            # for train_step in tqdm(range(1, 5)):
            batch = next(train_dataiter)
            data = batch['data'].cuda(args.device)
            #print(data.size())
            target = batch['soft_label'].cuda(args.device)
            # forward
            logits = self.net(data)
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      target,
                                                      reduction='sum')
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics

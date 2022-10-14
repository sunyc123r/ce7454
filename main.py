import argparse
import os
import time

import torch
from dataset import PSGClsDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision.models import resnet18,resnet50
from trainer import BaseTrainer
from model import Clip_Rel


def train(args):
    savename = f'e{args.epoch}_lr{args.lr}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}_md{args.mode}'
    os.makedirs('./checkpoints/{}'.format(args.model_name), exist_ok=True)
    os.makedirs('./results/{}'.format(args.model_name), exist_ok=True)

    # loading dataset
    train_dataset = PSGClsDataset(stage='train')
    train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)

    val_dataset = PSGClsDataset(stage='val')
    val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)

    test_dataset = PSGClsDataset(stage='test')
    test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=8)
    print('Data Loaded...', flush=True)
    device=torch.device("cuda:{}".format(args.device))

    if args.model_name=='resnet18':
        model=resnet18(pretrained=True)
        model.fc= torch.nn.Linear(512, 56)

    elif args.model_name=='resnet50':
        model=resnet50(pretrained=True)
        model.fc= torch.nn.Linear(2048, 56)
    
    else:
        print('load CLIP')
        model=Clip_Rel(device,args.dropout)

    model.to(device)
    print('Model Loaded...', flush=True)

    # loading trainer
    trainer = BaseTrainer(model,
                      train_dataloader,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch)
    evaluator = Evaluator(model, k=3)

    # loading model
    #model = resnet18(pretrained=True)
    #model.fc = torch.nn.Linear(512, 56)

    # train!
    print('Start Training...', flush=True)
    begin_epoch = time.time()
    best_val_recall = 0.0
    for epoch in range(0, args.epoch):
        train_metrics = trainer.train_epoch(args)
        val_metrics = evaluator.eval_recall(val_dataloader,args)

    # show log
        print(
        '{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format(args.model_name+savename, (epoch + 1), int(time.time() - begin_epoch),
                train_metrics['train_loss'], val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
flush=True)

    # save model
        if val_metrics['mean_recall'] > best_val_recall:
            torch.save(model.state_dict(), f'./checkpoints/{args.model_name}/{savename}_best.ckpt')
            best_val_recall = val_metrics['mean_recall']
            should_stop=0
        else:
            should_stop+=1
            print(should_stop)
        
        if should_stop>20:
            print('Early Stop', flush=True)
            break

    print('Training Completed...', flush=True)

    # saving result!
    print('Loading Best Ckpt...', flush=True)
    checkpoint = torch.load(f'checkpoints/{args.model_name}/{savename}_best.ckpt')
    model.load_state_dict(checkpoint)
    test_evaluator = Evaluator(model, k=3)
    check_metrics = test_evaluator.eval_recall(val_dataloader,args)
    if best_val_recall == check_metrics['mean_recall']:
        print('Successfully load best checkpoint with acc {:.2f}'.format(
        100 * best_val_recall),
          flush=True)
    else:
        print('Fail to load best checkpoint')
    result = test_evaluator.submit(test_dataloader,args)

    # save into the file
    with open(f'results/{args.model_name}/{savename}_{best_val_recall}.txt', 'w') as writer:
        for label_list in result:
            a = [str(x) for x in label_list]
            save_str = ' '.join(a)
            writer.writelines(save_str + '\n')
    print('Result Saved!', flush=True)

    return



if __name__ =='__main__':
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit_crop_adam')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--mode', type=str, default='cnn')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    train(args)
   

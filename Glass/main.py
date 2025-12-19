#-*- coding: utf-8 -*-
import argparse
import os
import datetime
import random
import torch
import numpy as np
from torch import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import model
from sklearn.metrics import roc_auc_score
import evaluation
import screening
import pandas as pd
from magloss import mag_loss, spread_loss
import matplotlib.pyplot as plt
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Torch version: {}, Gpu is available: {}'.format(torch.__version__,USE_CUDA))
torch.autograd.set_detect_anomaly(True)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def criterion(anchor,positive,negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    loss = -torch.log(torch.sigmoid((pos_sim-neg_sim)/args.tau)).mean()
    return loss

def magcriterion(anchor,positive,negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    magl = mag_loss()
    loss = -torch.log(torch.sigmoid((pos_sim-neg_sim)/args.tau)).mean() * magl(positive,anchor)
    return loss

def aggressive_magcriterion(anchor,positive,negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    magl = mag_loss()
    loss = magl(positive,anchor)/magl(negative,anchor)
    return loss

def div_magcriterion(anchor,positive,negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    magl = mag_loss()
    loss = -torch.log(torch.sigmoid((pos_sim-neg_sim)/args.tau)).mean() / magl(negative,anchor)
    return loss

def spread_criterion(anchor, positive, negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    spreadl = spread_loss()
    loss = -torch.log(torch.sigmoid((pos_sim-neg_sim)/args.tau)).mean() / spreadl(negative,anchor)
    return loss


def train(net, data_loader, train_optimizer, crit=criterion, max_batches=None):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for batch_idx, (anchor, pos_feature, neg_feature) in enumerate(train_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        anchor, pos_feature, neg_feature = anchor.to(device, non_blocking=True), pos_feature.to(device, non_blocking=True), neg_feature.to(device, non_blocking=True)
        anchor_emb = net(anchor)
        pos_emb = net(pos_feature)
        neg_emb = net(neg_feature)

        # Calculate loss
        loss = crit(anchor_emb, pos_emb, neg_emb)

        # Optimize
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += anchor.size(0) 
        total_loss += loss.item() * anchor.size(0)

        train_bar.set_description(f'Train Epoch: [{epoch}/{args.epochs}] Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num

def list_type(arg):
    return [int(x) for x in arg[1:-1].split(',')]




    return total_top1 / total_num * 100
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--num_workers', default=0, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-7, type=float, help='Weight_decay')
    parser.add_argument('--num_components', default=18, type=int, help='Number of components')
    parser.add_argument('--embedding_dim', default=128, type=int, help='The dimension of the embeddings associated with each component')
    parser.add_argument('--out_dim', default=512, type=int, help='The dimension of the out put feature')
    parser.add_argument('--fm_dim', default=4, type=int, help='The dimension for factorization of the adjacency matrix')
    parser.add_argument('--num_heads', default=1, type=int, help='Number of attention heads')
    parser.add_argument('--tau', default=0.1, type=float, help='Temperature scalling for loss function')
    parser.add_argument('--noise_std', default=0.1, type=float, help='Stanrd deviriation of noise perbulation for data augmentation')
   

    init_seed(2024)
    args = parser.parse_args()
    print(args)

    ####################### Step1: Data Preparation #######################
    interval= [500,600]
    print('The interval of glass transition temperatures to be SCREENED:', interval)
    train_path = args.root + '/train_tg.csv'
    valid_path = args.root + '/validation_tg.csv'
    test_path = args.root + '/test_tg.csv'
    traindata = utils.load_train(train_path) 
    validdata = utils.load_validate(valid_path)
    testdata = utils.load_test(test_path)
    mean, std = traindata.mean(axis=0) [:args.num_components], traindata.std(axis=0)[:args.num_components]
    train_data = utils.MyData(traindata, mean, std, args.num_components, interval, args.noise_std, phase = 'Training')
    memor_data = utils.MyData(traindata, mean, std, args.num_components, interval, args.noise_std, phase = 'Evaluation')
    valid_data = utils.MyData(validdata, mean, std, args.num_components, interval, args.noise_std, phase = 'Evaluation')
    test_data  = utils.MyData(testdata , mean, std, args.num_components, interval, args.noise_std, phase = 'Screening')
    print("Number of training samples within desired GT :{}; Number of training samples out of desired GT interval:{}".format(sum(train_data.label),len(train_data)-sum(train_data.label)))
    print("Number of validating samples within desired GT :{}; Number of validating samples out of desired GT interval:{}".format(sum(valid_data.label),len(valid_data)-sum(valid_data.label)))
    print('Number of testing samples to be SCREENED :{}'.format (len(test_data)))
    train_loader = DataLoader(train_data,                   
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=args.num_workers)
    memor_loader = DataLoader(memor_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers
                              )
    test_loader = DataLoader(test_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers
                              )


    ######################## Step2: Model Setup #######################
    losses_to_train = ['division magnitude', 'division spread', 'original']
    loss_dict = {
        'magnitude': magcriterion,
        'original': criterion,
        'aggressive magnitude': aggressive_magcriterion,
        'division magnitude': div_magcriterion,
        'division spread': spread_criterion,
    }

    if not os.path.exists('./results'):
        os.makedirs('./results')

    all_results = {name: {} for name in ['Precision', 'AUC', 'F1 Macro', 'F1 Micro', 'PR-AUC']}

    for loss in losses_to_train:
        modelr = model.DeepGlassNet(args.num_components, args.embedding_dim, args.fm_dim, args.num_heads, args.out_dim).to(device)

        ######################## Step3: Optimizer Config #######################
        optimizer = torch.optim.Adam(modelr.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)

        ######################## Step4: Model Training #######################
        result = []
        epochs_list = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train(modelr, train_loader, optimizer, loss_dict[loss])
            
            if epoch % 1 == 0:
                pre, auc, f1mac, f1mic, prauc = evaluation.eval(modelr, memor_loader, valid_loader)
                result.append([pre, auc, f1mac, f1mic, prauc])
                epochs_list.append(epoch)
                print('Validation Epoch ({}): [{}/{}]: Precision:{:.1f}%, AUC:{:.4f}, F1 Macro:{:.4f}, F1 Micro:{:.4f}, PR-AUC:{:.4f}'
                    .format(loss, epoch, args.epochs, pre * 100, auc, f1mac, f1mic, prauc))

            if epoch % 1 == 0:
                screened_id = screening.screen(modelr, memor_loader, test_loader, epoch, args)
                print('Top-10 Screened Samples at Epoch ({}): [{}/{}]'.format(loss, epoch, args.epochs))
                predict = testdata[screened_id]
                print(predict)

        result_np = np.array(result)
        metric_names = ['Precision', 'AUC', 'F1 Macro', 'F1 Micro', 'PR-AUC']
        for i, name in enumerate(metric_names):
            all_results[name][loss] = (epochs_list.copy(), result_np[:, i].copy())

        best_rest = result_np.max(axis=0)
        best_idx = result_np.argmax(axis=0)
        print('Best Result for {}: (Precision:{} at Epoch: {}), (AUC:{:.4f} at Epoch {}), (F1 Macro:{:.4f} at Epoch {}), (F1 Micro:{:.4f} at Epoch {}), (PR-AUC:{:.4f} at Epoch {})'
            .format(loss, best_rest[0], best_idx[0] + 1,
                            best_rest[1], best_idx[1] + 1,
                            best_rest[2], best_idx[2] + 1,
                            best_rest[3], best_idx[3] + 1,
                            best_rest[4], best_idx[4] + 1))
        print('\t')
    
    
    
    for metric_name, loss_results in all_results.items():
        plt.figure(figsize=(10, 6))
        for loss, (epochs, values) in loss_results.items():
            plt.plot(epochs, values, label=loss)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} across Loss Functions')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/all_losses_{metric_name.replace(" ", "_").lower()}.png')
        plt.close()
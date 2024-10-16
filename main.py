import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
import pickle
import time
import argparse

from torch.utils.data import DataLoader

from utils.tools import *
from model import CTGVAEModel


# args
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--lambda2', type=float, default=0.2)
parser.add_argument('--b_size', type=int, default=64)
parser.add_argument('--max_seq', type=int, default=100)
parser.add_argument('--emb_dim', type=int, default=80)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')


parser.add_argument('--input', type=str, default='./dataset/sample/')
parser.add_argument('--dataset', type=str, default='sample')


def accu_loss(pre, y, z, adj_rec, adj, x_rec, x, model):
    mse_loss = F.mse_loss(torch.log2(pre + 1), torch.log2(y + 1))
    stru_loss = F.binary_cross_entropy(adj_rec.view(-1), adj.view(-1))
    kl_divergence_1 = (0.5 / adj.size(0) * (1 + 2 * model.vgae_model.log_std - model.vgae_model.mean ** 2 - torch.exp(model.vgae_model.log_std) ** 2).sum(1).mean())
    stru_loss -= kl_divergence_1
    feat_loss = torch.mean(torch.square(x - x_rec))
    kl_divergence_2 = 0.5 * torch.mean(model.vgte_model.log_std - torch.square(model.vgte_model.mean) - torch.exp(model.vgte_model.log_std) + 1)
    feat_loss -= kl_divergence_2
    loss = args.lambda1 * mse_loss + args.lambda2 * stru_loss + (1 - args.lambda1 - args.lambda2) * feat_loss
    return loss


def collate(samples):
    # 输入`samples` 是一个列表
    # 每个元素都是一个dict 包含三个元素
    embs = list()
    labels = list()
    graphs = list()
    for sample in samples:
        embs.append(sample['x'])
        labels.append(sample['y'])
        graphs.append(sample['g'])
    batched_graph = dgl.batch(graphs)
    return {'g':batched_graph, 'x':torch.tensor(np.array(embs)).to(torch.float32), 'y':torch.tensor(np.array(labels)).to(torch.float32)}

def evaluate(model, accu_loss, dataloader):
    model.eval()
    total_loss = 0.0
    total_error_square = 0.0
    total_error_abs = 0.0
    total_batch = 0.0
    with torch.no_grad():
        for batch, batch_samples in enumerate(dataloader):
            b_g = batch_samples['g'].to(args.device)
            b_x = batch_samples['x'].to(args.device)
            b_y = batch_samples['y'].to(args.device)
            b_adj = b_g.adjacency_matrix().to_dense().to(args.device)
            total_batch += 1

            b_pre, b_z, b_adj_rec, b_x_rec = model(b_g, b_x)
            b_loss = accu_loss(b_pre, b_y, b_z, b_adj_rec, b_adj, b_x_rec, b_x, model)

            b_msle = np.mean(np.square(np.log2(b_y.cpu().numpy() + 1) - np.log2(b_pre.cpu().numpy() + 1)))
            b_mape = np.mean(
                np.abs(np.log2(b_y.cpu().numpy() + 1) - np.log2(b_pre.cpu().numpy() + 1)) / np.log2(b_y.cpu().numpy() + 2))
            # print('--------------Batch {:d} | Val Loss {:.4f} | Val MSLE {:.4f} | Val MAPE {:.4f}'.format(
            #    batch + 1, b_loss.item(), b_msle, b_mape))

            total_error_square += b_msle
            total_error_abs += b_mape
            total_loss += b_loss.item()


        msle = total_error_square / total_batch
        mape = total_error_abs / total_batch
        loss = total_loss

    return loss, msle, mape

def main(args):
    start_time = time.time()

    with open(args.input + 'train.pkl', 'rb') as ftrain:
        train_cascade, train_global, train_label, train_graph = pickle.load(ftrain)
    with open(args.input + 'val.pkl', 'rb') as fval:
        val_cascade, val_global, val_label, val_graph = pickle.load(fval)
    with open(args.input + 'test.pkl', 'rb') as ftest:
        test_cascade, test_global, test_label, test_graph = pickle.load(ftest)


    #dataset
    train_dataset = dglDataset(train_cascade, train_global, train_label, train_graph, args.b_size, args.max_seq)
    val_dataset = dglDataset(val_cascade, val_global, val_label, val_graph, args.b_size, args.max_seq, is_train=False)
    test_dataset = dglDataset(test_cascade, test_global, test_label, test_graph, args.b_size, args.max_seq, is_train=False)

    #dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.b_size, shuffle=False, num_workers=1, collate_fn=collate)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.b_size, shuffle=False, num_workers=1, collate_fn=collate)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.b_size, shuffle=False, num_workers=1, collate_fn=collate)

    #model
    model = CTGVAEModel(in_dim = args.emb_dim, hidden_dim1 = args.hidden_dim,  z_dim = args.z_dim, out_dim = 1, max_seq = args.max_seq, device = args.device, hidden_dim2 = 128, hidden_dim3 = 64).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    stopper = EarlyStopping(dataset=args.dataset, patience=args.patience)


    #train
    for epoch in range(args.epoch):
        model.train()
        total_batch = 0.0
        total_error_square = 0.0
        total_error_abs = 0.0
        total_loss = 0.0

        print("--------------------Epoch {:d}----------------------".format(epoch + 1))
        for batch, batch_samples in enumerate(train_loader):
            b_g = batch_samples['g'].to(args.device)
            b_x = batch_samples['x'].to(args.device)
            b_y = batch_samples['y'].to(args.device)
            b_adj = b_g.adjacency_matrix().to_dense().to(args.device)
            total_batch += 1


            b_pre, b_z, b_adj_rec, b_x_rec = model(b_g, b_x)
            b_loss = accu_loss(b_pre, b_y, b_z, b_adj_rec, b_adj, b_x_rec, b_x, model)
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()

            b_msle = np.mean(np.square(np.log2(b_y.cpu().detach().numpy() + 1) - np.log2(b_pre.cpu().detach().numpy() + 1)))
            b_mape = np.mean(np.abs(np.log2(b_y.cpu().detach().numpy() + 1) - np.log2(b_pre.cpu().detach().numpy() + 1)) / np.log2(b_y.cpu().detach().numpy() + 2))
            #print('--------------Batch {:d} | Train Loss {:.4f} | Train MSLE {:.4f} | Train MAPE {:.4f}'.format(
             #   batch + 1, b_loss.item(), b_msle, b_mape))

            total_error_square += b_msle
            total_error_abs += b_mape
            total_loss += b_loss.item()


        msle = total_error_square / total_batch
        mape = total_error_abs / total_batch
        loss = total_loss
        print('Epoch {:d} | Train Loss {:.4f} | Train MSLE {:.4f} | Train MAPE {:.4f}'.format(epoch + 1, loss, msle, mape))

        # val
        val_loss, val_msle, val_mape = evaluate(model, accu_loss, val_loader)
        print('Epoch {:d} | Val Loss {:.4f} | Val MSLE {:.4f} | Val MAPE {:.4f}'.format(
            epoch + 1, val_loss, val_msle, val_mape))
        early_stop = stopper.step(val_loss, val_msle, val_mape, model)
        if early_stop == True:
            break


    #TEST
    stopper.load_checkpoint(model)
    test_loss, test_msle, test_mape = evaluate(model, accu_loss, test_loader)
    print('Test loss {:.4f} | Test MSLE {:.4f} | Test MAPE {:.4f}'.format(test_loss, test_msle, test_mape))

    print('Finished! Time used: {:.3f}mins.'.format((time.time() - start_time) / 60))





if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)




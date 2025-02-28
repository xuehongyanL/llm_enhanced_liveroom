import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

from liveroom.data.dataset import FusionDataset, LLMEnhancedDataset
from liveroom.models.rnn import SalesPredictionModel
from liveroom.utils.random import reseed_everything

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--device', type=str, required=True)

parser.add_argument('--df', type=str, required=True)
parser.add_argument('--entry', type=str, required=True)
parser.add_argument('--state', type=str, required=True)
parser.add_argument('--action', type=str, required=True)
parser.add_argument('--read', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)

parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--temperature', type=float, default=500)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=0.96)

args = parser.parse_args()
print(args)

device = torch.device(args.device)


def run_exp(seed: int):
    print(f'Using seed {seed}')
    reseed_everything(seed)

    assert args.task in ['causal']

    data = FusionDataset(filename=args.df)
    data = LLMEnhancedDataset(fusion=data,
                              entry=args.entry,
                              action=args.action,
                              read=args.read,
                              state=args.state,)

    tabular_input_dim = len(data.fusion.x_columns)
    llm_input_dim = 32

    model = SalesPredictionModel(tabular_input_dim,
                                 llm_input_dim,
                                 args.hidden_dim,
                                 temperature=args.temperature,
                                 dropout=args.dropout).to(device)

    train_size, valid_size = len(data) * 7 // 10, len(data) // 10
    train_set = Subset(data, range(0, train_size))
    valid_set = Subset(data, range(train_size, train_size + valid_size))
    test_set = Subset(data, range(train_size + valid_size, len(data)))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=None)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=None)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=None)

    mse_loss = nn.MSELoss().to(device)
    mae_loss = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.96)

    def eval_model(dataloader):
        model.eval()
        total_cnt = 0
        total_loss_mse = 0.
        total_loss_mae = 0.
        total_loss_kl1 = 0.
        total_loss_kl2 = 0.
        for batch in dataloader:
            x, y_true = batch['x'].to(device), batch['y'].to(device)
            y_pred = model(x[:, :tabular_input_dim + llm_input_dim])

            loss_mse = mse_loss(y_pred[:, :1], y_true[:, :1])
            loss_mae = mae_loss(y_pred[:, :1], y_true[:, :1])
            loss_kl1 = F.kl_div((y_pred[:, 4:12] + 1e-9).log(), y_true[:, 4:12], reduction='batchmean')
            loss_kl2 = F.kl_div((y_pred[:, 12:20] + 1e-9).log(), y_true[:, 12:20], reduction='batchmean')

            total_cnt += x.shape[0]
            total_loss_mse += loss_mse.item() * x.shape[0]
            total_loss_mae += loss_mae.item() * x.shape[0]
            total_loss_kl1 += loss_kl1.item() * x.shape[0]
            total_loss_kl2 += loss_kl2.item() * x.shape[0]
        res = (math.sqrt(total_loss_mse / total_cnt),
            total_loss_mae / total_cnt,
            total_loss_kl1 / total_cnt,
            total_loss_kl2 / total_cnt)
        return f'({res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f})'

    print(f'Init loss: {eval_model(train_loader)}{eval_model(test_loader)}')
    for epoch in range(args.n_epoch):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            x, y_true, w = batch['x'].to(device), batch['y'].to(device), batch['w'].squeeze().to(device)

            y_pred = model(x[:, :tabular_input_dim + llm_input_dim])
            loss_mse = F.mse_loss(y_pred[:, :4], y_true[:, :4], reduction='none').mean(dim=1)

            loss_kl1 = F.kl_div((y_pred[:, 4:12] + 1e-9).log(), y_true[:, 4:12], reduction='none').mean(dim=1)
            loss_kl2 = F.kl_div((y_pred[:, 12:20] + 1e-9).log(), y_true[:, 12:20], reduction='none').mean(dim=1)

            mse_weight = 1.
            kl_weight = max(1. - (batch_idx * 0.025), 0.)
            sample_weight = 1 - w.clamp(-0.25, 0.25)

            loss = (sample_weight * (mse_weight * loss_mse + kl_weight * loss_kl1 + kl_weight * loss_kl2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{args.n_epoch}], Loss: {eval_model(train_loader)}{eval_model(test_loader)}')
        if (epoch + 1) % 10 == 0:
            torch.save(model, f'{args.out_dir}/seed{seed}_epoch{epoch+1}.pth')

    print('Et Voilà!')


if __name__ == '__main__':
    run_exp(seed=args.seed)

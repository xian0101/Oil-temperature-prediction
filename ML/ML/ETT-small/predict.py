import sys
import os
from models.Mtsfn import MtsfnModel
from utils.save_checkpoint import save_checkpoint
from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
import numpy as np
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from args import args
import matplotlib.pyplot as plt

def _get_data(flag):
    
    if flag == 'test' or flag == 'pred':
        args.data_path = 'test_set.csv'
    elif flag == 'val':
        args.data_path = 'validation_set.csv'
    else:
        args.data_path = 'train_set.csv'
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def predict(model):
    pred_data, pred_loader = _get_data(flag='pred')
    preds = []
    model.load_state_dict(torch.load('/home/ysma/project1/ETTH/checkpoints/model_cur.pth'))
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()
            # decoder input
            dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)
    preds = np.array(preds)
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    print(preds.shape)
    return

if __name__ == '__main__':
    # 实例化模型
    model = MtsfnModel(
        args
    ).float().cuda()
    
    predict(model)
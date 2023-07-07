import torch
from time import time
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

class Trainer():
    
    def __init__(self, model, optimizer, loss, metrics, device, logger, amp, interval=100):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.logger = logger
        self.amp = amp
        self.interval = interval
        
        # History
        self.loss_sum = 0 # Epoch loss sum
        self.loss_mean = 0 # epoch loss mean
        self.filenames = list()
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()
        self.elapsed_time = 0
        #learning rate scheduler
        self.scheduler = None
        
    def train(self, mode, dataloader, epoch_index=0):
        start_timestamp = time()
        self.model.train() if mode == 'train' else self.model.eval()
        
        for batch_index, (x, y, filename) in enumerate(tqdm(dataloader)):
            x, y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float32)
            
            # 혼합 정밀도 컨텍스트 활성화
            with autocast(enabled=self.amp is not None):
                # Inference
                y_pred = self.model(x)
                y_pred = y_pred.squeeze(-1)
                
                # Loss
                # y = y.unsqueeze(1)  # y의 크기를 [64]에서 [64, 1]로 변경 -> gpt가 한일..
                loss = self.loss(y_pred, y)
                
            # Update cosine annealing
            if mode == 'train':
                self.optimizer.zero_grad()
                
                if self.amp is None:
                    loss.backward()
                else:
                    # 스케일러 생성
                    scaler = GradScaler()
                    
                    # 손실 스케일링
                    scaler.scale(loss).backward()
                    
                    # 가중치 업데이트
                    scaler.step(self.optimizer)
                    
                    # 스케일 업데이트
                    scaler.update()
                    
                self.optimizer.step()
            
            elif mode in ['val','test']:
                pass
            else:
                raise ValueError('Mode should be either train, val, or test')
            
            # History
            # self.filenames += filename
            self.loss_sum += loss.item()
            y_pred = (y_pred > 0.5).to(torch.int)
            self.y_preds.append(y_pred)
            self.y.append(y)
            
            # Logging
            if batch_index % self.interval == 0:
                msg = f"batch: {batch_index}/{len(dataloader)} loss: {loss.item()}"
                self.logger.info(msg)
                
        # Epoch history
        self.loss_mean = self.loss_sum / len(dataloader)
        
        # Metric
        self.y_preds = torch.cat(self.y_preds, dim=0).cpu().tolist()
        self.y = torch.cat(self.y, dim=0).cpu().tolist()
        
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(self.y, self.y_preds)
            self.score_dict[metric_name] = score
        
        # Elapsed time
        end_timestamp = time()
        self.elapsed_time = end_timestamp - start_timestamp
        
    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()
        self.elapsed_time = 0
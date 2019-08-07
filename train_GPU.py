import time
import datetime
import os
import logging
import pdb

from timeit import default_timer as timer

import torch
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel



from data import *
from models import *
from utils import *



def crossentropyLoss(out, labels):
    return F.cross_entropy(out, labels)


def mseLoss(out, labels):
    return torch.nn.MSELoss(out, labels)


def prediction(model, validation_loader, num_classes=5, batch_size=36):
    with torch.no_grad():
        model.eval()
#        model.mode = 'valid'
        valid_loss, correctpred = 0, 0
        num_batch = len(validation_loader)
        for valid_data in validation_loader:
            images, labels = valid_data
            images = images.cuda()
            labels = labels.cuda()
            out = data_parallel(model, images)
#            out = model(images)
            valid_loss += mseLoss(out, labels)
##Softmax Loss:
#            scores = F.softmax(out, dim=1)
#            _, results = torch.max(scores, 1)
#            correctpred += torch.sum(results==labels)
        
        valid_loss /= num_batch
#        accuracy = correctpred.float()/(num_batch*batch_size)
        accuracy = 0
        return valid_loss, accuracy




def train(fold_index=3, num_classes=1, model_name='resnet101', checkPoint_start=0, lr=3e-4, batch_size=36):
    #Build the model:
    model = model_blindness(num_classes=1, inchannels=3, model_name=model_name).cuda()
    
    #Training parameters:
    epoch = 0
    iter_smooth = 100
    iter_valid = 200
    MAX_BATCH = 10000000
    
    #Choose the optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.9, 0.99), weight_decay=0.0002)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
    
    #Results and log:
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    os.makedirs(resultDir,exist_ok=True)
    ImageDir = resultDir + '/image'
    checkPoint = os.path.join(resultDir, 'checkpoint')
    os.makedirs(checkPoint, exist_ok=True)
    os.makedirs(ImageDir, exist_ok=True)
    
    log = Logger()
    log.open(os.path.join(resultDir, 'log_train.txt'), mode= 'a')
    log.write(' start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.write(' batch_size :{} \n'.format(batch_size))
    
    
    #Prepare the dataset:
    dataset = BlindnessDataset(mode='train', transform=None)
    train_loader, validation_loader = train_valid_dataset(dataset, batch_size, 0.2, True)
    num_image = len(dataset)
    iter_save = 2*num_image//batch_size
    
    
    #Initializa the losses:
    train_loss = 0.0
    valid_loss = 0.0
    batch_loss = 0.0
    train_loss_sum = 0
    iter_num = 0
    
    i = 1  #batch index
    #If need to load the previous model:
    skips = []
    if not checkPoint_start == 0:
        log.write('  start from{}, l_rate ={} \n'.format(checkPoint_start, lr))
        log.write('freeze={}, batch_size={}\n'.format(freeze, batch_size))
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=skips)
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        optimizer.load_state_dict(ckp['optimizer'])
        adjust_learning_rate(optimizer, lr)
        i = checkPoint_start
        epoch = ckp['epoch']
        
    log.write(
            ' rate     iter   epoch |   valid   train   batch  |Accuracy  time          \n')
    log.write(
            '----------------------------------------------------------------------------\n')
    
    start = timer()
    start_epoch = epoch
    cycle_epoch = 0
    
    #Freeze base-model layers:
    model.freeze()
    #set to train mode:
    model.train()
    
    while i < MAX_BATCH:
        for batchdata in train_loader:
            epoch = start_epoch + (i - checkPoint_start) * batch_size/num_image
            #We check the validation set @iter_valid:
            if i % iter_valid==0:
                valid_loss, valid_accuracy = prediction(model, validation_loader, num_classes=5, batch_size=batch_size)
                print('\r', end='', flush=True)
                log.write(
                    '%0.5f %5.2f k %5.2f  | %0.3f    %0.3f    %0.3f |%0.3f    %s \n' % ( \
                        lr, i / 1000, epoch, valid_loss, train_loss, batch_loss, valid_accuracy, 
                        time_to_str((timer() - start) / 60)))
                print('epoch=',epoch, 'valid_loss=',valid_loss)
                time.sleep(0.01)
                #set model back to train mode:
                model.train()
            
            #We save the training @iter_save
            if i % iter_save == 0 and (not i == checkPoint_start):
                torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                }, resultDir + '/checkpoint/%08d_optimizer.pth' % (i))

            images, labels = batchdata
#            pdb.set_trace()
#            print(images.shape)
#            N, H, W, C = images.shape
#            images = images.view(N, C, H, W).float()
            #We use GPU in GCP:
            images = images.cuda()
            labels = labels.cuda()
#            global_feat, local_feat, results = data_parallel(model,images)
            out = data_parallel(model,images)
            out = model(images)
            batch_loss = getLoss(out, labels)
            #Backpropagation:
            optimizer.zero_grad()
            batch_loss.backward()
#            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
#            results = torch.cat([torch.sigmoid(results), torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
            #move back to CPU and transform to numpy:
            batch_loss = batch_loss.data.cpu().numpy()
            train_loss_sum += batch_loss
            iter_num += 1
            
            if (i + 1) % iter_smooth == 0:
                train_loss = train_loss_sum/iter_num
                train_loss_sum = 0
                iter_num = 0

            print('\r%0.5f %5.2f k %5.2f  | %0.3f    %0.3f    %0.3f | %s  %d %d' % ( \
                    lr, i / 1000, epoch, valid_loss, train_loss, batch_loss,
                    time_to_str((timer() - start) / 60), checkPoint_start, i)
                , end='', flush=True)
            i += 1
           
        pass



if __name__ == '__main__':
    if 1:
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,5'
        freeze = False
        num_classes = 5
        model_name = 'resnet101'
        fold_index = 3
        checkPoint_start = 1824
        lr = 0.2e-4
        batch_size = 32
        train(fold_index, num_classes, model_name, checkPoint_start, lr, batch_size)

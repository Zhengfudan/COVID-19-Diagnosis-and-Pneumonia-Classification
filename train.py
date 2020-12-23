from my_dataset import Xinguan, get_data
import torchvision 
import torch.nn as nn
import torch
import time
import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from my_evaluate import evaluate
from senet.se_resnet import se_resnet50
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
parser.add_argument("--pretrained_weights",type=str, help="if specified starts from checkpoint model")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--num_workers", type=int, default=0, help="interval between saving model weights")
parser.add_argument("--data_path", type=int, default='/multi_classifier/split_info', help="training dataset")

opt = parser.parse_args()
split_info_path = opt.data_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    # load data
    print('--------------------loading data--------------------')
    # xinguan
    img_paths, targets, patient_ids, img_ids = get_data(split_info_path)
    trainset = Xinguan(img_paths[0], targets[0], patient_ids[0], img_ids[0])
    validset = Xinguan(img_paths[1], targets[1], patient_ids[1], img_ids[1])

    dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        
    print('-------------------data have loaded------------------')
    
    # init model, torchvision---resnet
    model = se_resnet50(num_classes=4)
    model = model.to(device)
    writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batch_size}')     
    
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))
       
    # 数据多个GPU并行运算！（单机多卡~）
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    max_recall = 0
    loss_list = []
    sig = nn.Softmax(dim=1)
    global_step = 0
    best_auc_epoch = 0
    best_auc = 0
    best_auc_recall = 0
    for epoch in range(opt.epochs):
        model.train() 
        start_time = time.time()

        # start training.
        for batch_i, batch_data in enumerate(dataloader):
            
            inputs = Variable(batch_data['image'].to(device),requires_grad=True)
            labels = Variable(batch_data['target'].to(device),requires_grad=False)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   

            predicted = outputs.data
            _, predicted = torch.max(predicted, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            acc = correct/total

            # log
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch + 1, opt.epochs, batch_i, len(dataloader))
            log_str += f"\n loss: {loss.item()}"
            log_str += f"\n acc: {acc}"
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('acc/train', acc, global_step)
            global_step += 1

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)   

        # 验证的时候单卡。
        per_img_res, per_patient_res = evaluate(model,validset)  
        ############
        precision = per_img_res['precision'] # dict
        recall =  per_img_res['recall']      # dict
        acc =  per_img_res['acc']            # float
        F1score =  per_img_res['F1score']   # dict
        auc =  per_img_res['auc']           # dict
        writer.add_scalars('Presion/test per img', precision, epoch)
        writer.add_scalars('Recall/test per img', recall, epoch) 
        writer.add_scalar('acc/test per img', acc, epoch)
        writer.add_scalars('F1score/test per img', F1score, epoch)
        writer.add_scalars('auc/test per img', auc, epoch)
        ############
        precision = per_patient_res['precision']
        recall =  per_patient_res['recall']
        acc =  per_patient_res['acc']
        F1score =  per_patient_res['F1score']
        auc =  per_patient_res['auc']

        writer.add_scalars('Presion/test per patient', precision, epoch)
        writer.add_scalars('Recall/test per patient', recall, epoch) 
        writer.add_scalar('acc/test per patient', acc, epoch)
        writer.add_scalars('F1score/test per patient', F1score, epoch)
        writer.add_scalars('auc/test per patient', auc, epoch)

        if epoch % opt.checkpoint_interval == 0:        
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")
        '''
        if 1.2*auc + recall > best_auc:
            best_auc = 1.2*auc + recall  
            best_auc_epoch = epoch 
        '''
    writer.close()
    testset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    evaluate(model,testset)
          
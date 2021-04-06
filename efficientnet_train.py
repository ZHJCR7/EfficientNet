"""
Video Face Manipulation Detection Through Ensemble of CNNs

Author: Hongjie Zhao
"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from network import fornet
import argparse
import os

from dataset.transform import efficientnet_data_transforms
from dataset.mydataset import MyDataset

def main():
    args = parse.parse_args()
    name = args.name
    train_list = args.train_list
    val_list = args.val_list
    batch_size =args.batch_size
    model_name = args.model_name
    initial_lr = args.lr
    epoches = args.epoches
    device = torch.device('cuda:{:d}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    net_class = getattr(fornet, args.net)

    output_path = os.path.join('./output', name).replace("\\", "/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.backends.cudnn.benchmark = True

    train_dataset = MyDataset(txt_path=train_list, transform=efficientnet_data_transforms['train'])
    val_dataset = MyDataset(txt_path=val_list, transform= efficientnet_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    # load model
    model: nn.Module = net_class().to(device)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=initial_lr, betas=(0.9,0.999), eps=1e-8)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        cooldown=20,
        min_lr=initial_lr * 1e-5
    )

    # Data load
    print("Loading data")
    print("The size of train data is:" + str(train_dataset_size))
    print("The size of validation data is:" + str(val_dataset_size))
    model = nn.DataParallel(model)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0

    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch+1,epoches))
        print('-'*10)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0

        # Train model
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.to(device)
            labels = labels.to(device)
            # labels = torch.unsqueeze(labels, 1).float()
            optimizer.zero_grad()
            outputs = model(image)
            # preds = torch.sigmoid(outputs)
            preds = torch.max(outputs, dim=1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 100):
                print('iteration {} train loss: {:.4f} Acc: {:4f}'.format(iteration, iter_loss/batch_size, iter_corrects/batch_size))

        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        #Validate mode
        model.eval()
        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.to(device)
                labels = labels.to(device)
                # labels = torch.unsqueeze(labels, 1).float()
                outputs = model(image)
                # preds = torch.sigmoid(outputs)
                preds = torch.max(outputs, dim=1)[1]
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        if not (epoch % 5):
            torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #EfficientNetAutoAttB4 or EfficientNetB4
    parse.add_argument('--net', type=str, default='EfficientNetAutoAttB4')
    parse.add_argument('--name', '-n', type=str, default='ffpp_efficient_c23')
    parse.add_argument('--model_name', '-mn', type=str, default='ffpp_c23.pkl')
    parse.add_argument('--train_list', '-tl', type=str, default='./data_list/Img_list_c23_train_path.txt')
    parse.add_argument('--val_list', '-vl', type=str, default='./data_list/Img_list_c23_val_path.txt')
    parse.add_argument('--batch_size', '-bz' ,type=int, default=2)
    parse.add_argument('--epoches', '-e', type=int, default=100)
    parse.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parse.add_argument('--device', type=int, help='GPU device id', default=0)
    main()

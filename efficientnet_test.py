"""
Video Face Manipulation Detection Through Ensemble of CNNs

Author: Hongjie Zhao
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import fornet
from scipy.io import savemat
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataset.transform import efficientnet_data_transforms
from dataset.mydataset import MyDataset

def main():
    args = parse.parse_args()
    name = args.name
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path

    epoch_number = args.model_path.split('/')[-1].split('_')[0]
    result_path = os.path.join('./result', name).replace("\\", "/")
    result_path = result_path + '/' + epoch_number + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    device = torch.device('cuda:{:d}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    test_dataset = MyDataset(txt_path=test_list, transform=efficientnet_data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0

    # load model
    print('Loading model...')
    net_class = getattr(fornet, args.net)
    model = net_class().to(device)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.eval()
    iteration = 0

    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            preds = torch.max(outputs, dim=1)[1]
            # acc_new = (preds == labels).float().mean()
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            res = {'lab': labels, 'score' : outputs, 'pred' : preds}
            results = {}
            for r in res:
                results[r] = res[r].squeeze().cpu().numpy()

            savemat('{0}{1}.mat'.format(result_path, iteration), results)
            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32) / batch_size))
            iteration += 1
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--net', type=str, default='EfficientNetAutoAttB4')
    parse.add_argument('--name', '-n', type=str, default='ffpp_efficientatt_c23')
    parse.add_argument('--batch_size', '-bz', type=int, default=4)
    parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Img_list_c23_test_path.txt')
    parse.add_argument('--model_path', '-mp', type=str, default='./output/ffpp_efficientatt_c23/40_ffpp_c23.pkl')
    parse.add_argument('--device', type=int, help='GPU device id', default=0)
    main()
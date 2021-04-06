import argparse
import glob
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def compute_result_file(rfn):
    rf = loadmat(rfn)
    res = {}
    for r in ['lab', 'score', 'pred']:
        res[r] = rf[r].squeeze()
    return res

def main():
    args = parse.parse_args()
    epoch = args.epoch
    result = args.result

    mat_file_path = result + str(epoch) + '/'
    mat_file = glob.glob(mat_file_path + '*.mat')
    print('{0} result files'.format(len(mat_file)))

    #complie the results into a single variable for processing
    total_results = {}
    for rfn in mat_file:
        rf = compute_result_file(rfn)
        for r in rf:
            if r not in total_results:
                total_results[r] = rf[r]
            else:
                total_results[r] = np.concatenate([total_results[r], rf[r]], axis=0)

    print('Found {0} total images with scores.'.format(total_results['lab'].shape[0]))
    print('  {0} results are real images'.format((total_results['lab'] == 1).sum()))
    print('  {0} results are fake images'.format((total_results['lab'] == 0).sum()))

    #compute the performance numbers
    pred_acc = (total_results['lab'] == total_results['pred']).astype(np.float32).mean()
    fpr, tpr, thresh = metrics.roc_curve(total_results['lab'], total_results['score'][:,1], drop_intermediate=False)
    AUC = auc(fpr, tpr)
    fnr = 1 -tpr
    eer = fnr[np.argmin(np.absolute(fnr - fpr))]

    #print out the performance numbers
    print('Prediction Accuracy: {0:.4f}'.format(pred_acc))
    print('AUC: {0:.4f}'.format(AUC))
    print('EER: {0:.4f}'.format(eer))

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR (%)')
    plt.ylabel('TPR (%)')
    # plt.xscale('log')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--epoch', type=int, default=40)
    parse.add_argument('--result', type=str, default='./result/ffpp_efficientatt_c23/')
    main()
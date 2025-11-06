#!/usr/bin/env python3
# Kiri Wagstaff
# May 14, 2025
# Generate a learning curve for MNIST 10k vs. zip code classification.

import sys
import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv
import random
import math
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from eval_MDW_data import eval_zip_codes
from train_cnn import ImageClassifier, train_model


# Test classifier clf on a set of zip codes with labels
# and nist_ids that index into image data in x_all
def test_zip_codes(clf, classifier, x_all, metadata, zip_code_file):

    labels, writers, nist_id = zip(*metadata)
    
    pred_zcs = []
    #print('Errors:')
    for lab, ids in zip(labels, nist_id):
        # Assemble the multi-digit number as an array of digits from ids
        # for individual digit prediction
        if classifier == 'CNN':
            to_tensor_transform = transforms.ToTensor()
            device = 'mps'
            test_data = [to_tensor_transform(x_all[id]) for id in ids]
            zc_test = torch.stack(test_data)
            zc_test = zc_test.to(device)
            with torch.no_grad(): # Disable gradient calculation during evaluation
                outputs = clf(zc_test)
            # Max returns values and indices
            _, pred_zc = torch.max(outputs.data, 1)
            pred_zc = pred_zc.to('cpu')
            pred_zc = pred_zc.tolist()
        else:
            # sklearn classifiers
            zc_test = np.array([np.asarray(x_all[id],
                                           dtype=np.uint8).reshape(-1)
                                for id in ids])
            pred_zc = clf.predict(zc_test)

        # Turn it into a string for comparison with the label
        pred_zc = ''.join(map(str, pred_zc))
        #if pred_zc != zc:
        #print(lab, pred_zc)
        #input()
        pred_zcs.append(pred_zc)

    res = eval_zip_codes(metadata, pred_zcs, zip_code_file)
    print('Error on %d zip codes: %.4f' % (len(pred_zcs),
                                           res['Frac_wrong']))
    return res


# Generate a learning curve to compare:
# (1) individual digit recognition on MNIST 10k test set
# (2) zipcode recognition on the 10k zip codes
# Train and evaluate for each number of training items
# (training sets are built cumulatively).
def learning_curve(classifier, x_train_all, y_train_all, x_test, y_test,
                   metadata, x_all, n_train, zip_code_file,
                   n_trials=1, seed=0,):

    res_mnist = np.full((n_trials, len(n_train)), np.nan)
    res_expect = np.full((n_trials, len(n_train)), np.nan)
    res_zipcode = {}
    for r in ['Frac_invalid',  'Frac_valid_wrong', 'Frac_wrong']:
        res_zipcode[r] = np.full((n_trials, len(n_train)), np.nan)

    if classifier == 'RF':
        clf = RandomForestClassifier(n_estimators=200)
        cname = 'RF-200'
    elif classifier == 'SVM':
        clf = SVC(kernel='rbf') # gamma='scale'
        cname = 'SVM'
    elif classifier == 'CNN':
        #n_epochs = 10
        #use_file = f'model-CNN-epochs-{n_epochs}.pt'
        n_epochs = 20
        use_file = f'model-CNN-VGG-epochs-30-e{n_epochs}.pt'
        device = 'mps'
        lr = 1e-3
        cname = 'CNN'
        # get the tensorized MNIST data
        batch_size = 32
        tensor_train = datasets.MNIST(root='data', train=True, download=True,
                                      transform=transforms.ToTensor())
        tensor_test = datasets.MNIST(root='data', train=False, download=True,
                                      transform=transforms.ToTensor())
    else:
        print(f'Unknown classifier {classifier}')
        return
        
    total_train = len(y_train_all)

    labels, writers, nist_id = zip(*metadata)

    for t in range(n_trials):
        random.seed(seed + t) # Repeatable but random
        train_ids = []
        cur_n_train = len(train_ids)
        for i, nt in enumerate(n_train):
            if nt > total_train:
                print(f'Error: cannot select {nt} from {total_train}.')
                break
        
            # Construct or grow the current training set
            n_add = nt - cur_n_train
            # Without replacement
            new_ids = random.sample([j for j in range(0, total_train)
                                     if j not in train_ids], n_add)
            train_ids += new_ids
            cur_n_train = len(train_ids)
            print(len(train_ids), len(np.unique(train_ids)))

            # Train from scratch
            if classifier == 'CNN':
                clf = ImageClassifier().to(device)
                optimizer = Adam(clf.parameters(), lr=lr)
                loss_fn = CrossEntropyLoss()

                # Load only the specified ids
                tensor_loader = DataLoader(tensor_train[train_ids],
                                           batch_size=batch_size, shuffle=True)

                for epoch in range(n_epochs):
                    # Train in batches
                    for images, labels in tensor_loader:
                        images, labels = images.to(device), labels.to(device)
            
                        # Forward pass
                        outputs = clf(images)
                        loss = loss_fn(outputs, labels)
            
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            else:
                # sklearn classifiers
                clf.fit(x_train_all[train_ids], y_train_all[train_ids])
            print(f'{len(train_ids)}: Training complete')

            # Evaluate on MNIST 10k
            if classifier == 'CNN':
                # Set it to evaluation mode (disable dropout and batch norm)
                clf.eval()
                with torch.no_grad(): # Disable gradient calculation during evaluation
                    for images, labels in tensor_test:
                        images, labels = images.to(device), labels.to(device)
                        outputs = clf(images)
                        # Max returns values and indices
                        _, predicted = torch.max(outputs.data, 1)
                        preds = predicted.to('cpu')
            else:            
                preds = clf.predict(x_test)

            res_mnist[t, i] = 1 - accuracy_score(y_test, preds)
            print('%d: Error on MNIST 10k test: %.4f' %
                  (nt, res_mnist[t, i]))

            # Expected error for five digits
            res_expect[t, i] = 1 - math.pow(1 - res_mnist[t, i], 5)
            print(res_mnist[t,i], res_expect[t,i])

            # Test on the zip codes
            res = test_zip_codes(clf, classifier, x_all, metadata, zip_code_file)
            for r in res:
                res_zipcode[r][t, i] = res[r]
            print('%d: Error on zip codes 10k test: %.4f' %
                  (nt, res_zipcode['Frac_wrong'][t, i]))

            # Save results
            np.savez('results-learning-%s.npz' % cname,
                     res_mnist=res_mnist,
                     res_zipcode=res_zipcode)

            # Plot results
            pl.clf()
            # Plot expected error for 5 digits
            pl.errorbar(n_train, np.nanmean(res_expect, axis=0),
                        yerr=np.nanstd(res_expect, axis=0),
                        color='green', marker='^', ls=':',
                        label='Five MNIST digits')
            pl.errorbar(n_train, np.nanmean(res_zipcode['Frac_wrong'], axis=0),
                        yerr=np.nanstd(res_zipcode['Frac_wrong'], axis=0),
                        color='red', marker='o', ls='-',
                        label='5-digit ZIP Codes')
            pl.errorbar(n_train, np.nanmean(res_mnist, axis=0),
                        yerr=np.nanstd(res_mnist, axis=0),
                        color='blue', marker='x', ls='--',
                        label='Single MNIST digit') # MNIST 10k
            pl.xlabel('Number of training items', fontsize=16)
            pl.ylabel('Test error', fontsize=16)
            bot, top = pl.ylim()
            pl.ylim((0, top))
            pl.legend(fontsize=12)
            pl.xticks(fontsize=12)
            pl.yticks(fontsize=12)
            pl.tight_layout()
            pl.savefig('learning_curve_err-%s-trials-%d.pdf' % (cname, t + 1))


# Train the specified type of classifer on MNIST training set
# and test it on MNIST test and zip_code test
def main(test_file, zip_code_file, classifier, seed):
    random.seed(seed)

    # Load the QMNIST data set that contain writer information
    # compat=False means I want the extended labels
    qtrain = datasets.QMNIST(root='./data', train=True, compat=True,
                             download=True)
    qtest = datasets.QMNIST(root='./data', train=False, compat=False,
                            download=True)
    qall = datasets.QMNIST(root='./data', what='nist',  compat=False,
                           download=True)
    x_all, lab_all = zip(*qall)
    wtrs_all = [int(lab[2]) for lab in lab_all]
    
    x_train_mnist, y_train_mnist = zip(*qtrain)
    x_train_mnist = np.array([np.asarray(im, dtype=np.uint8).reshape(-1)
                              for im in x_train_mnist])
    y_train_mnist = np.array(y_train_mnist)
    
    x_test_mnist, lab_test_mnist = zip(*qtest)
    x_test_mnist = np.array([np.asarray(im, dtype=np.uint8).reshape(-1)
                              for im in x_test_mnist])
    y_test_mnist = np.array([lab[0] for lab in lab_test_mnist])

    # Load in the zip code test file
    metadata = []
    with open(test_file, 'r') as inf:
        csv_rdr = csv.reader(inf)
        # Consume the header
        hdr = next(csv_rdr)
        for row in csv_rdr:
            metadata.append([row[0], row[1], [int(i) for i in row[2:]]])

    # Generate learning curve for individual and zip code tasks
    learning_curve(classifier, x_train_mnist, y_train_mnist,
                   x_test_mnist[:10000], y_test_mnist[:10000],
                   metadata, x_all,
                   [1000, 5000] + list(range(10000, 60001, 10000)),
                   #[1000, 5000],
                   zip_code_file, 10, seed)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--test_file',
                        default='data-multidigit/test_MDW_zip_code.csv',
                        help='File containing test numbers')
    parser.add_argument('-z', '--zip_code_file',
                        default='resources/us_zip_codes_unique.csv',
                        help='CSV file: single column with list of valid U.S. zip codes (default: %(default)s)')
    parser.add_argument('-c', '--classifier', choices=['RF', 'SVM', 'CNN'],
                        default='SVM',
                        help='Classifier to use (default: %(default)s)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed (default: %(default)s)')


    args = parser.parse_args()
    main(**vars(args))


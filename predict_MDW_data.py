#!/usr/bin/env python3
# Kiri Wagstaff
# May 13, 2025
# Generate predictions on MDW test sets.

import sys
import os
import io
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv
import random
import pickle
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from eval_MDW_data import read_MDW_data
from train_cnn import ImageClassifier, train_model
from create_MDW_image import create_multi_digit_image


# Read in NIST images and their metadata from QMNIST
def load_qmnist():

    # compat=False means we want the extended labels (metadata)
    qall = datasets.QMNIST(root='./data', what='nist',  compat=False,
                           download=True)
    images, image_metadata = zip(*qall)
    
    return images, image_metadata


# Read in MNIST training images and their metadata from QMNIST
def load_mnist_train():

    qtrain = datasets.QMNIST(root='./data', train=True, compat=True,
                            download=True)    
    x_train_mnist, y_train_mnist = zip(*qtrain)
    x_train_mnist = np.array([np.asarray(im, dtype=np.uint8).reshape(-1)
                              for im in x_train_mnist])
    y_train_mnist = np.array(y_train_mnist)

    return x_train_mnist, y_train_mnist


# Save the predictions from a CSV file in which each row is:
# prediction (as a string)
def save_predictions(preds_file, preds, verbose=True):

    with open(preds_file, 'w') as outf:
        csv_wtr = csv.writer(outf)
        # Write the header
        csv_wtr.writerow(['Prediction'])
        csv_wtr.writerows(preds)

    if verbose:
        print(f'Saved predictions to {preds_file}')


# Generate predictions on images and save to preds_file
# If n_items is -1, predict everything; otherwise only the first n_items
def make_predictions(images, test_metadata, preds_file,
                     classifier, n_trees, n_items=-1, seed=0):

    # Set classifier-specific parameters
    if classifier == 'RF':
        use_file = 'RF-%d-s%d.pkl' % (n_trees, seed)
    elif classifier == 'SVM':
        use_file = 'SVM-rbf.pkl'
    elif classifier == 'CNN':
        #n_epochs = 10
        #use_file = f'model-CNN-epochs-{n_epochs}.pt'
        #n_epochs = 20
        #use_file = f'model-CNN-5layers-epochs-30-e{n_epochs}.pt'
        n_epochs = 20
        use_file = f'model-CNN-VGG-epochs-30-e{n_epochs}.pt'
        device = 'mps'
    elif classifier == 'OCR':
        '''
        # Google's Tesseract system: Does not support handwritten text
        #import pytesseract

        # EasyOCR
        import easyocr
        reader = easyocr.Reader(['en'])
        '''

        # OCR.space
        import requests
        import tempfile
        '''

        # OpenAI
        from openai import OpenAI
        import base64
        client = OpenAI()
        '''
        
        use_file = None # Not needed
    else:
        print(f'Unknown classifier {classifier}')
        return

    # Load saved classifiers
    if classifier == 'OCR':
        pass
    elif os.path.exists(use_file):
        if classifier == 'CNN':
            clf = ImageClassifier().to(device=device)
            clf.load_state_dict(torch.load(use_file, map_location=device))
            clf.eval() # Set it to evaluation mode (disable dropout and batch norm)
        else: # sklearn models            
            with open(use_file, 'rb') as f:
                clf = pickle.load(f)
        print(f'Loaded previously trained classifier from {use_file}')
    else:
        # Train from scratch
        print('Training classifier')
        if classifier == 'RF':
            clf = RandomForestClassifier(n_estimators=n_trees,
                                         random_state=seed)
        elif classifier == 'SVM':
            #clf = SVC(kernel='linear') # C=1.0
            clf = SVC(kernel='rbf') # gamma='scale'
        elif classifier == 'CNN':
            clf = train_model(n_epochs=n_epochs, model_file=use_file, device=device)
        else:
            print(f"Don't know how to train {classifier}")
            return

        # for sklearn classifiers
        if classifier != 'CNN':
            x_train_mnist, y_train_mnist = load_mnist_train()

            clf.fit(x_train_mnist, y_train_mnist)
            print('Saving classifier')
            with open(use_file, 'wb') as f:
                pickle.dump(clf, f)
        else:
            with open(use_file, 'wb') as f: 
                torch.save(clf.state_dict(), f)
            print(f'Training complete! Model saved as {use_file}.')

    if n_items == -1:
        n_items = len(test_metadata)
    print(f'Classifying {n_items}/{len(test_metadata)} items')

    # Break out the metadata
    labels, _, nist_id = zip(*test_metadata[:n_items])

    preds = []
    i = 0
    for lab, ids in zip(labels, nist_id):
        # Assemble the multi-digit number as an array of digits from ids
        # for individual digit prediction
        # First remove any -1 placeholders
        ids = [id for id in ids if id != -1]

        if classifier == 'CNN':
            to_tensor_transform = transforms.ToTensor()
            test_data = [to_tensor_transform(images[id]) for id in ids]
            tdata = torch.stack(test_data)
            tdata = tdata.to(device)
            with torch.no_grad(): # Disable gradient calculation during evaluation
                outputs = clf(tdata)
            # Max returns values and indices
            _, pred_digits = torch.max(outputs.data, 1)
            pred_digits = pred_digits.to('cpu')
            pred_digits = pred_digits.tolist()
        elif classifier == 'OCR':
            test_data = np.array([images[id] for id in ids], dtype=np.uint8)
            im = create_multi_digit_image(test_data)
            '''
            # Google's Tesseract
            #pred = pytesseract.image_to_string(im)  # runs forever
            pred = pytesseract.image_to_string(im,
                                               config='-c tessedit_char_whitelist=0123456789')
            # Strip any newlines
            pred = pred.replace('\012','')
            if pred == '':
                # No text was recognized
                pred = 'None'
            '''

            '''
            # EasyOCR
            #pred = reader.readtext(im, detail=0)
            #pred = reader.readtext(im, detail=0, allowlist='0123456789')
            pred = reader.readtext(im, detail=0, allowlist='0123456789',
                                   decoder='beamsearch')
            #pred = reader.readtext(im, detail=0, allowlist='0123456789',
            #                       decoder='beamsearch', beamWidth=10)
            if pred == []:
                # No text was recognized
                pred = 'None'
            else:
                #pred = pred[0][1] # use if detail=1
                # pred may be a list; concatenate if so
                pred = ''.join(pred)
            '''

            # OCR.space
            # Package up the request
            payload = {'isOverlayRequired': False,
                       'apikey': 'K84683750988957',
                       'language': 'eng',
                       'filetype': 'PNG',
                       'scale': True,
                       'OCREngine': 2,
                       }

            # Create a temporary file to store the number image
            with tempfile.NamedTemporaryFile(suffix='.png',
                                             dir=os.getcwd()) as temp_file:
                pl.imsave(temp_file, im, cmap='gray')
    
                filename = temp_file.name
                try:
                    with open(filename, 'rb') as f:
                        r = requests.post('https://api.ocr.space/parse/image',
                                          files={filename: f},
                                          data=payload,
                                          )
    
                    #print(r.content.decode()['ParsedResults'][0]['ParsedText'])
                    res = r.json()
                    # Check for errors
                    if res['IsErroredOnProcessing']:
                        print(res['ErrorMessage'], res['ErrorDetails'])
                        pred = 'None'
                    else:
                        pred = res['ParsedResults'][0]['ParsedText']
                        # Strip any newlines
                        pred = pred.replace('\012','')
                        if pred == '':
                            # No text was recognized
                            pred = 'None'
                except:
                    pred = 'None'
            '''

            # OpenAI

            # Save the image to an in-memory buffer
            buffer = io.BytesIO()
            pl.imsave(buffer, im, format='png', cmap='gray') 
            buffer.seek(0) # Rewind the buffer to the beginning
            # Encode it as base64 and UTF-8
            image_binary_data = buffer.read()
            base64_bytes = base64.b64encode(image_binary_data)
            base64_string = base64_bytes.decode('utf-8') 

            # https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded#analyze-images
            response = client.responses.create(
                #model="gpt-4.1-nano",
                model="gpt-4.1",
                input=[
                    {"role": "user", "content": "Please extract the number in this image. Return only the number, no conversational text."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_string}",
                            }
                        ]
                    }
                ]
            )
            pred = response.output_text
            # Remove any internal spaces
            pred = pred.replace(' ','')
            '''
        else:
            # sklearn classifiers
            test_data = np.array([np.asarray(images[id],
                                             dtype=np.uint8).reshape(-1)
                                  for id in ids])
            pred_digits = clf.predict(test_data)

        if classifier != 'OCR':
            # Turn it into a string for comparison with the label
            pred = ''.join(map(str, pred_digits))
        #if pred != lab:
        #    print(lab, pred)
        #    input()
        preds.append([pred])
        i += 1
        if i % 100 == 0:
            print('.', flush=True, end='')
            # Save periodically
            save_predictions(preds_file, preds, verbose=False)

    print()
    save_predictions(preds_file, preds)


# Read in test items and make predictions
# using a digit classifier trained on the MNIST training set.
# Save predictions to preds_file.
def main(data_file, preds_file, classifier, n_trees, n_items=-1, seed=0):
    random.seed(seed)

    # Check arguments
    for filename in [data_file]:
        if not os.path.exists(filename):
            print(f'Error: could not find {filename}')
            sys.exit(1)

    # Read in test items
    test_metadata = read_MDW_data(data_file)

    # Read in the images from QMNIST
    images, _ = load_qmnist()

    # Generate the predictions and save to preds_file
    make_predictions(images, test_metadata, preds_file,
                     classifier, n_trees, n_items, seed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file', 
                        help='Input MDW CSV file with items stored as label, writer, NIST ids')
    parser.add_argument('preds_file', 
                        help='Output CSV file with predictions, one per line')
    parser.add_argument('-c', '--classifier', choices=['RF', 'SVM', 'CNN', 'OCR'],
                        default='SVM',
                        help='Classifier to use (default: %(default)s)')
    parser.add_argument('-t', '--n_trees', type=int, default=100,
                        help='Trees for the RF (default: %(default)s)')
    parser.add_argument('-n', '--n_items', type=int, default=-1,
                        help='How many items to test (-1 means all')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed (default: %(default)s)')

    args = parser.parse_args()
    main(**vars(args))


#!/usr/bin/env python3
# Kiri Wagstaff
# May 11, 2025
# Generate a multi-digit number by one writer and save it as an image.

import sys
import os
import torchvision.datasets as datasets
import random
import numpy as np
import pylab as pl
from create_MDW_data import load_writers, create_numeral_images


# Concatenate an array of NIST digit images into a single image
# and subtract from 255 to invert (so it is black on white)
def create_multi_digit_image(X):
    # Create a zip code image by concatenating digit images
    X_number = np.concatenate(X, axis=1)
    # Subtract from 255 to get black-on-white
    X_number = 255 - X_number

    return X_number


# Create a multi-digit number using only digits from this writer
# sampled from NIST test writers,
# and save the corresponding image to a file.
# If writer is -1, choose randomly from the writers_file.
# If seed is -1, don't seed the random number generator.
def main(number, writer, writers_file, seed=-1):

    # Load the QMNIST data set
    # compat=False means we want the extended labels with writer information
    qall = datasets.QMNIST(root='./data', what='nist',  compat=False,
                           download=True)
    _, lab_all = zip(*qall)

    if seed != -1:
        random.seed(seed)

    if writer == -1:
        # Pick a random test writer

        # Load in the train/test writer split
        wtrs_te = load_writers(writers_file)
        writer = random.choice(wtrs_te)
        print(f'Chose writer {writer}')

    # Get the digit images; pass number as a string for compatibility
    X, _ = create_numeral_images(str(number), writer, lab_all, qall)
    # Some few writers do not have all digits available
    if len(X) == 0:
        print(f'Could not generate {number} for writer {writer}; try another.')
        sys.exit(1)

    # Create a zip code image by concatenating digit images
    X_number = create_multi_digit_image(X)

    output_file = 'number-%s-w%d.png' % (number, writer)
    pl.imsave(output_file, X_number, cmap='gray')
    print(f'Saved {number} image to {output_file}')

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('number', type=int, help='Number to generate')
    parser.add_argument('writer', nargs='?', type=int, default=-1,
                        help='Writer id, -1 to choose randomly (default)')
    parser.add_argument('-w', '--writers_file',
                        default='resources/writers-all-test.csv',
                        help='CSV file: single column of writer ids (default: %(default)s)')
    parser.add_argument('-s', '--seed', type=int, default=-1,
                        help='Random seed (default: %(default)s, -1 means no seed)')


    args = parser.parse_args()
    main(**vars(args))


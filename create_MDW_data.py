#!/usr/bin/env python3
# Kiri Wagstaff
# May 11, 2025
# Construct multi-digit writer (MDW) test sets.

import sys
import os
import csv
import random
from torchvision import datasets
import numpy as np


# Load a list of writers from a .csv file
def load_writers(writers_file):

    wtrs = []

    with open(writers_file, 'r') as inf:
        csv_rdr = csv.reader(inf)
        # Consume the header
        hdr = next(csv_rdr)
        for row in csv_rdr:
            wtrs += [int(row[0])]

    return wtrs


# Create a numeral (series of digit images) by the specified writer
# by randomly choosing a matching image by that writer
# for each digit in the numeral (string).
# Get the image data itself from data.
# Return a numpy array of the images.
def create_numeral_images(numeral, writer, labels, data, verbose=False):

    # Labels for all of this writer's digits
    writer_labels = [lab for lab in labels if lab[2] == writer]
    if writer_labels == []:
        # No data from this writer
        if verbose:
            print(f'Error: No data for writer {writer}')
        return [], []

    if verbose:
        print(f'Writer {writer} has {len(writer_labels)} digits, making {numeral}')

    X = []
    nist_id = []

    # Pick an image by this writer for each digit
    for c in range(len(numeral)):
        digit = int(numeral[c])
        lab_wtr_digit = [lab for lab in writer_labels if lab[0] == digit]
        if lab_wtr_digit == []:
            # None of this digit from this writer
            if verbose:
                print(f'Error: No digit {digit} for writer {writer}')
            return [], []

        # Pick one
        lab_im = random.choice(lab_wtr_digit).numpy()

        # Get its image data, matching on NIST id
        X += [data[lab_im[5]][0]]
        nist_id += [lab_im[5]]

    # Label includes:
    # - numeral to predict: as a string with N digits
    # - NIST writer id - 0 to 2599
    # - N global NIST ids - 0 to 281769
    numeral_label = [numeral, writer, nist_id]
    X = np.array(X)

    return X, numeral_label


# Create a data test set of n_samples of the specified type,
# using the random seed, and save them to output_file
def main(domain, writers_file, n_samples, seed, output_file,
         zip_code_file):

    # Load the QMNIST data set
    # compat=False means we want the extended labels with writer information
    qall = datasets.QMNIST(root='./data', what='nist',  compat=False,
                           download=True)
    _, lab_all = zip(*qall)

    # Load in the train/test writer split
    wtrs_te = load_writers(writers_file)
    print(f'Read {len(wtrs_te)} writers from {writers_file}')

    random.seed(seed)

    # Generate and store labels for the generated numerals

    if domain == 'zip_code':
        # Generate using valid zip codes from USPS list
        # https://postalpro.usps.com/ZIP_Locale_Detail
        valid_zip_codes = []
        with open(zip_code_file, 'r') as inf:
            csv_rdr = csv.reader(inf)
            # Consume the header
            hdr = next(csv_rdr)
            for row in csv_rdr:
                valid_zip_codes.append(row[0])
        print(f'Read {len(valid_zip_codes)} zip codes from U.S. list.')

        # Sample with replacement: allow different writers, same zip code
        numerals = [str(z) for z in
                    random.choices(valid_zip_codes, k=n_samples)]
        n_digits = 5

    elif domain == 'check_amount':
        # Generate using list of amounts following Benford's law
        # with 1 to 5 digits before, and 2 digits after, the decimal
        # randalyze generate benford -c 10000 -w 5 -d 2 > check_amounts_benford.csv
        check_amounts = []
        with open(os.path.join('resources', 'check_amounts_benford.csv'), 'r') as inf:
            csv_rdr = csv.reader(inf)
            # Consume the header
            hdr = next(csv_rdr)
            for row in csv_rdr:
                check_amounts += [float(row[0])]

        # Ensure we have 3 to 7 digits by
        # multiplying by 100 to shift left 2 digits for fixed point,
        # and convert to string
        # Ensure 2 digits after the decimal
        numerals = ['%.2f' % c for c in
                    random.choices(check_amounts, k=n_samples)]
        # Remove the decimal
        numerals = [n[:len(n)-3] + n[-2:] for n in numerals]
        for n in numerals:
            if len(n) < 3 or len(n) > 7:
                print(n)

        n_digits = 7 # up to 5 before and 2 after the decimal place

    elif domain == 'clock_time':
        # Generate number of minutes from 0 to 1439
        clock_times = random.choices(range(0, 1439), k=n_samples)

        # Convert to HMM or HHMM
        numerals = ['%d%02d' % (int(m / 60), int(m % 60))
                    for m in clock_times]
        n_digits = 4 # 3 or 4 depending on hours

    else:

        print(f'Unsupported domain {domain}.')
        return

    labels = []

    for n, num in enumerate(numerals):

        # Keep trying writers until one has all the digits needed
        # (usually works first try)
        while True:
            # Pick a random test writer
            writer = random.choice(wtrs_te)

            X, lab = create_numeral_images(num, writer, lab_all, qall)
            # Some few writers do not have all digits available
            if len(X) > 0: # success!
                break

        if domain in ['check_amount', 'clock_time']:
            # Pad these out to have all the same number of digits
            # Prepend with -1 as the NIST id for unused digits
            n_digits_used = len(lab[0])
            n_pad = n_digits - n_digits_used
            lab[2] = [-1] * n_pad + lab[2]

        labels += [lab]

        if (n + 1) % 100 == 0:
            print('.', end='', flush=True)
    print()

    # Save the numeral data and labels
    (numerals, writers, nist_id) = zip(*labels)
    if output_file.endswith('.npz'):
        np.savez(output_file, labels=numerals, writers=writers,
                 nist_id=nist_id)
    else:
        # Default to assuming csv
        # Assemble data rows
        data = [[z, w] for z, w in zip(numerals, writers)]
        # Split nist_id list into individual ids and append
        # d.extend(n) updates the items in the data array
        d2 = [d.extend(n) for (d, n) in zip(data, nist_id)]
        with open(output_file, 'w') as outf:
            csvout = csv.writer(outf)
            # Write the header
            csvout.writerow(['Numeral', 'Writer'] +
                            ['NIST id %d' % i for i in range(n_digits)])
            csvout.writerows(data)

    print(f'Saved {len(numerals)} items to {output_file}')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--domain', default='zip_code',
                        choices=['zip_code', 'check_amount', 'clock_time'],
                        help='Type of data to generate')
    parser.add_argument('-n', '--n_samples', type=int, default=100,
                        help='How many numerals to create (default: %(default)s)')
    parser.add_argument('-w', '--writers_file',
                        default='resources/writers-all-test.csv',
                        help='CSV file: single column of writer ids to sample from (default: %(default)s)')
    parser.add_argument('-f', '--output_file', default='dataset.csv',
                        help='Can be a .npz or .csv file (default: %(default)s)')
    parser.add_argument('-z', '--zip_code_file',
                        default='resources/us_zip_codes_unique.csv',
                        help='CSV file: single column with list of valid U.S. zip codes (default: %(default)s)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed (default: %(default)s)')


    args = parser.parse_args()
    main(**vars(args))

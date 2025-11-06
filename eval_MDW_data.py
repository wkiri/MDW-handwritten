#!/usr/bin/env python3
# Kiri Wagstaff
# May 13, 2025
# Evaluate predictions on MDW test sets.

import sys
import os
import csv
import random
import numpy as np


# Read in test items (represented as metadata)
# from data_file and return the meta_data
def read_MDW_data(data_file):

    test_metadata = []

    with open(data_file, 'r') as inf:
        csv_rdr = csv.reader(inf)
        # Consume the header
        hdr = next(csv_rdr)
        for row in csv_rdr:
            # label, writer id, 1 or more NIST ids
            test_metadata.append([row[0], row[1], [int(i) for i in row[2:]]])

    return test_metadata


# Read in the predictions from a CSV file in which each row is:
# prediction (as a string), one per line.
# Return predictions
def read_predictions(preds_file):

    preds = []
    
    with open(preds_file, 'r') as inf:
        csv_rdr = csv.reader(inf)
        # Consume the header
        hdr = next(csv_rdr)
        for row in csv_rdr:
            # prediction (string)
            # Invert order for logic
            preds.append(row[0])

    return preds


def match_labels_preds(test_metadata, preds):

    # Break out the metadata and match labels to preds
    labels, _, _ = zip(*test_metadata)

    # Assume they match one to one; check length for sanity
    if len(labels) != len(preds):
        print('Error: labels and predictions are not the same size.')
        sys.exit(1)

    labels_preds = list(zip(labels, preds))

    return labels_preds


def match_labels_writers_preds(test_metadata, preds):

    # Break out the metadata and match labels to preds
    labels, writers, _ = zip(*test_metadata)

    # Assume they match one to one; check length for sanity
    if len(labels) != len(preds):
        print('Error: labels and predictions are not the same size.')
        sys.exit(1)

    labels_writers_preds = list(zip(labels, writers, preds))

    return labels_writers_preds


# Calculate error metrics for ZIP Code predictions in preds_file
# for items in data_file.
# Use zip_code_file to define the set of valid zip codes.
def eval_zip_codes(test_metadata, preds, zip_code_file):

    n_pred = len(preds)
    
    labels_preds = match_labels_preds(test_metadata, preds)

    # Find valid predictions
    valid_zip_codes = []
    with open(zip_code_file, 'r') as inf:
        csv_rdr = csv.reader(inf)
        # Consume the header
        hdr = next(csv_rdr)
        for row in csv_rdr:
            valid_zip_codes.append(row[0])

    valid = [(label, pred) for (label, pred) in labels_preds
             if pred in valid_zip_codes]
    n_valid = len(valid)
    
    # Report fraction of invalid predictions
    frac_invalid = (n_pred - n_valid) / n_pred
    print('Invalid ZIP Codes: %d/%d = %.4f' % (n_pred - n_valid,
                                                 n_pred, frac_invalid))

    # Report fraction of valid but wrong predictions (strict)
    n_valid_wrong = len([label for label, pred in valid
                         if label != pred])
    frac_valid_wrong = n_valid_wrong / n_pred
    print('Valid but wrong ZIP Codes: %d/%d = %.4f' % (n_valid_wrong,
                                                       n_pred,
                                                       frac_valid_wrong))

    # Report total number of errors
    n_wrong = (n_pred - n_valid) + n_valid_wrong
    frac_wrong = n_wrong / n_pred
    print('Total wrong ZIP Codes: %d/%d = %.4f' % (n_wrong, n_pred,
                                                   frac_wrong))

    # Break down census and high school errors
    print()
    # Get the matches again, with writers
    labels_writers_preds = match_labels_writers_preds(test_metadata, preds)
    census = [(label, pred) for (label, w, pred) in labels_writers_preds
              if int(w) < 2000]
    n_census = len(census)
    n_census_wrong = len([label for label, pred in census
                          if label != pred])
    frac_census_wrong = n_census_wrong / n_census
    print(f'Census employees error rate: {n_census_wrong}/{n_census} = {frac_census_wrong:.4f}')    

    high_school = [(label, pred) for (label, w, pred) in labels_writers_preds
              if int(w) >= 2000]
    n_high_school = len(high_school)
    n_high_school_wrong = len([label for label, pred in high_school
                          if label != pred])
    frac_high_school_wrong = n_high_school_wrong / n_high_school
    print(f'High school error rate: {n_high_school_wrong}/{n_high_school} = {frac_high_school_wrong:.4f}')    
    
    return {'Frac_invalid': frac_invalid,
            'Frac_valid_wrong': frac_valid_wrong,
            'Frac_wrong': frac_wrong}


# ------ Check amount metrics ------- #

# Calculate error metrics for check amount predictions in preds_file
# for items in data_file.
def eval_check_amounts(test_metadata, preds):

    n_pred = len(preds)
    
    labels_preds = match_labels_preds(test_metadata, preds)
    #n_pred = len(list(labels_preds))
    #print(n_pred)

    # The only invalid values are ones with leading zeros
    # and more than 3 digits - $0.XX is ok but $0X.XX is not.
    valid = [(label, pred) for (label, pred) in labels_preds
             if (len(pred) == 3 or pred[0] != '0') and pred != 'None']
    n_valid = len(valid)

    # Report fraction of invalid predictions
    frac_invalid = (n_pred - n_valid) / n_pred
    print('Invalid check amounts: %d/%d = %.4f' % (n_pred - n_valid,
                                                 n_pred, frac_invalid))

    # Report fraction of valid but wrong predictions (strict)
    n_valid_wrong = len([label for label, pred in valid
                         if label != pred])
    frac_valid_wrong = n_valid_wrong / n_pred
    print('Valid but wrong check amounts: %d/%d = %.4f' % (n_valid_wrong,
                                                         n_pred,
                                                         frac_valid_wrong))
    # Report total number of errors
    n_wrong = (n_pred - n_valid) + n_valid_wrong
    frac_wrong = n_wrong / n_pred
    print('Total wrong check amounts: %d/%d = %.4f' % (n_wrong, n_pred,
                                                       frac_wrong))

    # ----- Numeric metrics ---

    # Convert valid predictions to dollars for numeric comparison
    valid_dollars = [[int(label)/100.0, int(pred)/100.0]
                     for label, pred in valid]

    # Report total error in dollars (for valid predictions)
    err = [pred_dollar - true_dollar
           for true_dollar, pred_dollar in valid_dollars]
    abserr = [np.abs(pred_dollar - true_dollar)
           for true_dollar, pred_dollar in valid_dollars]
    total_err = np.sum(abserr)
    print('Valid amounts: total error in dollars: %.2f' % total_err)

    # Report average error in dollars (for valid predictions)
    mean_err = np.mean(err)
    print('Valid amounts: average error in dollars: %.2f' % mean_err)

    # Report max error in dollars (for valid predictions)
    max_err = np.max(abserr)
    print('Valid amounts: maximum error in dollars: %.2f' % max_err)

    # ----- Check for subgroup bias ---
    print()
    # Break down by amount
    for (sml, lrg) in [(0, 999), # no values less than $1.00
                       (1000, 9999),
                       (10000, 99999),
                       (100000, 999999),
                       (1000000, 9999999)]:
        bin_amts = [(label, pred) for (label, pred) in labels_preds
                    if (int(label) >= sml and int(label) <= lrg)]
        n_wrong = len([label for label, pred in bin_amts
                       if label != pred])
        err_rate = n_wrong * 1.0 / len(bin_amts)
        print(f'{sml}-{lrg}: {n_wrong} / {len(bin_amts)} = {err_rate:.4f}')

    print()
    # Break down by digit position
    for pos in range(7):
        bin_amts = [(label, pred) for (label, pred) in labels_preds
                    if (len(label) > pos)] # has at least pos+1 digits
        # Is pos digit wrong?
        n_wrong = len([label for label, pred in bin_amts
                       if label[pos] != pred[pos]])
        err_rate = n_wrong * 1.0 / len(bin_amts)
        print(f'Position {pos}: {n_wrong} / {len(bin_amts)} = {err_rate:.4f}')

    print()
    # Break down by leading digit
    for dig in range(10):  # 0 through 9
        bin_amts = [(label, pred) for (label, pred) in labels_preds
                    if int(label[0]) == dig]
        if len(bin_amts) == 0:
            continue
        # Is first digit wrong?
        wrong = [[label, pred] for label, pred in bin_amts
                 if label[0] != pred[0]]
        n_wrong = len(wrong)
        err_rate = n_wrong * 1.0 / len(bin_amts)
        print(f'Lead digit {dig}: {n_wrong} / {len(bin_amts)} = {err_rate:.4f}')
        bin_dollars = [[int(label)/100.0, int(pred)/100.0]
                       for label, pred in wrong]
        err_amt = [pred_dollar - true_dollar
                   for true_dollar, pred_dollar in bin_dollars]
        if len(wrong) > 0:
            print(f'  Avg error in dollars: {sum(err_amt)/len(err_amt):.2f}')
            print(f'  Avg error magnitude in dollars: {sum(np.abs(err_amt))/len(err_amt):.2f}')
        

    return {'Frac_invalid': frac_invalid,
            'Frac_valid_wrong': frac_valid_wrong,
            'Frac_wrong': frac_wrong,
            'Total_error': total_err,
            'Mean_error': mean_err,
            'Max error': max_err}


# ------ Clock time metrics ------- #

# Check validity of clock_time (string) as HMM or HHMM.
# Valid times are 0:00 to 23:59.
def valid_clock_time(clock_time):

    if clock_time == 'None' or len(clock_time) < 3:
        return False

    try:
        int(clock_time)
    except:
        return False

    hours = int(clock_time[:len(clock_time) - 2])
    minutes = int(clock_time[-2:])

    return (0 <= hours and hours <= 23 and
            0 <= minutes and minutes <= 59)


# Convert label or predicted clock time from string to minutes
def to_min(clock_time):

    hours = int(clock_time[:len(clock_time) - 2])
    minutes = int(clock_time[-2:])

    return hours * 60 + minutes

    
# Calculate error metrics for clock time predictions in preds_file
# for items in data_file.  
def eval_clock_times(test_metadata, preds):

    n_pred = len(preds)
    
    labels_preds = match_labels_preds(test_metadata, preds)

    # Find valid predictions
    valid = [(label, pred) for (label, pred) in labels_preds
             if valid_clock_time(pred)]
    n_valid = len(valid)
    
    # Report fraction of invalid predictions
    frac_invalid = (n_pred - n_valid) / n_pred
    print('Invalid clock times: %d/%d = %.4f' % (n_pred - n_valid,
                                                 n_pred, frac_invalid))

    # Report fraction of valid but wrong predictions (strict)
    n_valid_wrong = len([label for label, pred in valid
                         if label != pred])
    frac_valid_wrong = n_valid_wrong / n_pred
    print('Valid but wrong clock times: %d/%d = %.4f' % (n_valid_wrong,
                                                         n_pred,
                                                         frac_valid_wrong))

    # Report total number of errors
    n_wrong = n_pred - n_valid + n_valid_wrong
    frac_wrong = n_wrong / n_pred
    print('Total wrong clock times: %d/%d = %.4f' % (n_wrong, n_pred,
                                                     frac_wrong))    

    # ----- Numeric metrics ---

    # Convert valid predictions to minutes for numeric comparison
    valid_mins = [[to_min(label), to_min(pred)] for label, pred in valid]

    # Report total error in minutes (for valid predictions)
    err = [pred_min - true_min for true_min, pred_min in valid_mins]
    abserr = [np.abs(pred_min - true_min) for true_min, pred_min in valid_mins]
    total_err = np.sum(abserr)
    print('Valid times: total error in minutes: %.2f' % total_err)

    # Report average error in minutes (for valid predictions)
    mean_err = np.mean(err)
    print('Valid times: average error in minutes: %.2f' % mean_err)

    # Report max error in minutes (for valid predictions)
    max_err = np.max(abserr)
    print('Valid times: maximum error in minutes: %.2f' % max_err)

    return {'Frac_invalid': frac_invalid,
            'Frac_valid_wrong': frac_valid_wrong,
            'Frac_wrong': frac_wrong,
            'Total_error': total_err,
            'Mean_error': mean_err,
            'Max error': max_err}
    

# Read in test items and predictions
# and compute error metrics relevant to domain.
def main(data_file, preds_file, domain, zip_code_file):

    # Check arguments
    for filename in [data_file, preds_file]:
        if not os.path.exists(filename):
            print(f'Error: could not find {filename}')
            sys.exit(1)

    # Read in test items
    test_metadata = read_MDW_data(data_file)

    # Read in the predictions
    preds = read_predictions(preds_file)

    # Evaluate the predictions in domain-specific ways
    if domain == 'zip_code':
        eval_zip_codes(test_metadata, preds, zip_code_file)

    elif domain == 'check_amount':
        eval_check_amounts(test_metadata, preds)

    elif domain == 'clock_time':
        eval_clock_times(test_metadata, preds)

    else:
        print(f'Unsupported domain {domain}.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file', 
                        help='Input MDW CSV file with items stored as label, writer, NIST ids')
    parser.add_argument('preds_file', 
                        help='CSV file with predictions, one per line')
    parser.add_argument('-d', '--domain', default='zip_code',
                        choices=['zip_code', 'check_amount', 'clock_time'],
                        help='Type of data to generate'),
    parser.add_argument('-z', '--zip_code_file',
                        default='resources/us_zip_codes_unique.csv',
                        help='CSV file: single column with list of valid U.S. zip codes (default: %(default)s)')

    args = parser.parse_args()
    main(**vars(args))


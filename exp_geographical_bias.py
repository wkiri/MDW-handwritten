#!/usr/bin/env python3
# Kiri Wagstaff
# May 14, 2025
# Evaluate a classifier on its ZIP Code recognition predictions and
# plot error geographically (choropleth) to check for bias.

import sys
import os
import csv
import numpy as np
import pylab as pl
import geopandas as gpd
from eval_MDW_data import read_MDW_data, read_predictions, match_labels_preds


# Evaluate classifier on ZIP Codes predictions and plot spatially.
def main(data_file, preds_file, zip_code_file):

    # Check for geographical bias

    # Read in test items
    test_metadata = read_MDW_data(data_file)
    # Read in the predictions
    preds = read_predictions(preds_file)
    print(f'Read {len(preds)} predictions from {preds_file}.')

    # Match them up
    labels_preds = list(match_labels_preds(test_metadata, preds))

    # Load the list of valid zip codes
    valid_zip_codes = []
    with open(zip_code_file, 'r') as inf:
        csv_rdr = csv.reader(inf)
        # Consume the header
        hdr = next(csv_rdr)
        for row in csv_rdr:
            valid_zip_codes.append(row[0])
    print(f'Read {len(valid_zip_codes)} zip codes from U.S. list.')

    err_rate = []
    cname = preds_file.split('-')[-1].split('.')[0]
    print(f'Classifier: {cname}')
    # Error rates based on zip codes in the MDW test file
    errfile = f'zip-code-errors-MDW-sector-{cname}.npz'
    if os.path.exists(errfile):
        err_rate = np.load(errfile)['err_rate']
        print(f'Loaded previous results from {errfile}')
    else:
        for d in range(10): # 0 through 9
            zip_codes = [(zc, pred) for (zc, pred) in labels_preds
                         if int(zc[0]) == d]
            print(f'Zone {d}: {len(zip_codes)} zip codes')
            sectors = list(set([zc[:2] for (zc, pred) in zip_codes]))
            print(f'Zone {d}: {len(sectors)} sectors')

            for s in sectors:
                sector_zip_codes = [(zc, pred) for (zc, pred) in zip_codes
                                    if zc[:2] == s]
                # Calculate error rate
                errors = [(zc, pred) for (zc, pred) in sector_zip_codes
                          if zc != pred]
                err_rate += [[s, len(errors) * 1.0 / len(sector_zip_codes)]]

                np.savez(errfile, err_rate=err_rate)
        print(f'Saved results to {errfile}')

    # Plot the results
    sector_json = os.path.join('resources', 'zcta5.sector.geo.json')
    if not os.path.exists(sector_json):
        zcta5_json = os.path.join('resources', 'zcta5.geo.json')
        if not os.path.exists(zcta5_json):
            print(f'Could not find {zcta5_json} shapefile.')
            print('Please download from https://github.com/jgoodall/us-maps/blob/master/geojson/zcta5.geo.json')
            sys.exit(1)
        print(f'Generating sectors from {zcta5_json}')
        gdf = gpd.read_file(zcta5_json)
        gdf['Sector'] = [z[:2] for z in gdf['ZCTA5CE10']]
        valid = gdf.loc[gdf.geometry.is_valid]
        sector = valid.dissolve(by='Sector')
        # Ideally save this out so we don't re-do it each time
        sector.to_file(sector_json, driver="GeoJSON")
    else:
        print(f'Loading sectors from {sector_json}')
        sector = gpd.read_file(sector_json)

    #err_rate = np.load('zip-code-errors-all-sector-100trials.npz')['err_rate']

    # you can't assign by loc, only add a column
    # so we have to sort and hope they line up (spot check passes)
    err_rate = np.array(err_rate)
    err_sorted = err_rate[np.argsort(err_rate[:,0])].astype(float)
    sector['Err'] = err_sorted[:,1]
    #print(err_sorted)
    minsec = np.argmin(err_sorted[:,1])
    maxsec = np.argmax(err_sorted[:,1])
    print(f'Sector errors range from {err_sorted[minsec,1]:.4f} (sector {err_sorted[minsec,0]:02.0f}XXX) to {err_sorted[maxsec,1]:.4f} (sector {err_sorted[maxsec,0]:02.0f}XXX)')

    #sector.plot(column = 'Err', legend=True, vmin=0.00, vmax=0.16)
    sector.plot(column = 'Err', legend=True, vmin=0.00, vmax=0.12)

    pl.gcf().set_size_inches(5, 2.5, forward=True)
    pl.xlim((-130, -65))
    pl.ylim((25, 50))
    pl.tight_layout()

    # Remove axis ticks
    ax = pl.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    pl.xticks([])
    pl.yticks([])

    # Make the legend match height of map
    ax = pl.gcf().axes
    map_box = ax[0].get_position()
    leg_box = ax[1].get_position()
    ax[1].set_position([leg_box.x0, map_box.y0, leg_box.width, map_box.height])

    #figfile = 'res-sector-err-%s-3.pdf' % cname
    # Using data from the MDW test file
    figfile = f'res-MDW-sector-err-{cname}.pdf'
    pl.savefig(figfile)
    print(f'Saved choropleth to {figfile}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file',
                        help='Input MDW CSV file with items stored as label, writer, NIST ids')
    parser.add_argument('preds_file',
                        help='Output CSV file with predictions, one per line')
    parser.add_argument('-z', '--zip_code_file',
                        default='resources/us_zip_codes_unique.csv',
                        help='CSV file: single column with list of valid U.S. zip codes (default: %(default)s)')

    args = parser.parse_args()
    main(**vars(args))

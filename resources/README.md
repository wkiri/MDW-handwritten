# Resources for Multi-Digit Writer (MDW) Data Sets

Author: Kiri Wagstaff, wkiri@wkiri.com

## Writer Lists

To enable the creation of multi-digit data sets using the QMNIST data,
and ensure that writers can be used separately for training and test
data, we established standard lists of writers to be used in each
setting, Wtr and Wte.  For full compatibility with MNIST, we
initialized Wtr (training) and Wte (test) with the 539 and 535
(respectively) writers employed in the MNIST training and test sets,
then randomly split the remaining (previously unused) 2,505 writers
between Wtr and Wte.

Most scripts in this repository default to sampling from the full test
writer set, but you can specify which list to use when generating data
sets. 

* `writers-mnist-train.csv`: 539 writer ids used for MNIST
* `writers-mnist-test.csv`: 535 writer ids used for MNIST
* `writers-all-train.csv`: 1791 writer ids
* `writers-all-test.csv`: 1788 writer ids

## U.S. ZIP Codes

`us_zip_codes_unique.csv`

This file contains a list of unique U.S. ZIP Codes, obtained from
[the USPS](https://postalpro.usps.com/ZIP_Locale_Detail)
(version from May 2, 2025).  These are unique values from the
"DELIVERY ZIPCODE" column.

`zcta5.sector.geo.json`

This file was created by "dissolving" the full-resolution ZCTA5
polygons defining all U.S. ZIP Codes into larger sectors that group
ZIP Codes by their first two digits.  The original source of the
full-resolution polygons (thanks to John Goodall) is at:

[https://github.com/jgoodall/us-maps/blob/master/geojson/zcta5.geo.json](https://github.com/jgoodall/us-maps/blob/master/geojson/zcta5.geo.json)

## Handwritten Check Amounts

`check_amounts_benford.csv`

This file contains a list of 10,000 numbers to simulate handwritten
check amounts.  They are sampled from the range $0.00 to $99999.99
according to Benford's Law. We used the
[randalyze](https://pypi.org/project/randalyze/) package as follows: 

```
randalyze generate benford -c 10000 -w 5 -d 2 > check_amounts_benford.csv
```


<!--
 LocalWords:  MDW QMNIST Wtr Wte MNIST mnist csv ZIPCODE ZCTA benford
 LocalWords:  randalyze
 -->

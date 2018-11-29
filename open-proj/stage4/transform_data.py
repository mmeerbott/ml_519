#!/usr/bin/env python3.6
import argparse
import csv
from pandas import read_csv


def append_out(filename):
    """ Helper function. Appends '_out' (filename.ext to filename_out.ext)
        
    filename : str
    """
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + ['out'])


def years_in_header(filename):
    """ Transforms the years (+data) into individual rows
        Requires the format to be:
            state,[years,...]

        Does not rename the header field for data as it is never provided,
        the user must do this
    """
    csvout = append_out(filename)
    with open(filename, 'r') as infile, open(csvout, 'w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        oldheader = next(reader)  # first line is the header
        writer.writerow(['state', 'year'])  # write new header

        for oldrow in reader:
            # get every year, and just turn it into a row with the same state
            for i, year in enumerate(oldheader[1:]):
                newrow = [oldrow[0]]  # state
                newrow.extend([year, oldrow[i+1]])  # year and associated data
                writer.writerow(newrow)
                newrow = []  # reset the row we wrote

    return csvout


def average_no_years():
    """ Assigns years to all data without years by duplicating the data.
        Assumes the data given is the average
    """
    pass


def no_states():
    """ Assigns states to the data by duplicating data.
        Assumes that the data is equal through the nation.
    """
    pass

if __name__=="__main__":
    #load base csv file 
    parser = argparse.ArgumentParser(
        description='Reshape a file to have states and years in rows'
    )
    parser.add_argument('filename',
	help="CSV file to look at"
    )

    # TODO add options to reshape specific formats? 
    # might be better to just run by hand through console

    # TODO remove this as it's a shortcut for testing
    print(years_in_header('datasets/average_income.csv'))

#!/usr/bin/env python3.6
import argparse
from pandas import read_csv


def append_out(filename):
    """ Helper function. Appends '_out' (filename.ext to filename_out.ext)
        
    filename : str
    """
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + ['out'])


def years_in_header():
    """ Transforms the years (+data) into individual rows
    """

    pass


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

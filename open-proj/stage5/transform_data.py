#!/usr/bin/env python3.6
import argparse
import csv
import pandas as pd

"""
    Functions to rearrange the data into our desire format of:
    year, state, [rest of the data...]
"""


def append_out(filename):
    """ Helper function. Appends '.out' (filename.ext to filename.out.ext)
        
    filename : str
    """
    return "{0}.{2}.{1}".format(*filename.rsplit('.', 1) + ['out'])


def move_years_in_header(filenames):
    """ Transforms the years (+data) into individual rows
        Requires header format to be:
            state,[years,...]

        Transforms into:
            year, state

        Does not rename the header field (as it is never provided),
        the user must do this
    """
    for filename in filenames:
        csvout = append_out(filename)
        with open(filename, 'r') as infile, open(csvout, 'w') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            oldheader = next(reader)  # first line is the header
            writer.writerow(['state', 'year'])  # write new header

            for oldrow in reader:
                # get each year, and turn it into a row with the same state
                for i, year in enumerate(oldheader[1:]):
                    newrow = [oldrow[0]]  # state
                    newrow.extend([year, oldrow[i+1]])  # year and data
                    writer.writerow(newrow)
                    newrow = []  # reset the row written


def add_years(filenames):
    """ Assigns years to all data without years by duplicating the data.
        Assumes the data given is the average
        Years range from [1992, 2016]
        Expects State, but no year
    """
    for filename in filenames:
        csvout = append_out(filename)
        with open(filename, 'r') as infile, open(csvout, 'w') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            oldheader = next(reader)  # first line is the header
            newheader = ['year']
            newheader.extend(oldheader)
            writer.writerow(newheader)  # write new header
            years = range(1992, 2016)

            for oldrow in reader:
                for year in years:
                    newrow = [year]
                    newrow.extend(oldrow)
                    writer.writerow(newrow)
                    newrow = []  # reset the row written


def add_states(filenames):
    """ Assigns states to the data by duplicating data.
        Assumes that the data is equal through the nation.
        Expects years
    """
    for filename in filenames:
        csvout = append_out(filename)
        with open(filename, 'r') as infile, open(csvout, 'w') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            oldheader = next(reader)  # first line is the header
            newheader = ['state']
            newheader.extend(oldheader)
            writer.writerow(newheader)  # write new header
            states = ['Alabama','Alaska','Arizona','Arkansas','California',
                     'Colorado','Connecticut','Delaware','District of Columbia',
                     'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana',
                     'Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland',
                     'Massachusetts','Michigan','Minnesota','Mississippi',
                     'Missouri','Montana','Nebraska','Nevada','New Hampshire',
                     'New Jersey','New Mexico','New York','North Carolina',
                     'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania',
                     'Rhode Island','South Carolina','South Dakota','Texas',
                     'Utah','Vermont','Virginia','Washington','West Virginia',
                     'Wisconsin','Wyoming']

            for oldrow in reader:
                for state in states:
                    newrow = [state]
                    newrow.extend(oldrow)
                    writer.writerow(newrow)
                    newrow = []  # reset the row written


def join(filenames):
    """ Joins multiple files on the columns 'year' and 'state'
    """
    # base file, begin merging onto it
    left_df = pd.read_csv(filenames[0])

    for filename in filenames[1:]:
        right_df = pd.read_csv(filename)
        left_df  = pd.merge(left_df, right_df, how='left', 
                            left_on=['year', 'state'], 
                            right_on=['year', 'state']
                           )
        
    left_df.to_csv('combined_data.out.csv')


if __name__=="__main__":
    #load base csv file 
    parser = argparse.ArgumentParser(
        description='Reshape a file to have states and years in rows'
    )
    parser.add_argument('method',
        choices=['add_years', 'add_states', 'move_years', 'merge'],
	help='Method to use to preprocess the data file(s)'
    )
    parser.add_argument('files', nargs='+',
	help='CSV files to process'
    )
    args = parser.parse_args()

    if args.method == 'add_years':
        add_years(args.files)
    elif args.method == 'add_states':
        add_states(args.files)
    elif args.method == 'move_years':
        move_years_in_header(args.files)
    elif args.method == 'merge':
        print(args.files)
        join(args.files)

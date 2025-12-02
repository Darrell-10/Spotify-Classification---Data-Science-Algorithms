"""
Programmer: Jon Larson
Class: Cpsc 322-01, Fall 2025
Programming Assignment #2-6
10/7/25
Description: This file contains
the class and implementation that 
is used for table manipulation and
statistics.
"""

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return (len(self.data), len(self.column_names))

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid

        """
        col_index = -1

        for i in range(len(self.column_names)):
            if i == col_identifier:
                col_index = i
                break

        if col_index == -1:
            for i in range(len(self.column_names)):
                if self.column_names[i] == col_identifier:
                    col_index = i
                    break
        
        if col_index == -1:
            raise ValueError("Invalid col_identifier.")
        
        column = []
        for row in self.data:
            val = row[col_index]
            if include_missing_values or val != "NA":
                column.append(val)

        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i in range(len(self.data)): #row loop
            for j in range(len(self.data[i])): #column loop
                value = self.data[i][j]
                if value != "NA":
                    try:
                        self.data[i][j] = float(value)
                    except ValueError:
                        pass #don't change values

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        rows = []
        for i in range(len(self.data)):
            keep = True
            for j in range(len(row_indexes_to_drop)):
                if i == row_indexes_to_drop[j]:
                    keep = False
                    break
            if keep:
                rows.append(self.data[i])
        self.data = rows
        pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = True
            self.column_names = []
            self.data = []

            for row in reader:
                if header:
                    self.column_names = row
                    header = False
                else:
                    self.data.append(row)
        
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            #write header
            writer.writerow(self.column_names)
            #wrtie rows
            for i in range(len(self.data)):
                writer.writerow(self.data[i])
        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        duplicates = [] #holds duplicate row indexes
        seen = [] #holds key-value pairs for the rows we have seen before
        key_index = []
        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if self.column_names[j] == key_column_names[i]:
                    key_index.append(j)
                    break
        #loop over rows to find duplicates
        for row in range(len(self.data)):
            key_vals = [] #key-value pair list
            for k in range(len(key_index)):
                key_vals.append(self.data[row][key_index[k]])
            #check if seen and if so add to list
            have_seen = False
            for l in range(len(seen)):
                same = True
                for m in range(len(key_vals)):
                    if key_vals[m] != seen[l][m]:
                        same = False
                        break
                if same:
                    have_seen = True
                    break
            if have_seen:
                duplicates.append(row) #have seen so add to list
            else:
                seen.append(key_vals) #first time seen so add to list

        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        rows_to_keep = [] #stores the rows we will not remove
        #loop for each row
        for i in range(len(self.data)):
            row_has_missing_val = False #set flag for if a row has an "NA" value
            #loop for each value in row
            for j in range(len(self.data[i])):
                if self.data[i][j] == "NA": #check if cell has missing value
                    row_has_missing_val = True
                    break
            #keep rows with no missing values
            if not row_has_missing_val:
                rows_to_keep.append(self.data[i])
        #update the table
        self.data = rows_to_keep

        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        column_index = -1
        for i in range(len(self.column_names)):
            if self.column_names[i] == col_name:
                column_index = i
                break
        if column_index == -1:
            return #exit function if the col_name is not found
        
        #calculate avg for missing values
        total = 0
        count = 0
        for i in range(len(self.data)):
            val = self.data[i][column_index]
            if val != "NA":
                total += float(val) #
                count += 1
        
        if count == 0:
            return #avoid dividing by zero/end function
        
        avg = total / count
        #replace values with calculated avg
        for i in range(len(self.data)):
            if self.data[i][column_index] == "NA":
                self.data[i][column_index] = avg
        pass

   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_table = MyPyTable()
        summary_table.column_names = ["attribute", "min", "max", "mid", "avg", "median"]
        summary_table.data = []

        for col_name in col_names:
            column_index = -1
            for i in range(len(self.column_names)):
                if self.column_names[i] == col_name:
                    column_index = i
                    break
            if column_index == -1:
                continue #skip column if it is not found
            vals = []
            for row in range(len(self.data)):
                val = self.data[row][column_index]
                if val != "NA":
                    vals.append(float(val)) 
            
            if len(vals) == 0:
                continue #skip column if no numeric values found
            
            #sort values to calculate median value
            sorted_vals = sorted(vals)

            #calculate stats
            minimum = sorted_vals[0]
            maximum = sorted_vals[-1]
            mid_value = (minimum + maximum) / 2
            length = len(sorted_vals)
            average = sum(sorted_vals) / length

            if length % 2 == 1:
                #odd number length
                median = sorted_vals[length // 2]
            else:
                #even number length
                median = (sorted_vals[(length // 2) - 1] + sorted_vals[length // 2]) / 2 #average of two middle values
            
            #add calculated stats to summary_table
            summary_table.data.append([col_name, minimum, maximum, mid_value, average, median])

        return summary_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = MyPyTable()
        joined_table.column_names = self.column_names[:]

        for col in other_table.column_names:
            if col not in key_column_names:
                joined_table.column_names.append(col)
        
        self_key_index = []
        other_key_index = []
        for key in key_column_names:
            for i in range(len(self.column_names)):
                if self.column_names[i] == key:
                    self_key_index.append(i)
                    break
            for j in range(len(other_table.column_names)):
                if other_table.column_names[j] == key:
                    other_key_index.append(j)
                    break
        for self_row_index in range(len(self.data)):
            self_row = self.data[self_row_index]

            self_key_vals = []
            for i in range(len(self_key_index)):
                key_index = self_key_index[i]
                self_key_vals.append(self_row[key_index])

            for other_row_index in range(len(other_table.data)):
                other_row = other_table.data[other_row_index]

                other_key_vals = []
                for j in range(len(other_key_index)):
                    key_index = other_key_index[j]
                    other_key_vals.append(other_row[key_index])
                
                same = True
                for k in range(len(self_key_vals)):
                    if self_key_vals[k] != other_key_vals[k]:
                        same = False
                        break
                if same:
                    combine_row = []
                    #add values from self_row
                    for i in range(len(self_row)):
                        combine_row.append(self_row[i])

                    #add non-key columns from other_row
                    for i in range(len(other_table.column_names)):
                        is_key = False
                        for j in range(len(key_column_names)):
                            if other_table.column_names[i] == key_column_names[j]:
                                is_key = True
                                break
                        if not is_key:
                            combine_row.append(other_row[i])
                    #append the combined row to the joined table
                    joined_table.data.append(combine_row)

        return joined_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        joined_table = MyPyTable()
        joined_table.column_names = []
        for i in range(len(self.column_names)):
            joined_table.column_names.append(self.column_names[i])
        for i in range(len(other_table.column_names)):
            is_key = False
            for j in range(len(key_column_names)):
                if other_table.column_names[i] == key_column_names[j]:
                    is_key = True
                    break
            if not is_key:
                joined_table.column_names.append(other_table.column_names[i])
        
        self_key_index = []
        other_key_index = []
        for i in range(len(key_column_names)):
            key = key_column_names[i]
            for j in range(len(self.column_names)):
                if self.column_names[j] == key:
                    self_key_index.append(j)
                    break
            for k in range(len(other_table.column_names)):
                if other_table.column_names[k] == key:
                    other_key_index.append(k)
                    break

        same_other_rows = []

        for self_row_index in range(len(self.data)):
            self_row = self.data[self_row_index]

            self_key_vals = []
            for i in range(len(self_key_index)):
                self_key_vals.append(self_row[self_key_index[i]])
            
            row_match = False

            for other_row_index in range(len(other_table.data)):
                other_row = other_table.data[other_row_index]

                other_key_vals = []
                for j in range(len(other_key_index)):
                    other_key_vals.append(other_row[other_key_index[j]])

                same = True
                for k in range(len(self_key_vals)):
                    if self_key_vals[k] != other_key_vals[k]:
                        same = False
                        break
                if same:
                    row_match = True
                    same_other_rows.append(other_row_index)

                    combine_row = []
                    for i in range(len(self_row)):
                        combine_row.append(self_row[i])
                    for i in range(len(other_table.column_names)):
                        is_key = False
                        for j in range(len(key_column_names)):
                            if other_table.column_names[i] == key_column_names[j]:
                                is_key = True
                                break
                        if not is_key:
                            combine_row.append(other_row[i])
                    joined_table.data.append(combine_row)
            if not row_match:
                combine_row = []
                for i in range(len(self_row)):
                    combine_row.append(self_row[i])
                for i in range(len(other_table.column_names)):
                    is_key = False
                    for j in range(len(key_column_names)):
                        if other_table.column_names[i] == key_column_names[j]:
                            is_key = True
                            break
                    if not is_key:
                        combine_row.append("NA")
                joined_table.data.append(combine_row)
        
        for other_row_index in range(len(other_table.data)):
            if other_row_index not in same_other_rows:
                other_row = other_table.data[other_row_index]
                combine_row = []
                for i in range(len(self.column_names)):
                    is_key = False
                    for j in range(len(key_column_names)):
                        if self.column_names[i] == key_column_names[j]:
                            is_key = True
                            break
                    if is_key:
                        key_index_in_other = other_key_index[key_column_names.index(self.column_names[i])]
                        combine_row.append(other_row[key_index_in_other])
                    else:
                        combine_row.append("NA")
                for i in range(len(other_table.column_names)):
                    is_key = False
                    for j in range(len(key_column_names)):
                        if other_table.column_names[i] == key_column_names[j]:
                            is_key = True
                            break
                    if not is_key:
                        combine_row.append(other_row[i])
                joined_table.data.append(combine_row)

        return joined_table
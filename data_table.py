"""
HW-2 Data Table implementation.

NAME: George Calvert
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
from tabulate import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        self.__columns = columns.copy()
        self.__values = values.copy()

        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        # pretty print using tabulate wth the first value being the header
        return tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        return self.values()[self.columns().index(column)]


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """

        # TODO
        # check if column is in matrix
        # then remove value from that row if in matrix and the column
        if column in self.columns():
            colCopy = self.columns()
            valCopy = self.values()
            i = colCopy.index(column)
            colCopy.remove(column)
            valCopy.pop(i)
            self.__columns = colCopy
            self.__values = valCopy
        else:
            raise IndexError('not a valid column')
        
        pass

    
    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.

        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        # check if columns and values are equal
        if self.columns() == other.columns() and self.values() == other.values():
            return True
        return False

        pass

    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        if columns is None:
            return self.__values.copy()
        if not set(columns) <= set(self.columns()):
            raise ValueError('duplicate column names')
        return [self[column] for column in columns]


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """
    
        numVals = 0
        copyCol = columns
        # return all columns if none given
        if columns == None:
            numVals = len(self.columns())
            copyCol = self.columns()
        # empty results
        indexes = []
        val = []
        # go through column indexes
        for colName in copyCol:
            indexes.append(self.columns().index(colName))
        # get values from col indexes
        for index in indexes:
            val.append(self.values()[index])
        return DataRow(copyCol, val)
        pass

    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        self.__columns = columns.copy()
        self.__row_data = []


    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """
        # TODO
        # make empty array with first value being list of headers
        table = []
        for x in range(self.row_count()):
            table.append(self.__row_data[x].values())
        return tabulate(table, headers=self.columns())
        pass

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """
        return self.__row_data[row_index]

    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        # TODO
        # if index is in table then removes data row value
        try:
            copyRow = self.rows_copy()
            copyRow.pop(row_index)
            self.__row_data = copyRow
        except:
            raise IndexError('index out of range')
        pass

        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        return self.__columns.copy()

    def rows_copy(self):
        return self.__row_data.copy()

    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_data: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
        # TODO
        # result is equal to a new data row with values 
        result = DataRow(self.columns(), row_values)
        # add to row data
        self.__row_data.append(result)
    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        # TODO
        # new data table
        result = DataTable(self.columns())
        copyRow = self.rows_copy()
        # add data row of given indexes
        for value in row_indexes:
            r = copyRow[value]
            result.append(r.values())
        # return data row
        return result
        pass

    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        # TODO
        data_row_copy = self.__getitem__(row_index)
        data_row_copy.__setitem__(column, new_value)

        rowCopy = self.rows_copy()
        rowCopy[row_index] = data_row_copy
        self.__row_data = rowCopy


        pass

    
    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            nonmatches: Include non matches in answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine are must be in both tables.
            Duplicate column names removed from table2 portion of result.

        """
        # TODO
        # make new datatable
        table1Cols = table1.columns()
        table2Cols = table2.columns()
        new_columns = table1.columns()
        for col in columns:
            table2Cols.remove(col)
        new_columns += table2Cols
        result = DataTable(new_columns)

        # make dictionary to find matches
        # key equals the DataRow with the matching columns and element is the row number
        rows = []
        indexes = []
        matched_rows = []

        for x in range(table1.row_count()):
            # need to check for same values
            if set(table1[x].values(columns)) not in rows:
                rows.append(set(table1[x].values(columns)))
                indexes.append([x])
            else:
                index = rows.index(set(table1[x].values(columns)))
                indexes[index].append(x)
        # find matches and put into new table
        for y in range(table2.row_count()):
            # match
            if set(table2[y].values(columns)) in rows and len(columns) != 0:
                index = rows.index(set(table2[y].values(columns)))
                data_rows = indexes[index]
                for row in data_rows:
                    matched_rows.append(row)
                    table1_row = table1[row].values()
                    table1_row.extend(table2[y].values(table2Cols))
                    result.append(table1_row)
            # no match for table 2 cols so check if non matches
            if non_matches and ((set(table2[y].values(columns)) not in rows) or len(columns) == 0):
                temp = table2[y]
                d = DataRow(new_columns, ["" for x in range(len(new_columns))])
                for c in new_columns:
                    try:
                        d[c] =  temp[c] 
                    except:
                        d[c] = ''
                result.append(d.values()) 
        # non_matches for table 1 values
        if non_matches:
            for r in range(table1.row_count()):
                if r not in matched_rows:
                    temp = ["" for x in range(len(table2Cols))]
                    values = table1[r].values()
                    values.extend(temp)
                    result.append(values)
        return result

        pass
    def drop(self, columns):
        """Removes the given columns from the current table.
        Args:
        column: the name of the columns to drop
        """
        # make new columns for data table
        new_columns = self.columns()
        for col in columns:
            new_columns.remove(col)

        # list to be filled with new datarows
        new_row_data = []
        for row in range(self.row_count()):
            new_row_data.append(DataRow(new_columns, self[row].values(new_columns)))
        # set new values
        self.__columns = new_columns
        self.__row_data = new_row_data
        pass

    
    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
        # TODO
        try:
            value = int(value)
            return value
        except:
        # either float or string
        # try float conversion
            try:
                value = float(value)
                return value
            # value is a string
            except:
                return value
        pass
    

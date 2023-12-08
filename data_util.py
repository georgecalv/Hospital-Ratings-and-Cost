"""Data utility functions.

NAME: George Calvert
DATE: Fall 2023
CLASS: CPSC 322

"""

from math import sqrt

from data_table import DataTable, DataRow
import matplotlib.pyplot as plt


def normalize(table, column):
    """Normalize the values in the given column of the table. This
    function modifies the table.

    Args:
        table: The table to normalize.
        column: The column in the table to normalize.

    """
    # #TODO
    # get max and min for col
    column_list = column_values(table, column)
    column_list = [DataTable.convert_numeric(x) for x in column_list]
    minimum = min(column_list)
    maximum = max(column_list)
    for x in range(len(column_list)):
        table[x][column] = float((column_list[x] - minimum) / (maximum - minimum))
    pass



def discretize(table, column, cut_points):
    """Discretize column values according to the given list of n-1
    cut_points to form n ordinal values from 1 to n. This function
    modifies the table.

    Args:
        table: The table to discretize.
        column: The column in the table to discretize.

    """
    column_list = column_values(table, column)
    column_list = [DataTable.convert_numeric(x) for x in column_list]
    maximum = max(column_list)
    cut_points.append(maximum + 1)
    row_modified = []
    num_discretize = 1
    for val in range(len(cut_points)):
        for row in range(table.row_count()):
            if row not in row_modified and table[row][column] < cut_points[val]:
                table[row][column] = num_discretize
                row_modified.append(row) 
        num_discretize += 1

    #TODO
    pass


#----------------------------------------------------------------------
# HW4
#----------------------------------------------------------------------


def column_values(table, column):
    """Returns a list of the values (in order) in the given column.
    Args:
        table: The data table that values are drawn from
        column: The column whose values are returned
    
    """
    #TODO
    result = []
    for row in range(table.row_count()):
        result.append(table[row][column])
    return result
    pass



def mean(table, column):
    """Returns the arithmetic mean of the values in the given table
    column.
    Args:
        table: The data table that values are drawn from
        column: The column to compute the mean from
    Notes: 
        Assumes there are no missing values in the column.
    """
    #TODO
    mean = lambda xs : None if not len(xs) else sum(xs) / len(xs)
    return summary_stat(table, column, mean)
    pass


def variance(table, column):
    """Returns the variance of the values in the given table column.
    Args:
        table: The data table that values are drawn from
        column: The column to compute the variance from
    Notes:
        Assumes there are no missing values in the column.
    """
    #TODO
    x_bar = mean(table, column)
    variance = lambda xs: sum((x - x_bar) ** 2 for x in xs) / len(xs)
    return summary_stat(table, column, variance)
    pass


def std_dev(table, column):
    """Returns the standard deviation of the values in the given table
    column.
    Args:
        table: The data table that values are drawn from
        column: The colume to compute the standard deviation from
    Notes:
        Assumes there are no missing values in the column.
    """
    #TODO
    return (variance(table, column)) ** (1/2)
    pass



def covariance(table, x_column, y_column):
    """Returns the covariance of the values in the given table columns.
    
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x-values"
        y_column: The column with the "y-values"
    Notes:
        Assumes there are no missing values in the columns.        
    """
    #TODO
    x_bar = mean(table, x_column)
    y_bar = mean(table, y_column)
    sum = 0
    for row in range(table.row_count()):
        sum += (table[row][x_column] - x_bar) * (table[row][y_column] - y_bar)
    try:
        sum = sum / (table.row_count())
        return sum
    except:
        return sum

    pass


def linear_regression(table, x_column, y_column):
    """Returns a pair (slope, intercept) resulting from the ordinary least
    squares linear regression of the values in the given table columns.
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"
    """
    x_bar = mean(table, x_column)
    y_bar = mean(table, y_column)
    s = 0
    var = lambda xs: sum((x - x_bar) ** 2 for x in xs)
    denom = summary_stat(table, x_column, var)
    for row in range(table.row_count()):
        s += (table[row][x_column] - x_bar) * (table[row][y_column] - y_bar)
    try:
        m = s / denom
        b = y_bar - (m * x_bar)
        return (m, b)
    except:
        return (0, 0)
    pass


def correlation_coefficient(table, x_column, y_column):
    """Return the correlation coefficient of the table's given x and y
    columns.
    Args:
        table: The data table that value are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"
    Notes:
        Assumes there are no missing values in the columns.        
    """
    x_bar = mean(table, x_column)
    y_bar = mean(table, y_column)
    numerator = 0
    for row in range(table.row_count()):
        numerator += (table[row][x_column] - x_bar) * (table[row][y_column] - y_bar)
    x_var = lambda xs: sum((x - x_bar) ** 2 for x in xs)
    y_var = lambda xs: sum((y - y_bar) ** 2 for y in xs)
    denom = (summary_stat(table, x_column, x_var) * summary_stat(table, y_column, y_var)) ** (1/2)
    r = numerator / denom
    return r
    pass


def frequency_of_range(table, column, start, end):
    """Return the number of instances of column values such that each
    instance counted has a column value greater or equal to start and
    less than end. 
    
    Args:
        table: The data table used to get column values from
        column: The column to bin
        start: The starting value of the range
        end: The ending value of the range
    Notes:
        start must be less than end
    """
    #TODO
    result = 0
    if start <= end:
        for row in range(table.row_count()):
            if table[row][column] >= start and table[row][column] < end:
                result += 1
    return result
    pass


def histogram(table, column, nbins, xlabel, ylabel, title, filename=None):
    """Create an equal-width histogram of the given table column and number of bins.
    
    Args:
        table: The data table to use
        column: The column to obtain the value distribution
        nbins: The number of equal-width bins to use
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)
    Notes:
        If filename is given, chart is saved and not displayed.
    """
    #TODO
    l = []
    for row in range(table.row_count()):
        l.append(table[row][column])

    plt.figure()
    plt.title(title)
    plt.hist(l, nbins)
    plt.xlabel(xlabel)
    plt.xticks(rotation='vertical')
    plt.ylabel(ylabel)

    plt.figure(figsize=(120,120))

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()
    pass


def scatter_plot_with_best_fit(table, xcolumn, ycolumn, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values that includes the "best fit" line.
    
    Args:
        table: The data table to use
        xcolumn: The column for x-values
        ycolumn: The column for y-values
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)
    Notes:
        If filename is given, chart is saved and not displayed.
    """
    #TODO
    x_list = []
    y_list = []
    line_tuple = linear_regression(table, xcolumn, ycolumn)
    m = line_tuple[0]
    b = line_tuple[1]
    for row in range(table.row_count()):
        x_list.append(table[row][xcolumn])
        y_list.append(table[row][ycolumn])
    pred_y = []
    for val in x_list:
        pred_y.append((m * val) + b)

    plt.figure()
    plt.title(title)
    plt.scatter(x_list, y_list)
    plt.plot(x_list, pred_y, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()
    pass

#----------------------------------------------------------------------
# HW3
#----------------------------------------------------------------------

def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    # TODO
    # blank list to start
    result = []
    # go through column and add to resukt list if not seen before
    for row in range(table.row_count()):
        temp = table[row][column]
        if temp not in result:
            result.append(temp)
    return result
    pass


def remove_missing(table, columns):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    result = DataTable(table.columns())
    if table.row_count() != 0:
        data_rows = table.rows_copy()
        list_rows = []
        final_rows = []
        for r in data_rows:
            list_rows.append(r.values(columns))
            final_rows.append(r.values())
        # check column in table
        for col in columns:
            try:
                x = table[0][col]
            except:
                raise IndexError("column is not in table")
            
        # go through full table
        length = len(list_rows)
        x = 0
        # delete rows with '' in column given
        while x < length:
            temp = list_rows[x]
            # go through vals in row
            for val in temp:
                if val == '':
                    list_rows.pop(x)
                    final_rows.pop(x)
                    x -= 1
                    length -= 1
            x += 1
        # add row to result 
        for x in final_rows:
            result.append(x)
    return result
    pass


def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """
    # TODO
    list_no_duplicates = []
    list_duplicates = []
    result = DataTable(table.columns())
    # go through rows
    for row in table:
        # no duplicate yet
        if row.values() not in list_no_duplicates:
            list_no_duplicates.append(row.values())
        # is a duplicate but not already added to result
        elif row.values() not in list_duplicates:
            list_duplicates.append(row.values())
            result.append(row.values())

    return result
    pass

                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    # TODO
    result = DataTable(table.columns())
    non_duplicate_list = []
    # go through table
    for x in range(table.row_count()):
        temp = table[x].values()
        if temp not in non_duplicate_list:
            non_duplicate_list.append(temp)
    # set table row values to non_duplicate list
    for values in non_duplicate_list:
        result.append(values)
    return result
    pass


def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on
    """
    # TODO
    # list with indexes of type data table
    # size of list is the number of unique columns
    result_list = []
    unique_list = []
    for row in range(table.row_count()):
        # list with columns to match on
        temp = table[row].values(columns)

        # add to unique list
        # add table with values to result_list
        if temp not in unique_list:
            unique_list.append(temp)
            t = DataTable(table.columns())
            t.append(table[row].values())
            result_list.append(t)

        # already seen
        else:
            index = unique_list.index(temp)
            result_list[index].append(table[row].values())
    return result_list

pass


def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """
    # TODO

    # table_copy = remove_missing(table, column)
    # list = []
    # for row in range(table_copy.row_count()):
    #     list.append(table_copy[row].values(column)[0])
    # return(function(list))
    list = []
    for row in range(table.row_count()):
        if(table[row][column] != ''):
            list.append(table[row][column])
    return(function(list))
    pass


def replace_missing(table, column, partition_columns, function): 
    """Replace missing values in a given table's column using the provided
     function over similar instances, where similar instances are
     those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """
    # TODO
    result = DataTable(table.columns())
    data_table_list = partition(table, partition_columns)
    for data_table in data_table_list:
        stat = summary_stat(data_table, column, function)
        for row in range(data_table.row_count()):
            if data_table[row][column] == '':
                data_table.update(row, column, stat)
            result.append(data_table[row].values())
    return result
    pass


def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """
    # TODO
    group_list = partition(table, partition_column)
    col_list = []
    stat_list = []
    for t in group_list:
        stat_list.append(summary_stat(t, stat_column, function))
        col_list.append(t[0][partition_column])
    return col_list, stat_list
    
    pass


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """
    # TODO
    group_list = partition(table, partition_column)
    freq_list = []
    num_list = []
    for t in group_list:
        freq_list.append(t.row_count())
        num_list.append(t[0][partition_column])
    return num_list, freq_list
    pass


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # reset figure
    plt.figure()
    # dummy y values
    yvalues = [1] * len(xvalues)
    # create an x-axis grid
    plt.grid(axis='x', color='0.85', zorder=0)
    # create the dot chart (with pcts)
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=16, zorder=3)
    # get rid of the y axis
    plt.gca().get_yaxis().set_visible(False)
    # assign the axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # TODO
    plt.figure()
    plt.title(title)
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()





def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_labels: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # TODO
    plt.figure()
    plt.title(title)
    plt.bar(bar_values, bar_names)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # TODO
    plt.figure()
    plt.title(title)
    plt.scatter(xvalues, yvalues)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()


def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # TODO
    plt.figure()
    plt.title(title)
    plt.boxplot(distributions, labels=labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()


    

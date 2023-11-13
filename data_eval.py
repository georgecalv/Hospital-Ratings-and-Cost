"""Machine learning algorithm evaluation functions. 

NAME: George Calvert
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_learn import *
from data_util import *
import random


def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    # make tbale folds
    result = [DataTable(table.columns()) for x in range(k)]
    p_list = partition(table, [label_column])
    # go through each partition
    for part in p_list:
        # distribute each row in each partition
        counter = 0
        for row in range(part.row_count()):
            # put into fold
            result[counter].append(part[row].values())
            if counter == (k - 1):
                counter = 0
            else:
                counter += 1
    return result     
    pass


def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    if len(tables) < 1:
        raise ValueError("Bad Union")
    cols_check = tables[0].columns()
    result_table = DataTable(tables[0].columns())
    for table in tables:
        if table.columns() != cols_check:
            raise ValueError("Mismatching columns")
        for row in range(table.row_count()):
            result_table.append(table[row].values())
    return result_table
    pass


def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # make confusion matrix base
    confusion_cols = ["actual"]
    # row for confusion matrix
    dist = distinct_values(test, label_col)
    confusion_cols.extend(dist)
    confusion_matrix = DataTable(confusion_cols)
    values = [0 for x in range(len(confusion_cols))]
    for col in dist:
        values[0] = col
        confusion_matrix.append(values)

    # go through each row in test table to get predictions
    for row in range(test.row_count()):
        prediction_tuple = naive_bayes(train, test[row], label_col, continuous_cols, categorical_cols)
        prediction_label = prediction_tuple[0][0]
        actual = test[row][label_col]
        actual_col = column_values(confusion_matrix, "actual")
        index_actual = actual_col.index(actual)
        confusion_matrix[index_actual][prediction_label] += 1

    return confusion_matrix

    pass


def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # make confusion matrix base
    confusion_cols = ["actual"]
    # row for confusion matrix
    dist = distinct_values(table, label_col)
    confusion_cols.extend(dist)
    confusion_matrix = DataTable(confusion_cols)
    values = [0 for x in range(len(confusion_cols))]
    for col in dist:
        values[0] = col
        confusion_matrix.append(values)

    # k folds
    folds = stratify(table, label_col, k_folds)
    for test in folds:
        fold_matrix = naive_bayes_eval(table, test, label_col, cont_cols, cat_cols)
        matrix_cols = fold_matrix.columns()
        matrix_cols.remove('actual')
        for row in range(fold_matrix.row_count()):
            for col in matrix_cols:
                confusion_matrix[row][col] += fold_matrix[row][col]
    return confusion_matrix

    
    pass


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # make confusion matrix base
    confusion_cols = ["actual"]
    # row for confusion matrix
    dist = distinct_values(table, label_col)
    confusion_cols.extend(dist)
    confusion_matrix = DataTable(confusion_cols)
    values = [0 for x in range(len(confusion_cols))]
    for col in dist:
        values[0] = col
        confusion_matrix.append(values)

    # evaluate knn for k folds
    folds = stratify(table, label_col, k_folds)
    # go through table in each fold
    for test in folds:
        fold_matrix = knn_eval(table, test, vote_fun, k, label_col, num_cols, nom_cols)
        matrix_cols = fold_matrix.columns()
        matrix_cols.remove('actual')
        for row in range(fold_matrix.row_count()):
            for col in matrix_cols:
                confusion_matrix[row][col] += fold_matrix[row][col]
    return confusion_matrix
    pass

def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    training_set = DataTable(table.columns())
    test_set = DataTable(table.columns())
    indexes = random.sample(range(table.row_count()), test_set_size)
    for row in range(table.row_count()):
        if row in indexes:
            test_set.append(table[row].values())
        else:
            training_set.append(table[row].values())
    return (training_set, test_set)
    pass


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.
    """

    # make confusion matrix
    confusion_cols = ['actual']
    label_vals = distinct_values(test, label_col)
    confusion_cols.extend(label_vals)
    confusion_matrix = DataTable(confusion_cols)

    vals = [0 for val in label_vals]
    for label in label_vals:
        temp = [label]
        temp.extend(vals)
        confusion_matrix.append(temp)


    # get testing results
    for row in range(test.row_count()):
        closest_dict = knn(train, test[row], k, numeric_cols, nominal_cols) 
        closest_instances = list_of_closest(closest_dict)
        # get predicted label
        predicted_labels = vote_fun(closest_instances, list(closest_dict.keys()), label_col)
        predicted = predicted_labels[0]
        actual = test[row][label_col]

        # find what row to add value to 
        actual_row = column_values(confusion_matrix, 'actual').index(actual)
        # row is the actual and the column is the value predicted by the model
        confusion_matrix[actual_row][predicted] += 1
    return confusion_matrix
    pass

def list_of_closest(closest_dict):
    """Returns a list of the elements in a dictionary. 

    Args:
        closest_dict: A dictuionary of key value pairs where the key
                        is the distance and the element is a 
                        dataRows

    Returns: A list of dataRows

    Note: Each element is a list to traverse
    """
    keys = closest_dict.keys()
    result = []
    for key in keys:
        for data_row in closest_dict[key]:
            result.append(data_row)
    return result


def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    denom = 0
    actual_list = [confusion_matrix[row]['actual'] for row in range(confusion_matrix.row_count())]
    numer = confusion_matrix[actual_list.index(label)][label]
    new_cols = confusion_matrix.columns().copy()
    new_cols.remove('actual')
    new_cols.remove(label)
    new_rows = [x for x in range(confusion_matrix.row_count())]
    # remove row with the acutal of the label
    index = column_values(confusion_matrix, 'actual').index(label)
    new_rows.remove(index)
    for row in new_rows:
        for col in new_cols:
            numer += confusion_matrix[row][col]


    # for x in range(confusion_matrix.row_count()):
    #     for y in range(1, len(confusion_matrix.columns())):
    #         denom += confusion_matrix[x][y]
    for x in range(confusion_matrix.row_count()):
        for col in confusion_matrix.columns():
            if not(col == 'actual'):
                denom += confusion_matrix[x][col]
    return (numer / denom)
    pass



def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    index = column_values(confusion_matrix, 'actual').index(label)
    numer = confusion_matrix[index][label]
    denom = 0
    for row in range(confusion_matrix.row_count()):
        denom += confusion_matrix[row][label]
    return (numer / denom)
    pass



def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    index = column_values(confusion_matrix, 'actual').index(label)
    numer = confusion_matrix[index][label]
    denom = 0
    for col in confusion_matrix.columns():
        if not(col == 'actual'):
            denom += confusion_matrix[index][col]

    return (numer / denom)
    pass


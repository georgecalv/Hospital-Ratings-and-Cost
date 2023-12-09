"""Machine learning algorithm evaluation functions. 

NAME: George Calvert
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from random import *



#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------

def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """
    training_set = DataTable(table.columns())
    testing_set = DataTable(table.columns())
    list_row_nums = [x for x in range(table.row_count())]
    for row in range(table.row_count()):
        num = randint(0, table.row_count() - 1)
        if num in list_row_nums:
            list_row_nums.remove(num)
        training_set.append(table[num].values())
    testing_set = table.rows(list_row_nums)
    return (training_set, testing_set)

    
    pass



def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col: The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    #TODO
    # distributions
    l = label_col[0]
    dist_vals = distinct_values(table, l)
    dict_count = {}
    for val in dist_vals:
        dict_count[val] = column_values(table, l).count(val)
    
    training_set = DataTable(table.columns())
    test_set = DataTable(table.columns())
    rows_added = []
    # create_table
    for label, count in dict_count.items():
        # get indexes of rows with label
        indexes = [x for x in range(table.row_count()) if table[x][l] == label]
        shuffle(indexes)
        # num rows to include in table
        temp_size = int(test_set_size * count / table.row_count())
        
        # get data rows
        x = 0
        while x < temp_size:
            test_set.append(table[indexes[x]].values())
            x += 1
        
        rows_added.extend(indexes[0:temp_size])

    for r in range(table.row_count()):
        if r not in rows_added:
            training_set.append(table[r].values())
    return (training_set, test_set)
    pass
    


def tdidt_eval_with_tree(dt_root, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       td_root: The decision tree to use.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    """
    # confusion matrix table 
    confusion_cols = ["actual"]
    # cols for confusion matrix
    instances = summarize_instances(dt_root)
    for item, num in instances.items():
        confusion_cols.append(item)
    confusion_matrix = DataTable(confusion_cols)
    values = [0 for x in range(len(confusion_cols) - 1)]
    # rows for confusion
    for col in confusion_cols:
        if not(col == 'actual'):
            temp = [col]
            temp.extend(values)
            confusion_matrix.append(temp)
    # fill
    for row in range(test.row_count()):
        prediction, percent = tdidt_predict(dt_root, test[row])
        actual = test[row][label_col]
        # if value predicted not in test set not in training set
        try:
            index = column_values(confusion_matrix, 'actual').index(actual)
        except:
            temp_col = confusion_matrix.columns()
            temp_col.append(actual)
            new_confusion_matrix = DataTable(temp_col)
            for row in range(confusion_matrix.row_count()):
                temp = confusion_matrix[row].values()
                temp.append(0)
                new_confusion_matrix.append(temp)
            new_row = [actual]
            temp = [0 for x in range(new_confusion_matrix.column_count() - 1)]
            new_row.extend(temp)
            new_confusion_matrix.append(new_row)
            confusion_matrix = new_confusion_matrix
            index = column_values(confusion_matrix, 'actual').index(actual)
        confusion_matrix[index][prediction] += 1 
    return confusion_matrix
    pass



def random_forest(table, remainder, F, M, N, label_col, columns):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """
    # Create N bootstrap samples from remainder build tree and test
    # key = accuracy, element equals list of trees
    result_dict = {}
    for x in range(N):
        # create boostrap sample to make tree
        training_set, validation_set = bootstrap(remainder)
        tree = tdidt_F(training_set, label_col, F, columns)
        # clean
        tree = resolve_attribute_values(tree, table)
        tree = resolve_leaf_nodes(tree)

        # get confusion matrix for tree
        matrix = tdidt_eval_with_tree(tree, validation_set, label_col, columns)

        # calculate accuracy
        cols = matrix.columns()
        cols.remove('actual')
        acc = 0
        for c in cols:
            acc += accuracy(matrix, c)
        try:
            acc /= len(cols)
        except:
            acc = 0
        # if acc in result_dict:
        #     result_dict[acc].append(tree)
        # else:
            # result_dict[acc] = [tree]
        result_dict[acc] = tree
    # choose highest M trees
    key_list = list(result_dict.keys())
    key_list.sort(reverse=True)
    result = []
    num = 0
    while len(result) < M:
        try:
            result.append((result_dict[key_list[num]], key_list[num]))
        except:
            break 
    return result
    pass



def random_forest_eval(table, training, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        training: training set to use
        test: test set to use
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    # make forest
    # divide into remainder and test
    # remainder = 2/3 |D|
    # remainder, new_test = stratified_holdout(table, label_col, (table.row_count() // 3))
    forest = random_forest(table, training, F, M, N, label_col, columns)
    # make confusion_matrix
    confusion_cols = ["actual"]
    dist = distinct_values(table, label_col)
    confusion_cols.extend(dist)
    confusion_matrix = DataTable(confusion_cols)
    vals = [0 for x in range(confusion_matrix.column_count() - 1)]
    for col in confusion_cols:
        if col != "actual":
            temp = [col]
            temp.extend(vals)
            confusion_matrix.append(temp)
    # make temp datarow
    temp_row = DataRow(table.columns(), ['' for x in range(table.column_count())])
    # evaluate forest
    for row in range(test.row_count()):
        # list of predicted rows
        prediction_list = []
        # list of accuracies for prediction
        cor_scores = []
        # temp row
        temp_row = DataRow(table.columns(), test[row].values())
        # get predictions
        for tree in forest:
            prediction, percent = tdidt_predict(tree[0], test[row])
            temp_row[label_col] = prediction
            prediction_list.append(temp_row)
            cor_scores.append(percent)
        # list of prediction
        final_predict = weighted_vote(prediction_list, cor_scores, label_col)
        final_predict = final_predict[0]
        actual = test[row][label_col]
        
        # put in matrix
        index = column_values(confusion_matrix, 'actual').index(actual)
        confusion_matrix[index][final_predict] += 1
    return confusion_matrix
    pass



def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    #TODO
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

    tree = tdidt(train, label_col, columns)
    tree = resolve_attribute_values(tree, train)
    tree = resolve_leaf_nodes(tree)
    num_not_found = 0
    for row in range(test.row_count()):
        try:
            prediction = tdidt_predict(tree, test[row])
            predicted_label = prediction[0]
            actual_label = test[row][label_col]
            actual_col = column_values(confusion_matrix, "actual")
            index_actual = actual_col.index(actual_label)
            confusion_matrix[index_actual][predicted_label] += 1
        except:
            num_not_found += 1
    print("Number values not in tree", num_not_found)
    return confusion_matrix
    pass


def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

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
        fold_matrix = tdidt_eval(table, test, label_col, columns)
        matrix_cols = fold_matrix.columns()
        matrix_cols.remove('actual')
        for row in range(fold_matrix.row_count()):
            for col in matrix_cols:
                confusion_matrix[row][col] += fold_matrix[row][col]
    return confusion_matrix

    pass

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


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    training_set = DataTable(table.columns())
    test_set = DataTable(table.columns())
    indexes = sample(range(table.row_count()), test_set_size)
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
    try:
        result = (numer / denom)
    except:
        result = 0
    return result
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
    try:
        result = (numer / denom)
    except:
        result = 0
    return result
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

    try:
        result = (numer / denom)
    except:
        result = 0
    return result
    pass


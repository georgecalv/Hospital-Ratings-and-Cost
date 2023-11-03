"""Machine learning algorithm implementations.

NAME: George Calvert
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import DataTable, DataRow


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table. (datarow)
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """
    # dictionary of k key value pairs
    dict_neighbors = {}
    # go through rows
    for row in range(table.row_count()):
        # what columns to compare
        dist = 0
        # count numerical distance
        for col in numerical_columns:
            # add to distance
            dist += (instance[col] - table[row][col]) ** 2
        # count nominal distance
        for nom_col in nominal_columns:
            if instance[nom_col] != table[row][nom_col]:
                dist += 1
        # check if checking columns has values
        if len(numerical_columns) > 0 or len(nominal_columns) > 0:
            # check if key exists and add to dictionary
            if dist in dict_neighbors:
                dict_neighbors[dist].append(table[row])
            else:  
                dict_neighbors[dist] = [table[row]]
    # return the k closest
    # closest dictionary
    closest = {}
    sorted_key_list = sorted(dict_neighbors.keys())
    if k <= len(sorted_key_list):
        for x in range(k):
            closest[sorted_key_list[x]] = dict_neighbors[sorted_key_list[x]]
    else:
        return dict_neighbors
    
    return closest
    pass



def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    labels = []
    freq = []

    for row in instances:
        if row[labeled_column] not in labels:
            labels.append(row[labeled_column])
            freq.append(1)
        else:
            index = labels.index(row[labeled_column])
            freq[index] += 1
    result = []
    maximum = max(freq)
    for value in range(len(freq)):
        if freq[value] == maximum:
            result.append(labels[value])
    return result
    pass



def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    # list of majority voted columns
    labels = []
    points = []
    for dist in range(len(scores)):
        # label not in labels already
        if instances[dist][labeled_column] not in labels:
            labels.append(instances[dist][labeled_column])
            points.append(scores[dist])
        # label exists
        else:
            index = labels.index(instances[dist][labeled_column])
            points[index] += scores[dist]
    result = []
    maximum = max(points)
    for value in range(len(labels)):
        if points[value] == maximum:
            result.append(labels[value])
    return result

    pass


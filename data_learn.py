"""Machine learning algorithm functions. 

NAME: George Calvert
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

import math

def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    #TODO
    instance = table[0][label_col]
    for row in range(table.row_count()):
        if instance != table[row][label_col]:
            return False
    return True
    pass


def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    #TODO
    # key = label, element = count
    dict_labels = {}
    for row in range(table.row_count()):
        if table[row][label_col] not in dict_labels:
            dict_labels[table[row][label_col]] = 1
        else:
            dict_labels[table[row][label_col]] += 1
    keys = list(dict_labels.keys())
    result = []
    for key in keys:
        result.append(LeafNode(key, dict_labels[key], table.row_count()))
    return result
    pass


def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """
    #TODO
    # partition on each column in columns 
    # key = entropy calculated, element = list of columns with that entropy
    result = {}
    total_number = table.row_count()
    partition_dict = {}
    label_list = []
    table_list = []
    # distinct values of label_col
    for col in columns:
        p_list = partition(table, [col])
        partition_dict[col] = p_list
    for col in columns:
        label = col
        table_list = partition_dict[label]
        lable_entropy = 0
        for t in table_list: 
            entropy = t.row_count() / total_number
            d_vals = distinct_values(t, label_col)
            column = column_values(t, label_col)
            e = 0
            for val in d_vals:
                c = column.count(val)
                c /= len(column)
                e += -1 * (c * (math.log(c, 2)))
            entropy *= e
            lable_entropy += entropy
        if lable_entropy in result:
            result[lable_entropy].append(label)
        else:
            result[lable_entropy] = [label]
    return result
    pass



def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    #TODO
    # base cases
    # no rows in partition
    if table.row_count() < 1:
        return None 
    # label col is all the same
    elif same_class(table, label_col):
        return [LeafNode(table[0][label_col], table.row_count(), table.row_count())]
    # no more columns to partition on
    elif len(columns) == 0:
        return build_leaves(table, label_col)
    
    # recursive case
    # find e new for all columns
    e_new_list = calc_e_new(table, label_col, columns)

    # find smallest e_new
    key_list = list(e_new_list.keys())
    min_key = min(key_list)
    partition_col = e_new_list[min_key][0]
    new_cols = []
    for col in columns:
        if col != partition_col:
            new_cols.append(col)
    # create attribute node and fill values nodes(recursive calls on partition)
    p_list = partition(table, [partition_col])
    dict_vals = {}
    for p in p_list:
        dict_vals[p[0][partition_col]] = tdidt(p, label_col, new_cols)
    return AttributeNode(partition_col, dict_vals)
    pass


def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    #TODO
    # base cases
    result = {}
    # leaf node
    if type(dt_root) == LeafNode:
        # go through vals in leaf node
        return {dt_root.label: dt_root.count}
    # list of leaf nodes
    elif type(dt_root) == list:
        temp_dict = {}
        for leaf in dt_root:
            temp_dict[leaf.label] = leaf.count
        return temp_dict
              
    # recursive case
    keys_list = list(dt_root.values.keys())
    for key in keys_list:
        temp_dict = summarize_instances(dt_root.values[key])
        k = list(temp_dict.keys())
        for x in k:
            if x not in result:
                result[x] = temp_dict[x]
            else:
                result[x] += temp_dict[x]
    return result
    pass


def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    # base cases
    if type(dt_root) == LeafNode:
        copy_root = LeafNode(dt_root.label, dt_root.count, dt_root.total)
        return copy_root
    elif type(dt_root) == list:
        copy_list = dt_root.copy()
        return copy_list
    # recursive case
    else:
        new_dt_root = AttributeNode(dt_root.name, dt_root.values.copy())

        # go through copies
        # val is the dictionary key and child is the list of leaves or an attribute node 
        new_dt_root_values = {}
        for val, child in dt_root.values.items():
            # if it is a list
            if type(child) == list:
                maximum = child[0].count
                new_node = LeafNode(child[0].label, child[0].count, child[0].total)
                for leaf in child:
                    if leaf.count > maximum:
                        maximum = leaf.count
                        new_node = LeafNode(leaf.label, leaf.count, leaf.total)
                new_dt_root_values[val] = [new_node]
            # attribute node in list
            else:
                new_dt_root_values[val] = resolve_leaf_nodes(child)
        new_dt_root.values = new_dt_root_values
        return new_dt_root
    pass


def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """
    # base cases
    if type(dt_root) == LeafNode:
        copy_root = LeafNode(dt_root.label, dt_root.count, dt_root.total)
        return copy_root
    elif type(dt_root) == list:
        copy_list = dt_root.copy()
        return copy_list
    # recursive case
    else:
        new_dt_root = AttributeNode(dt_root.name, dt_root.values.copy())
        new_dt_root_vals = {}
        dist_list = distinct_values(table, dt_root.name)
        edges = list(dt_root.values.keys())
        # unequal edges
        if len(edges) != len(dist_list):
            # summary is the leaves coming off the attributre node
            summary = summarize_instances(dt_root)
            list_new_nodes = []
            total = 0
            # get total to make percentages
            for label, num in summary.items():
                total += num
            # create list of leaf nodes
            for label, num in summary.items():
                temp = LeafNode(label, num, total)
                list_new_nodes.append(temp)
            return list_new_nodes
        # equal edges
        else:
            for val, child in dt_root.values.items():
                if type(child) == AttributeNode:
                    temp = resolve_attribute_values(child, table)
                    new_dt_root_vals[val] = temp
                else:
                    new_dt_root_vals[val] = child
            new_dt_root.values = new_dt_root_vals
            return new_dt_root

        # new_dt_root = AttributeNode(dt_root.name, dt_root.values.copy())
        # new_dt_root_vals = {}
        # dist_list = distinct_values(table, dt_root.name)
        # edges = list(dt_root.values.keys())
        # # not right amount of edges
        # if len(dist_list) != len(edges):
        #     new_node = []
        #     total = 0
        #     for val, child in dt_root.values.items():
        #         if type(child) == AttributeNode:
        #             temp = resolve_attribute_values(child, table)
        #             for n in temp:
        #                 total += n.total
        #             new_node.extend(temp)
        #         else:
        #             for n in child:
        #                 total += n.total
        #             new_node.extend(child)
        #     for leaf in new_node:
        #         leaf.total = total
        #     return new_node
        # # same edges so do recursive calls on attribute nodes
        # else:
        #     for val, child in dt_root.values.items():
        #         if type(child) == AttributeNode:
        #             temp = resolve_attribute_values(child, table)
        #             new_dt_root_vals[val] = temp
        #         else:
        #             new_dt_root_vals[val] = child
        #     new_dt_root.values = new_dt_root_vals
        #     return new_dt_root
    pass


def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    """
    # leaf
    if type(dt_root) == LeafNode:
        result = (dt_root.label, dt_root.percent())
        return result
    # attribute
    elif type(dt_root) == AttributeNode:
        label = dt_root.name
        next_node_index = instance[label]
        next_node = dt_root.values[next_node_index]
        return tdidt_predict(next_node, instance)
    # list leaf nodes
    else:
        max_count = dt_root[0].count
        temp = dt_root[0]
        for leaf in dt_root:
            if leaf.count > max_count:
                max_count = leaf.count
                temp = leaf
        return (temp.label, temp.percent())
    pass
def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (labels, prob) consisting of a list of the labels
        with the highest probability and the corresponding highest
        probability.

    """
    # catgorical probabilities
    # key = probability, element = list of labels
    prob_dict = {}
    # label probaility
    class_probs = {}
    # key = label, element is a list of rows where label occured
    list_indices = {}
    dist = distinct_values(table, label_col)
    # prob label and getting row indices for row
    for label in dist:
        for row in range(table.row_count()):
            if table[row][label_col] == label:
                if label in class_probs:
                    class_probs[label] += 1
                else:
                    class_probs[label] = 1
                # dictionary for rows with label
                if label in list_indices:
                    list_indices[label].append(row)
                else:
                    list_indices[label] = [row]
    if len(categorical_cols) >= 1:
        # prob instance given label fo categorical
        for label in dist:
            rows = list_indices[label]
            count = 0
            # go through categorical columns
            # val = prob for x given label
            val = 1
            for cat in categorical_cols:
                # count rows that have the same val in cat with label
                count = 0
                for row in rows:
                    if table[row][cat] == instance[cat]:
                        count += 1
                # numerator for feature calc
                frac = (count / class_probs[label])
                val *= frac
            prob = val * (class_probs[label] / table.row_count())
            if prob in prob_dict:
                prob_dict[prob].append(label)
            else:
                prob_dict[prob] = [label]
    # probs for continuous
    if len(continuous_cols) >= 1:
        for label in dist:
            for con in continuous_cols:
                temp = table.rows(list_indices[label])
                mew = mean(temp, con)
                std = std_dev(temp, con)
                prob_cont = (gaussian_density(instance[con], mew, std) * (class_probs[label] / table.row_count()))
                if prob_cont in prob_dict:
                    prob_dict[prob_cont].append(label)
                else:
                    prob_dict[prob_cont] = [label]
    actual_prob = max(prob_dict)
    return (prob_dict[actual_prob], actual_prob) 
    pass


def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    p = (2.0 * math.pi)
    denom = (math.sqrt(p) * sdev)
    frac = (1 / denom)
    exp = (-1.0 * ((x - mean)) ** 2) / (2 * ((sdev) ** 2))
    prod = math.e ** exp
    result = frac * prod
    return result
    pass


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
    # numerical
    # key = distance, element equals the datarow
    dict_neighbors = {}
    matching_cols = numerical_columns + nominal_columns

    # compute closest
    for row in range(table.row_count()):
        dist = 0
        # go through matching
        for col in matching_cols:
            # calc dist
            if col in numerical_columns:
                dist += (instance[col] - table[row][col]) ** 2
            elif col in nominal_columns:
                if table[row][col] != instance[col]:
                    dist += 1
        # add to dictionary
        if dist in dict_neighbors:
            dict_neighbors[dist].append(table[row])
        else:  
            dict_neighbors[dist] = [table[row]]
    keys_list = list(dict_neighbors.keys())
    keys_list.sort()
    count = 0
    result = {}
    for key in keys_list:
        count += 1
        if count <= k:
            result[key] = dict_neighbors[key]
        else:
            break
    return result
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


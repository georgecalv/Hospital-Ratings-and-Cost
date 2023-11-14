import pytest
import os
from data_learn import *
from data_eval import *
from data_util import *
from data_table import *

#----------------------------------------------------------------------
# knn tests
#----------------------------------------------------------------------

def test_one_row_one_col_knn():
    table = DataTable(['a'])    
    table.append([1])
    # exact match
    instance = DataRow(['a'], [1])
    result = knn(table, instance, 1, ['a'])
    assert len(result) == 1
    assert result[0] == [table[0]]
    # one off
    instance = DataRow(['a'], [2])
    result = knn(table, instance, 1, ['a'])
    assert len(result) == 1
    assert result[1] == [table[0]]
    # one off (other direction)
    instance = DataRow(['a'], [0])
    result = knn(table, instance, 1, ['a'])
    assert len(result) == 1
    assert result[1] == [table[0]]
    # k = 2 still returns one row
    instance = DataRow(['a'], [1])
    result = knn(table, instance, 2, ['a'])
    assert len(result) == 1
    assert result[0] == [table[0]]

    
def test_two_rows_one_col_knn():
    table = DataTable(['a'])
    table.append([1])
    table.append([2])
    # one instance for k=1
    instance = DataRow(['a'], [1])
    result = knn(table, instance, 1, ['a'])
    assert len(result) == 1
    assert result[0] == [table[0]]
    # two instances for k=1
    instance = DataRow(['a'], [1.5])
    result = knn(table, instance, 1, ['a'])
    assert len(result) == 1
    assert len(result[0.25]) == 2
    assert table[0] in result[0.25] and table[1] in result[0.25]
    # two instances for k=2
    instance = DataRow(['a'], [1])
    result = knn(table, instance, 2, ['a'])
    assert len(result) == 2
    assert result[0] == [table[0]]
    assert result[1] == [table[1]]
    
    
def test_two_rows_two_col_knn():
    table = DataTable(['a', 'b'])
    table.append([1, 2])
    table.append([2, 1])
    # one instance for k=1
    instance = DataRow(['a', 'b'], [2, 1])
    result = knn(table, instance, 1, ['a', 'b'])
    assert len(result) == 1
    assert result[0] == [table[1]]
    # two instances for k=1
    instance = DataRow(['a', 'b'], [1.5, 1.5])
    result = knn(table, instance, 1, ['a', 'b'])
    assert len(result) == 1
    assert len(result[0.5]) == 2
    assert table[0] in result[0.5] and table[1] in result[0.5]
    # two instances for k=2
    instance = DataRow(['a', 'b'], [2, 1])
    result = knn(table, instance, 2, ['a', 'b'])    
    assert len(result) == 2
    assert result[0] == [table[1]]
    assert result[2] == [table[0]]
    # one instance for k=2
    instance = DataRow(['a', 'b'], [1, 1])
    result = knn(table, instance, 2, ['a', 'b'])    
    assert len(result) == 1
    assert len(result[1]) == 2
    assert table[0] in result[1] and table[1] in result[1]

    
def test_mult_rows_mult_col_knn():
    table = DataTable(['a', 'b', 'c', 'd'])
    table.append([1, 2, 3, 4])
    table.append([2, 4, 6, 8])
    table.append([1, 3, 5, 7])
    table.append([4, 3, 2, 1])
    table.append([3, 1, 4, 2])
    instance = DataRow(['a', 'b', 'c', 'd'], [1, 3, 3, 4])
    # k = 1
    result = knn(table, instance, 1, table.columns())
    assert len(result) == 1
    assert result[1] == [table[0]]
    # k = 2
    result = knn(table, instance, 2, table.columns())
    assert len(result) == 2
    assert result[1] == [table[0]]
    assert table[2] in result[13] and table[4] in result[13]
    # k = 3
    result = knn(table, instance, 3, table.columns())
    assert len(result) == 3
    print(result.keys())
    assert result[1] == [table[0]]
    assert table[2] in result[13] and table[4] in result[13]
    assert result[19] == [table[3]]
    

def test_nominal_col_knn():
    table = DataTable(['a', 'b'])
    table.append([1, 2])
    table.append([2, 1])
    # one instance for k=1
    instance = DataRow(['a', 'b'], [3, 1])
    result = knn(table, instance, 1, [], ['b'])
    print(result)
    print(table)
    assert len(result) == 1
    assert result[0] == [table[1]]
    # two instances for k=2
    result = knn(table, instance, 2, [], ['b'])
    assert len(result) == 2
    assert result[0] == [table[1]]
    assert result[1] == [table[0]]
    # two instances, one result for k=1
    instance = DataRow(['a', 'b'], [2, 2])
    result = knn(table, instance, 1, [], ['a', 'b'])
    assert len(result) == 1
    assert result[1] == [table[0], table[1]]
    # two instances, two results for k=2
    instance = DataRow(['a', 'b'], [3, 1])
    result = knn(table, instance, 2, [], ['a', 'b'])
    assert len(result) == 2
    print(result)
    assert result[1] == [table[1]]
    assert result[2] == [table[0]]
    

def test_numerical_nominal_knn():
    table = DataTable(['a', 'b', 'c'])
    table.append([1, 2, 'y'])
    table.append([2, 4, 'y'])
    table.append([1, 3, 'n'])
    table.append([4, 3, 'n'])
    table.append([3, 1, 'y'])
    instance = DataRow(['a', 'b', 'c'], [2, 2, 'n'])
    # k = 1
    result = knn(table, instance, 1, ['a', 'b'], ['c'])
    assert len(result) == 1
    print(result)
    assert table[0] in result[2] and table[2] in result[2]
    # k = 2
    result = knn(table, instance, 2, ['a', 'b'], ['c'])
    assert len(result) == 2
    assert table[0] in result[2] and table[2] in result[2]
    assert result[3] == [table[4]]
    # k = 3
    result = knn(table, instance, 3, ['a', 'b'], ['c'])
    assert len(result) == 3
    assert table[0] in result[2] and table[2] in result[2]
    assert result[3] == [table[4]]
    print(result[5])
    assert table[1] in result[5] and table[3] in result[5]

# def test_no_column_combine_table():
#     t1 = DataTable(['x', 'y'])
#     t1.append([1, 20])
#     t1.append([3, 40])
#     t1.append([2, 30])
#     t2 = DataTable(['b', 'z'])
#     t2.append([30, 300])
#     t2.append([20, 100])
#     t2.append([50, 500])
#     t2.append([20, 200])
#     t2.append([60, 600])
#     t3 = DataTable.combine(t1, t2, [], True)
#     assert t3.columns() == ['x', 'y', 'b', 'z']
#     assert t3.row_count() == 8
#     rows = [t3[i].values() for i in range(t3.row_count())]
#     assert [1, 20, '', ''] in rows
#     assert [3, 40, '', ''] in rows
#     assert [2, 30, '', ''] in rows
#     assert ['', '', 30, 300] in rows
#     assert ['', '', 20, 100] in rows
#     assert ['', '', 50, 500] in rows
#     assert ['', '', 20, 200] in rows
#     assert ['', '', 60, 600] in rows


# def test_one_column_combine_table():
#     t1 = DataTable(['x', 'y'])
#     t1.append([1, 20])
#     t1.append([3, 40])
#     t1.append([2, 30])
#     t2 = DataTable(['y', 'z'])
#     t2.append([30, 300])
#     t2.append([20, 100])
#     t2.append([50, 500])
#     t2.append([20, 200])
#     t2.append([60, 600])
#     # # non_matches is false
#     t3 = DataTable.combine(t1, t2, ['y'])
#     assert t3.columns() == ['x', 'y', 'z']
#     assert t3.row_count() == 3
#     rows = [t3[i].values() for i in range(t3.row_count())]
#     assert [1, 20, 100] in rows
#     assert [1, 20, 200] in rows
#     assert [2, 30, 300] in rows
#     # non_matches is true
#     t3 = DataTable.combine(t2, t1, ['y'], True)
#     assert t3.columns() == ['y', 'z', 'x']
#     assert t3.row_count() == 6
#     rows = [t3[i].values() for i in range(t3.row_count())]    
#     assert [20, 100, 1] in rows
#     assert [20, 200, 1] in rows
#     assert [30, 300, 2] in rows
#     assert [40, '', 3] in rows
#     assert [50, 500, ''] in rows
#     assert [60, 600, ''] in rows
    
    
# def test_two_column_combine_table():
#     t1 = DataTable(['x', 'y', 'z'])
#     t1.append([1, 10, 100])
#     t1.append([2, 20, 200])
#     t1.append([2, 10, 200])
#     t1.append([3, 30, 300])
#     t2 = DataTable(['z', 'u', 'x'])
#     t2.append([200, 60, 2])
#     t2.append([100, 60, 1])
#     t2.append([400, 60, 2])
#     t2.append([100, 60, 1])
#     # non_matches is false
#     t3 = DataTable.combine(t1, t2, ['x', 'z'])
#     assert t3.columns() == ['x', 'y', 'z','u']
#     assert t3.row_count() == 4
#     rows = [t3[i].values() for i in range(t3.row_count())]
#     assert rows.count([1, 10, 100, 60]) == 2
#     assert [2, 20, 200, 60] in rows
#     assert [2, 10, 200, 60] in rows
#     # non_matches is true
#     t3 = DataTable.combine(t1, t2, ['x', 'z'], True)
#     assert t3.columns() == ['x', 'y', 'z','u']
#     assert t3.row_count() == 6
#     rows = [t3[i].values() for i in range(t3.row_count())]
#     assert rows.count([1, 10, 100, 60]) == 2
#     assert [2, 20, 200, 60] in rows
#     assert [2, 10, 200, 60] in rows
#     assert [3, 30, 300, ''] in rows
#     assert [2, '', 400, 60] in rows
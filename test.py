import pytest
import os
from data_table import *


def test_no_column_combine_table():
    t1 = DataTable(['x', 'y'])
    t1.append([1, 20])
    t1.append([3, 40])
    t1.append([2, 30])
    t2 = DataTable(['b', 'z'])
    t2.append([30, 300])
    t2.append([20, 100])
    t2.append([50, 500])
    t2.append([20, 200])
    t2.append([60, 600])
    t3 = DataTable.combine(t1, t2, [], True)
    assert t3.columns() == ['x', 'y', 'b', 'z']
    assert t3.row_count() == 8
    rows = [t3[i].values() for i in range(t3.row_count())]
    assert [1, 20, '', ''] in rows
    assert [3, 40, '', ''] in rows
    assert [2, 30, '', ''] in rows
    assert ['', '', 30, 300] in rows
    assert ['', '', 20, 100] in rows
    assert ['', '', 50, 500] in rows
    assert ['', '', 20, 200] in rows
    assert ['', '', 60, 600] in rows


def test_one_column_combine_table():
    t1 = DataTable(['x', 'y'])
    t1.append([1, 20])
    t1.append([3, 40])
    t1.append([2, 30])
    t2 = DataTable(['y', 'z'])
    t2.append([30, 300])
    t2.append([20, 100])
    t2.append([50, 500])
    t2.append([20, 200])
    t2.append([60, 600])
    # # non_matches is false
    t3 = DataTable.combine(t1, t2, ['y'])
    assert t3.columns() == ['x', 'y', 'z']
    assert t3.row_count() == 3
    rows = [t3[i].values() for i in range(t3.row_count())]
    assert [1, 20, 100] in rows
    assert [1, 20, 200] in rows
    assert [2, 30, 300] in rows
    # non_matches is true
    t3 = DataTable.combine(t2, t1, ['y'], True)
    assert t3.columns() == ['y', 'z', 'x']
    assert t3.row_count() == 6
    rows = [t3[i].values() for i in range(t3.row_count())]    
    assert [20, 100, 1] in rows
    assert [20, 200, 1] in rows
    assert [30, 300, 2] in rows
    assert [40, '', 3] in rows
    assert [50, 500, ''] in rows
    assert [60, 600, ''] in rows
    
    
def test_two_column_combine_table():
    t1 = DataTable(['x', 'y', 'z'])
    t1.append([1, 10, 100])
    t1.append([2, 20, 200])
    t1.append([2, 10, 200])
    t1.append([3, 30, 300])
    t2 = DataTable(['z', 'u', 'x'])
    t2.append([200, 60, 2])
    t2.append([100, 60, 1])
    t2.append([400, 60, 2])
    t2.append([100, 60, 1])
    # non_matches is false
    t3 = DataTable.combine(t1, t2, ['x', 'z'])
    assert t3.columns() == ['x', 'y', 'z','u']
    assert t3.row_count() == 4
    rows = [t3[i].values() for i in range(t3.row_count())]
    assert rows.count([1, 10, 100, 60]) == 2
    assert [2, 20, 200, 60] in rows
    assert [2, 10, 200, 60] in rows
    # non_matches is true
    t3 = DataTable.combine(t1, t2, ['x', 'z'], True)
    assert t3.columns() == ['x', 'y', 'z','u']
    assert t3.row_count() == 6
    rows = [t3[i].values() for i in range(t3.row_count())]
    assert rows.count([1, 10, 100, 60]) == 2
    assert [2, 20, 200, 60] in rows
    assert [2, 10, 200, 60] in rows
    assert [3, 30, 300, ''] in rows
    assert [2, '', 400, 60] in rows
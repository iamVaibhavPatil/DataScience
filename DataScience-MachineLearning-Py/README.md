# Introduction
Data Science and Machine learning using Python. We will explore different tools -

1) Numpy  
2) Pandas  
3) Matplotlib  
4) Seaborn  
5) Plotly and Cufflinks
6) Machine Learning Models  
    - Linear Regression  
    - Cross Validation and Bias-Variance Trade Off
    - Logistic Regression  
    - K Nearest Neighbors  
    - Decision Trees and Random Forest  
    - Support Vector Machines  
    - K Means Clustering  
    - Principal Component Anlysis
    - NLP  
    - Recommender System  
    - PySpark  
    - Neural Nets and Deep Learning  

# Envrionment Setup  
Anaconda Download and Install.

# Python for Data Analysis - NumPy
Numpy is a `Linear Algebra` library for Python. It is so important for Data Science with Python is that almost all of the libraries in the PyData Ecosystem rely on Numpy as one of their main building blocks.

NumPy is also incredibly fast, as it has bindings to C libraries.

If we have Anaconda distribution, we can install NumPy by executing below command-

```
# Recommended if Anaconda is installed.
conda install numpy

# Using Python, if no Anaconda distribution
pip install numpy
```

## NumPy Arrays  
NumPy arrays come in two flavors: `Vectors` and `Matrices`.  

Vectors are strictly 1-d arrays and matrices are 2-d(matrix can have only one row or column).

We can convert existing list or list of list into 1-d or 2-d array respectively.

```
my_list = [1,2,3]

import numpy as np

# 1-d array
arr = np.array(my_list)
arr
>>> array([1, 2, 3])

# 2-d Array
my_mat = [[1,2,3],[4,5,6],[7,8,9]]
np.array(my_mat)
>>>
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

Most common way to create numpy array is using `arange(start, stop, step)`. It is similar to `range()` function in python.

```
np.arange(0,11)
>>> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

np.arange(0,11,2)
>>> array([ 0,  2,  4,  6,  8, 10])
```

We can also create specific array like arrays with all zero values by specifying number of rows or columns.

`zeros(number of elements)` for 1-d arrays  
`zeros((rows, columns))` for 2-d arrays. Pass tuple of rows and columns.

```
np.zeros(3)
>>> array([0., 0., 0.])

np.zeros((3,3))
>>> array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
```

Similarly, `ones` for arrays with all 1.

```
np.ones(4)
>>> array([1., 1., 1., 1.])

np.ones((2,3))
>>> array([[1., 1., 1.],
       [1., 1., 1.]])
```

`linspace(start, stop, num)`- This will return `evenly spaced` num of points within start to stop range.

```
np.linspace(0,5,10)
>>>
array([0.        , 0.55555556, 1.11111111, 1.66666667, 2.22222222,
       2.77777778, 3.33333333, 3.88888889, 4.44444444, 5.        ])
```

We can also create `identity matrix`. Identity matrix is used in linear algebra. Number of rows and columns are equal. All diagonal elements are 1 and others are 0.

```
np.eye(4)
>>>
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
```

Arrays of random elements.

```
np.random.rand(3,4)
>>>
array([[0.80940006, 0.9859906 , 0.26152869, 0.98569846],
       [0.5094053 , 0.22883829, 0.62847165, 0.27576971],
       [0.98594542, 0.38165645, 0.0266356 , 0.30627481]])
```

If we want to return sample array with `standard normal distribution` or `Gaussian distribution`, we can use `randn`.

This will return numbers from standard distribution centered around 0 instead of uniform distribution from 0 to 1.

```
np.random.randn(2)
>>>
array([0.62264086, 0.55017301])

np.random.randn(4,4)
>>>
array([[-0.34059302,  1.14487483,  2.22898505, -0.88188641],
       [-1.74616441, -0.10110956, -0.38235507, -0.69916654],
       [-0.79018516, -1.02698109,  0.12468266, -1.51640138],
       [ 0.78902964, -0.59662587,  0.26986646, -0.62834846]])
```

We can plot this numbers using data visualization.

`randint`- Returns random integers from low to a high number.

```
np.random.randint(1,100)
>>> 77

# Get 5 random from 1 to 100- Inclusive of 1 and exclusive to 100.
np.random.randint(1,100,5)
>>> array([55, 80, 55, 43, 22])
```

**Attributes and methods on array**  

`reshape`- Reshape existing array to new dimension.

```
arr = np.arange(25)
arr
>>>
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])

arr.reshape(5,5)
>>>
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
```

`max()` - Max value in array  
`min()` - Min value in array  
`argmax()` - Index of Max value in array  
`argmin` - Index of Min value in array  
`shape` - Returns Shape of array  
`dtype` - Datatype in array  

```
ranarr = np.random.randint(0,50,10)
ranarr
>>>
array([26,  5, 11, 36, 23,  6,  2,  5, 15, 47])

ranarr.max()
>>> 47

ranarr.min()
>>> 2

ranarr.argmax()
>>> 9

ranarr.argmin()
>>> 6

arr.shape

arr.dtype
>>>
dtype('int32')

from numpy.random import randint
randint(2,10)
>>> 3
```

## NumPy Array Indexing  
Array indexing starts at 0 in NumPy, similar to Python indexing.

```
import numpy as np
arr = np.arange(0,11)
arr
>>> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr[8]
>>> 8

arr[0:5]
>>> array([0, 1, 2, 3, 4])

# FROM START TO UPTO INDEX BUT Not Inclusive
arr[:6]
>>> array([0, 1, 2, 3, 4, 5])

# FROM INDEX TO REST OF ARRAY
arr[5:]
>>> array([ 5,  6,  7,  8,  9, 10])
```

NumPy array differs from Python list, because of their ability to broadcast. When we slice or take some elements from original arrays and make change in them, then it changes the original arrays as well. It is just a view of original array not the copy.

In order to create a copy of array, we need to use copy method on array.

```
arr
>>>
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

slice_arr = arr[0:5]
slice_arr
>>>
array([0, 1, 2, 3, 4])

# BROADCAST to 99
slice_arr[:] = 99

slice_arr
>>>
array([99, 99, 99, 99, 99])

#Original array is changed.
arr
>>> 
array([99, 99, 99, 99, 99,  5,  6,  7,  8,  9, 10])
```

Here original array is changed because of change in the slice array. This is because, it is only view to original array, not the copy of the array.

We could use `arr.copy()` method to get copy, so that original array will not be affected when we make changes in sliced or copied array.

```
arr
>>> array([99, 99, 99, 99, 99,  5,  6,  7,  8,  9, 10])

arr_copy = arr.copy()
arr_copy
>>> array([99, 99, 99, 99, 99,  5,  6,  7,  8,  9, 10])

arr_copy[:] = 100
arr_copy
>>> array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])

arr
>>> array([99, 99, 99, 99, 99,  5,  6,  7,  8,  9, 10])
```

In 2-dimensional array, we have 2 ways to grab the element using index.

```
arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
arr_2d

>>>
array([[ 5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])

arr_2d[0][1]  # Using Double Bracket
>>> 10

arr_2d[0,1]   # Using Single Bracket
>>> 10
```

We can grab sub matrices using colon(:) same as array indexing.

```
array([[ 5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])

arr_2d[1:]
>>>
array([[20, 25, 30],
       [35, 40, 45]])

arr_2d[:2,1:]
>>>
array([[10, 15],
       [25, 30]])

arr_2d[1:,:2]
>>>
array([[20, 25],
       [35, 40]])
```

Comparision operator on the array will return the boolean array.

```
arr = np.arange(1,11)
arr
>>> array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

bool_arr = arr > 5
bool_arr
>>>
array([False, False, False, False, False,  True,  True,  True,  True,
        True])

# Now we can filter the original array
arr[bool_arr]
>>> array([ 6,  7,  8,  9, 10])


# In practice, we will do as below instead of all above steps. They are same steps, but in shorthand
arr[arr > 5]
>>> array([ 6,  7,  8,  9, 10])

arr[arr < 3]
>>> array([1, 2])
```

## NumPy Operations  
We can do -
- Array with Array  
- Array with Scalars  
- Universal Array Functions  

```
import numpy as np
arr = np.arange(0,11)

# Array with Array operation like add, substract, multiply
arr
>>> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr + arr
>>> array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20])

arr - arr
>>> array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

arr * arr
>>> array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100])

# Array with Scalar like add 100 to each element of the array. The operation is broadcast to each element in the array

arr + 100
>>> array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

arr - 100
>>> array([-100,  -99,  -98,  -97,  -96,  -95,  -94,  -93,  -92,  -91,  -90])

arr * 100
>>> array([   0,  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000])

arr ** 2
>>> array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100], dtype=int32)

```

**Univeral array functions**- NumPy comes with many universal array function which is used to perform operations on array on each elements.

```
np.sqrt(arr)
>>>
array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ,
       3.16227766])

np.exp(arr)
>>> array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
       5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
       2.98095799e+03, 8.10308393e+03, 2.20264658e+04])

np.min(arr)
>>> 0

np.max(arr)
>>> 10

np.sin(arr)
>>> array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849,
       -0.54402111])
```

# Python for Data Analysis - Pandas
- Pandas is an open source library built on top of NumPy.  
- It allows for fast analysis and data cleaning and preparation.  
- It excels in performance and productivity.  
- It also has built-in visualization features.  
- It can work with data from a wide variety of sources.  

We can install pandas as below -

```
conda install pandas
#OR
pip install pandas
```

## Series
Series is similar to NumPy array. It is built on the top of NumPy array object. In Series we access data by labels instead of index as we do in arrays.

We can create Series using list, numpy arrays, dictionaries and can assign labels as well.

Series can hold any type of data.

```
import numpy as np
import pandas as pd

# CREATE 4 Different DataTypes
labels = ['a','b','c']
my_data = [10,20,30]
arr = np.array(my_data)
d = {'a':10,'b':20,'c':30}

# HERE SERIES WILL ASSIGN DEFAULT INDEX
pd.Series(data = my_data)
>>>
0    10
1    20
2    30
dtype: int64

# WE CAN ASSIGN LABELS TO THE SERIES
pd.Series(data=my_data, index=labels)
>>>
a    10
b    20
c    30
dtype: int64

# WE CAN PASS IN ORDER INSTEAD OF ASSIGNMENTS
pd.Series(my_data,labels)
>>>
a    10
b    20
c    30
dtype: int64

# CREATE SERIES FROM NUMPY ARRAYS
pd.Series(arr)
>>>
0    10
1    20
2    30
dtype: int32

# ASSIGN LABELS
pd.Series(arr, labels)
>>>
a    10
b    20
c    30
dtype: int32

# CREATE SERIES FROM DICTIONARIES
pd.Series(d)
>>>
a    10
b    20
c    30
dtype: int64

# SERIES CAN HOLD ANY TYPE OF DATA
pd.Series(data=labels)
>>>
0    a
1    b
2    c
dtype: object
```

Grabbing information out of series looks similar to getting it from dictionaries. Padas builds series similar to hashtable. Lookups are very fast.

```
ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
ser1
>>>
USA        1
Germany    2
USSR       3
Japan      4
dtype: int64

ser2 = pd.Series([1,2,5,4],['USA','Germany','Italy','Japan'])
ser2
>>>
USA        1
Germany    2
Italy      5
Japan      4
dtype: int64

ser1['USA']
>>> 1

ser3 = pd.Series(data=labels)
ser3
>>>
0    a
1    b
2    c
dtype: object

ser3[1]
>>> 'b'

# OPERATION ON SERIES. IT WILL ADD VALUES FROM BOTH SERIES. WHICHEVER IS NOT MATCHED WILL RETURN NaN.
ser1 + ser2
>>>
Germany    4.0
Italy      NaN
Japan      8.0
USA        2.0
USSR       NaN
dtype: float64
```

## DataFrames
DataFrames are datatype for storing tables like formats.

```
import numpy as np
import pandas as pd
from numpy.random import randn

np.random.seed(101)

# CREATE DATAFRAME
df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
df
>>>

    W	        X	        Y	        Z
A	2.706850	0.628133	0.907969	0.503826
B	0.651118	-0.319318	-0.848077	0.605965
C	-2.018168	0.740122	0.528813	-0.589001
D	0.188695	-0.758872	-0.933237	0.955057
E	0.190794	1.978757	2.605967	0.683509
```
Each of these columns is a series. So, W,X,Y,Z is a series with A,B,C,D,E index.

```
df['W']
>>>
A    2.706850
B    0.651118
C   -2.018168
D    0.188695
E    0.190794
Name: W, dtype: float64

type(df['W'])
>>> pandas.core.series.Series

type(df)
>>> pandas.core.frame.DataFrame

# WE CAN GRAB SERIES AS BELOW AS WELL.
df.W
>>>
A    2.706850
B    0.651118
C   -2.018168
D    0.188695
E    0.190794
Name: W, dtype: float64
```

We can also get multiple columns from DataFrame. If we get single column, it will return a `Series`, but if we ask for multiple columns, we will get `DataFrame`.

```
# PASS MULTIPLE COLUMNS AS LIST
df[['W','Z']]
>>>

    W	        Z
A	2.706850	0.503826
B	0.651118	0.605965
C	-2.018168	-0.589001
D	0.188695	0.955057
E	0.190794	0.683509
```

We can create new column using existing columns as below-

```
# CREATE NEW COLUMN AS ADDITION OF OTHER 2
df['new'] = df['W'] + df['Y']
df
>>>
    W	        X	        Y	        Z	        new
A	2.706850	0.628133	0.907969	0.503826	3.614819
B	0.651118	-0.319318	-0.848077	0.605965	-0.196959
C	-2.018168	0.740122	0.528813	-0.589001	-1.489355
D	0.188695	-0.758872	-0.933237	0.955057	-0.744542
E	0.190794	1.978757	2.605967	0.683509	2.796762
```

We can also remove columsn using `drop` method on DataFrames. When we drop column from DataFrame, by default it will not drop from original DataFrame, we have to sepcify that to drop `inplace=True` to drop from DataFrame.
We also have to sepcify the axis=1 for dropping column.

```
df.drop('new', axis=1, inplace=True)
df
>>>

    W	        X	        Y	        Z
A	2.706850	0.628133	0.907969	0.503826
B	0.651118	-0.319318	-0.848077	0.605965
C	-2.018168	0.740122	0.528813	-0.589001
D	0.188695	-0.758872	-0.933237	0.955057
E	0.190794	1.978757	2.605967	0.683509
```

We can also drop the row. We dont need to specify axis for dropping row, as default axis=0.

```
df.drop('E')
>>>
    W	        X	        Y	        Z
A	2.706850	0.628133	0.907969	0.503826
B	0.651118	-0.319318	-0.848077	0.605965
C	-2.018168	0.740122	0.528813	-0.589001
D	0.188695	-0.758872	-0.933237	0.955057
```

DataFrames are just fancy index markers on the top of NumPy array. Rows are refered as axis=0 and columns are referred as axis=1, because when we do `df.shape`, it will return a tuple (5,4), so we have 5 rows and 4 columns and there index is 0 and respectively.

When selecting columns, we pass the list of columns or single column name. When selecting rows, we can do this 2 ways.

`df.loc['A']` - This will returns series. So it confirms not only columns are series but rows are also series in pandas.  
`df.iloc[0]` - This will grab series based on the index location.

```
df.loc['A']
>>>
W    0.302665
X    1.693723
Y   -1.706086
Z   -1.159119
Name: A, dtype: float64

df.iloc[0]
>>>
W    0.302665
X    1.693723
Y   -1.706086
Z   -1.159119
Name: A, dtype: float64

# SELECTING PERTICULAR CELL - row and column
df.loc['B','Y']
>>> 0.16690463609281317

# GET SUBSET OF DATA - PASS LIST OF rows and LIST OF columns
df.loc[['A','B'],['W','Y']]
>>>

    W	        Y
A	0.302665	-1.706086
B	-0.134841	0.166905
```

We can also do conditional selection on DataFrame. This is similar to the NumPy. When we do comparision operator, it will return True and False in the results.
```
df > 0
>>>
	W	    X	    Y	    Z
A	True	True	False	False
B	False	True	True	True
C	True	True	True	True
D	False	False	False	True
E	False	True	True	True

df[df>0]
>>>
	W	        X	        Y	        Z
A	0.302665	1.693723	NaN	        NaN
B	NaN	        0.390528	0.166905	0.184502
C	0.807706	0.072960	0.638787	0.329646
D	NaN	        NaN	        NaN	        0.484752
E	NaN	        1.901755	0.238127	1.996652

# GET CONDITIONAL ROW SELECTION
df['W']>0
>>>
A     True
B    False
C     True
D    False
E    False
```

When we pass condition based on the columns, then we will get all filtered values which are True, we won't get any False value.

```
df[df['W']>0]
>>>
    W	        X	        Y	        Z
A	0.302665	1.693723	-1.706086	-1.159119
C	0.807706	0.072960	0.638787	0.329646

df[df['Z']<0]
>>>
    W	        X	        Y	        Z
A	0.302665	1.693723	-1.706086	-1.159119

# COMBINE MULTIPLE FILTERS
df[df['W']>0]['X']
>>>
A    1.693723
C    0.072960
Name: X, dtype: float64

# Grab multiple columns
df[df['W']>0][['X','Y']]
>>>
    X	        Y
A	1.693723	-1.706086
C	0.072960	0.638787
```

When we want to use logical operators to get results from multiple conditions, we can't use python `and` because python gets confused, as we pass series as input instead of single boolean values.

We need to use and(`&`), or(`|`)

```
# This will return error as we are doing logical and with series, which creates ambiguous results.
df[(df['W']>0) and (df['Y']>10)]

# We need to use &
df[(df['W']>0) & (df['Y']>0)]
>>>
	W	        X	    Y	        Z
C	0.807706	0.07296	0.638787	0.329646

df[(df['W']>0) | (df['Y']>0)]
>>>
    W	        X	        Y	        Z
A	0.302665	1.693723	-1.706086	-1.159119
B	-0.134841	0.390528	0.166905	0.184502
C	0.807706	0.072960	0.638787	0.329646
E	-0.116773	1.901755	0.238127	1.996652
```


Now lets take look at the reseting the index and seeting it's values. `df.reset_index()` will reset the index back to numbers and add index column for old values. It is not inplace operation. We need to specify explicitly to make it inplace.

```
df.reset_index()
>>>
    index	W	    X	        Y	        Z
0	A	0.302665	1.693723	-1.706086	-1.159119
1	B	-0.134841	0.390528	0.166905	0.184502
2	C	0.807706	0.072960	0.638787	0.329646
3	D	-0.497104	-0.754070	-0.943406	0.484752
4	E	-0.116773	1.901755	0.238127	1.996652
```

We can also set the index to new values as below. We will create new column states and add it in the DataFrame.
We will use `set_index(column_name)`. This will override the old index with new one.

```
newind = 'CA NY WY OR CO'.split()
df['states'] = newind
df
>>>
	W	        X	        Y	        Z	        states
A	0.302665	1.693723	-1.706086	-1.159119	CA
B	-0.134841	0.390528	0.166905	0.184502	NY
C	0.807706	0.072960	0.638787	0.329646	WY
D	-0.497104	-0.754070	-0.943406	0.484752	OR
E	-0.116773	1.901755	0.238127	1.996652	CO

df.set_index('States')
>>>
States  W	        X	        Y	        Z	        states					
CA	    0.302665	1.693723	-1.706086	-1.159119	CA
NY	    -0.134841	0.390528	0.166905	0.184502	NY
WY	    0.807706	0.072960	0.638787	0.329646	WY
OR	    -0.497104	-0.754070	-0.943406	0.484752	OR
CO	    -0.116773	1.901755	0.238127	1.996652	CO
```

**MultiIndex and Index Hierarchy** - 

```
# INDEX LEVELS
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
#CREATE A LIST OF TUPLES
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

# CREATE MULTI-LEVEL DATAFRAME
df = pd.DataFrame(randn(6,2), hier_index,['A','B'])

df
>>>
        A	        B
G1	1	-0.993263	0.196800
    2	-1.136645	0.000366
    3	1.025984	-0.156598
G2	1	-0.031579	0.649826
    2	2.154846	-0.610259
    3	-0.755325	-0.346419

df.loc['G1'].loc[1]
>>>
A   -0.993263
B    0.196800
Name: 1, dtype: float64
```

We can name these levels of rows-

```
# DATATYPES OF INDEX IN PANDAS
df.index.names
>>> FrozenList([None, None])

# NOW SET THE NEW DATATYPES
df.index.names = ['Groups','Num']
df
>>>
Groups	Num A	        B		
G1	    1	-0.993263	0.196800
        2	-1.136645	0.000366
        3	1.025984	-0.156598
G2	    1	-0.031579	0.649826
        2	2.154846	-0.610259
        3	-0.755325	-0.346419

# SELECT PERTICULAR VALUE
df.loc['G2'].loc[2]['A']
>>> 2.154846443259472
```

`Cross-section(xs)` method in the dataframe. Lets value of row 1 from level Num.
```
df.xs(1,level='Num')
>>>
Groups	A	        B	
G1	    -0.993263	0.196800
G2	    -0.031579	0.649826
```

## Missing Data
When we are working with Pandas and data, if the data is missing, then Pandas can fill in the data or drop columns or rows having missing values.

`dropna()` - Drop rows having missing values.  
`dropna(axis=1)` - Drop columns having missing values.  
`fillna(value='FILL VALUE')` - Fill missing values with specified values

```
import numpy as np
import pandas as pd
d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df = pd.DataFrame(d)
df
>>>
    A	B	C
0	1.0	5.0	1
1	2.0	NaN	2
2	NaN	NaN	3

df.dropna()
>>>
    A	B	C
0	1.0	5.0	1

df.dropna(axis=1)
>>>
	C
0	1
1	2
2	3

# SET THRESHOLD IN ORDER TO DROP THE ROW OR COLUMN
df.dropna(thresh=2)
>>>
    A	B	C
0	1.0	5.0	1
1	2.0	NaN	2

#FILL MISSING VALUES
df.fillna(value='FILL VALUE')
>>>
	A	        B	        C
0	1	        5	        1
1	2	        FILL VALUE	2
2	FILL VALUE	FILL VALUE	3
```

## GroupBy
GroupBy allows you to group together rows based off of a column and perform aggregate function on them. Aggregate functions takes series of values and output one value.

Lets groupBy on DataFrame-

```
import numpy as np
import pandas as pd

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
        'Person': ['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
        'Sales': [200,120,340,124,243,350]}

df = pd.DataFrame(data)
df
>>>
    Company	Person	Sales
0	GOOG	Sam	    200
1	GOOG	Charlie	120
2	MSFT	Amy	    340
3	MSFT	Vanessa	124
4	FB	    Carl	243
5	FB	    Sarah	350

# GroupBy on Column Company
byComp = df.groupby('Company')

# Mean
byComp.mean()
>>>
        Sales
Company	
FB	    296.5
GOOG	160.0
MSFT	232.0

# Sum
byComp.sum()
>>>
	    Sales
Company	
FB	    593
GOOG	320
MSFT	464

# Standard Deviation
byComp.std()
>>>
	    Sales
Company	
FB	    75.660426
GOOG	56.568542
MSFT	152.735065

# GET SUM OF FB
byComp.sum().loc['FB']
>>> Sales    593

# COUNT
df.groupby('Company').count()
>>>
	    Person	Sales
Company		
FB	    2	    2
GOOG	2	    2
MSFT	2	    2

# Max and Min

# Describe to get the summary of everything in DataFrame
df.groupby('Company').describe()
>>>
        Sales
	    count	mean	std	        min	    25%	    50%	    75%	    max
Company								
FB	    2.0	    296.5	75.660426	243.0	269.75	296.5	323.25	350.0
GOOG	2.0	    160.0	56.568542	120.0	140.00	160.0	180.00	200.0
MSFT	2.0	    232.0	152.735065	124.0	178.00	232.0	286.00	340.0

df.groupby('Company').describe().transpose()
>>>
        Company	FB	        GOOG	    MSFT
Sales	count	2.000000	2.000000	2.000000
        mean	296.500000	160.000000	232.000000
        std	    75.660426	56.568542	152.735065
        min	    243.000000	120.000000	124.000000
        25%	    269.750000	140.000000	178.000000
        50%	    296.500000	160.000000	232.000000
        75%	    323.250000	180.000000	286.000000
        max	    350.000000	200.000000	340.000000
```

## Merging, Joining and Concatenating
pd.merge
pd.join
pd.concat

## Operations

```
import numpy as np
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],
                   'col2':[444,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()
>>>
    col1	col2	col3
0	1	    444	    abc
1	2	    555	    def
2	3	    666	    ghi
3	4	    444	    xyz
```

**Unique** - Getting unique values or number of unique values. We can use `unique` or `nunique`.

```
df['col2'].unique()
>>> array([444, 555, 666], dtype=int64)

df['col2'].nunique()
>>> 3
```

**value_counts** - How many times each value appeared in the result.
```
df['col2'].value_counts()
>>>
444    2
555    1
666    1
Name: col2, dtype: int64

df[df['col1']>2]
>>>
	col1	col2	col3
2	3	    666	    ghi
3	4	    444	    xyz

df[(df['col1']>2) & (df['col2']==444)]
>>>
    col1	col2	col3
3	4	    444	    xyz
```

**apply** - Broadcast a function to each element int the DataFrame. We can create a function and apply it to entire DataFrame or single column.

This is the most powerful functions in pandas.

```
def times2(x):
    return x*2

df.apply(times2)

	col1	col2	col3
0	2	    888	    abcabc
1	4	    1110	defdef
2	6	    1332	ghighi
3	8	    888	    xyzxyz

# APPLY a LAMBDA Expression
df['col3'].apply(lambda x: x*2)
>>>
0    abcabc
1    defdef
2    ghighi
3    xyzxyz
Name: col3, dtype: object
```

**drop** - Remove column

```
df.drop('col1', axis=1)
>>>
	col2	col3
0	444	    abc
1	555	    def
2	666	    ghi
3	444	    xyz

df.columns
>>>
Index(['col1', 'col2', 'col3'], dtype='object')

df.index
>>>
RangeIndex(start=0, stop=4, step=1)
```

**sort_values(by='col')** - Sort Values by column names.
```
df.sort_values(by='col2')
>>>
    col1	col2	col3
0	1	    444	    abc
3	4	    444	    xyz
1	2	    555	    def
2	3	    666	    ghi
```

**isnull()** - Return True if null, else False.
```
df.isnull()
>>>
    col1	col2	col3
0	False	False	False
1	False	False	False
2	False	False	False
3	False	False	False
```

**pivot_table** - create new multiindex DataFrame from the dataframe by specifying value, index, and columns in the new pivot_table.
```
data = {'A':['foo','foo','foo','bar','bar','bar'],
        'B':['one','one','two','two','one','one'],
        'C':['x','y','x','y','x','y'],
        'D':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
df
>>>
	A	B	C	D
0	foo	one	x	1
1	foo	one	y	3
2	foo	two	x	2
3	bar	two	y	5
4	bar	one	x	4
5	bar	one	y	1

df.pivot_table(values='D',index=['A','B'], columns=['C'])
>>>
	C	x	y
A	B		
bar	one	4.0	1.0
    two	NaN	5.0
foo	one	1.0	3.0
    two	2.0	NaN
```

## Data Input and Output
Pandas has abilities to read or write data for wide variety of data sources like CSV, Excel, HTMl, SQL etc.
In order to work with HTML and SQl, we need to install below libraries.

```
conda install sqlalchemy
conda install lxml
conda install html5lib
conda install BeautifulSoup4
```

We can read data from `csv` files -

```
import pandas as pd

pd.read_csv('example')
>>>
    a	b	c	d
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15
```

In order to write to a file, we need DataFrame, so lets read and write.
```
df = pd.read_csv('example')
df
>>>
    a	b	c	d
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15

# WRITE - WE DON'T WANT TO OUTPUT THE INDEX AS A COLUMN
df.to_csv('My_output', index=False)
```

We can also read `excel` file. While reading excel, we dont read formulas and formating in the excel. We just do the data. Pandas considers each sheet as a DataFrame.

```
pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
>>>

    a	b	c	d
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15

df.to_excel('Excel_Sample2.xlsx',sheet_name='NewSheet')
pd.read_excel('Excel_Sample2.xlsx')
>>>
	a	b	c	d
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15
```

For reading from `Html`, we need to install additional libraries. We will read data from Fedral government website for the failed banks- https://www.fdic.gov/bank/individual/failed/banklist.html

Pandas basically tries to read the tables markings from the html page and store it in the `list`.

```
data = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')

# RETURNS list as it contains many tables
type(data)
>>> list

# SHOW DATA
data[0]
>>>

    Bank Name	City	ST	CERT	Acquiring Institution	Closing Date	Updated Date
0	Washington Federal Bank for Savings	Chicago	IL	30570	Royal Savings Bank	December 15, 2017	February 21, 2018
1	The Farmers and Merchants State Bank of Argonia	Argonia	KS	17719	Conway Bank	October 13, 2017	February 21, 2018
2	Fayette County Bank	Saint Elmo	IL	1802	United Fidelity Bank, fsb	May 26, 2017	July 26, 2017
```

Read or writing data from `SQL`. We should always use Database specific SQL engine libray for connecting with specific Database and perform SQl for that flavor of DB.

Here we will create simple sqlite engine and interact with it in memory.

```
from sqlalchemy import create_engine

# CREATE SQLITE DATABASE RUNNING IN MEMORY
engine = create_engine('sqlite:///:memory:')

# WRITE TO A TABLE USING THE ENGINE
df.to_sql('my_table',engine)

# READ FROM THE DATABASE
sqldf = pd.read_sql('my_table', con=engine)

sqldf
>>>
	index	a	b	c	d
0	0	    0	1	2	3
1	1	    4	5	6	7
2	2	    8	9	10	11
3	3	    12	13	14	15
```

## Pandas Exercise

### SF Salary 
See the code
### Ecommerce Purchase
See the code

# Python for Data Visualization - Matplotlib
MatplotLib is the most popular plotting libraries for Python. It was designed to have similar feel to MATLAB's graphical plotting library.

We can install using -
```
conda install matplotlib
#OR
pip install matplotlib
```

We can use the matplotlib in the Jupyter notebook. There are 2 ways to create matplotlib-  
- Function method  way
- Object Oriented way  

When we are not using Jupyter notebook, we need to explicitely call `plot.show()` method.

```python
# IMPORT
import matplotlib.pyplot as plt

# ENABLE TO SEE MATPLOTLIB INSIDE JUPYTER NOTEBOOK
%matplotlib inline

# USING NUMPY CREATE DATAPOINTS
import numpy as np
x = np.linspace(0,5,11)
y = x ** 2
x
>>> array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])
y
>>> array([ 0.  ,  0.25,  1.  ,  2.25,  4.  ,  6.25,  9.  , 12.25, 16.  , 20.25, 25.  ])

# FUNCTIONAL WAY
plt.plot(x,y)
>>> [<matplotlib.lines.Line2D at 0x23ee0bf0b38>]
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/first.png?raw=true "Plot")
```python
#SUBPLOT TO PLOT MULTIPLE PLOTS
plt.subplot(1,2,1)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')
>>> [<matplotlib.lines.Line2D at 0x23ee1340d68>]
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/subplot.png?raw=true "Plot")

Above is the functional way to create plots. Lets dive into Object oriented way to creating plottings. In Object Oriented ploting, we have more control over the placings of the graphs. We create a figure and add axes to it.

```python
# OBJECT ORIENTED METHOD
fig = plt.figure()

# CREATE CANVAS BY GIVING AXES - LEFT, BOTTOM, WIDTH, HEIGHT
axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')

>>> Text(0.5,1,'Title')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/first.png?raw=true "Plot")
```python
#PLACING THE PLOTS WITHIN
fig = plt.figure()

axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y,'r')
axes1.set_title('LARGER PLOT')
axes2.plot(y,x,'b')
axes2.set_title('SMALLER PLOT')
>>> Text(0.5,1,'SMALLER PLOT')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/oopsmall.png?raw=true "Plot")
```python
#SUBPLOTS
fig,axes = plt.subplots(nrows=1,ncols=2)

# AXIS IS THE ARRAYS OF MATPLOTLIB AXIS OBJECTS
axes[0].plot(x,y)
axes[0].set_title('First Plot')
axes[1].plot(y,x)
axes[1].set_title('Second Plot')

# TO AVOID OVERLAPPING
plt.tight_layout()
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/subplot2.png?raw=true "Plot")

### Figure Size, Aspect Ratio and DPI -
Matplotlib allows you to control these things. We can do this for plot and subplots.

```
fig = plt.figure(figsize=(8,2), dpi=100)

ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/dpi.png?raw=true "Plot")
```python
# ADD LEGEND
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x,x**2,label='X Squared')
ax.plot(x,x**3,label='X Cubed')

ax.legend() # WE CAN PASS THE LOCATION NUMBER AS WELL
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/legend.png?raw=true "Plot")

### Customizing Options - Settings Lines Colors, Lines sizes

```python
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

# color - WE CAN ADD COLOR OR RGB HEX CODE AS WELL
# linewidth(lw) - Default is 1
# alpha(0 to 1) - controls, how transparent line is.
# linestyle(ls) - Linestyle whether dash(--),dash-dotted(-.), steps
# marker - marker on line. We can show o,+,1
# markersize - size of marker on line
# markerfacecolor - 
# markeredgewidth -
# markeredgecolor -
ax.plot(x,y,color='purple',lw=3, ls='-',marker='o',markersize=10)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/plotstyle.png?raw=true "Plot")

### Control over axis appearance

```
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

#CONTROL OVER AXIS APPEARANCE

ax.plot(x,y,color='purple',lw=2, ls='--')
ax.set_xlim([0,1])
ax.set_ylim([0,2])
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/04.Python-for-Data-Visualization-Matplotlib/plots/axiscontrl.png?raw=true "Plot")

## Matplotlib exercise
See the code

# Python for Data Visualization - Seaborn
Seaborn is a `statistical plotting` library and it is built on the top of matplotlib. It has beautiful default sytle. It is designed to work well with Pandas DataFrame objects.

We can install using -

```python
conda install seaborn
# OR
pip install seaborn
```

### Distribution Plots
Distribution plots helps to show the ditribution of datasets using plots. Seaborn comes with some sample datasets. We will use those datasets to plot the graphs.

```python
import seaborn as sns
%matplotlib inline

# LOAD tips DataFrame
tips = sns.load_dataset('tips')
tips.head()
>>>
    total_bill	tip	    sex	    smoker	day	time	size
0	16.99	    1.01	Female	No	    Sun	Dinner	2
1	10.34	    1.66	Male	No	    Sun	Dinner	3
2	21.01	    3.50	Male	No	    Sun	Dinner	3
3	23.68	    3.31	Male	No	    Sun	Dinner	2
4	24.59	    3.61	Female	No	    Sun	Dinner	4
```

**distplot()** - Shows distribution of uni-variable(only one variable) of ditribution.

```Python
# For total_bill check the distribution
# kde - 
# bins - Number of bins
sns.distplot(tips['total_bill'],kde=False,bins=40)

# Plot below tells, most of the bills are between 10 and 20 dollars
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/distplot.png?raw=true "Plot")

**jointplot** - Jointplot will join 2 columns and show the distribution against them. For example, we want to check distribution of tips against the total_bill.

```python
# DISTRIBUTION OF TIPS on TOTAL_BILL
# HIGHER THE BILL, HIGER THE TIPS.
# kind - By Default scatter, hex, reg(linear regression), kde(density)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/joinplot.png?raw=true "Plot")

**pairplot** - Plot pairwise relationship across entire DataFrame, atleast for the numerical columns. It basically does joinplot for everypossible combination of numerical columns on the DataFrame.

It also accepts `hue` arguments in which we will pass the categorial column. For example - Sex - Male/Female. It then will show coloring for the categories of the data.

```
sns.pairplot(tips)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/pairplot.png?raw=true "Plot")

```python
# With hue and palette
sns.pairplot(tips,hue='sex',palette='coolwarm')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/pairplot_hue.png?raw=true "Plot")

**rugplot** - Similar to distplot but instead of histogram, it plot the rug counts of the data.

```python
sns.rugplot(tips['total_bill'])
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/rugplot.png?raw=true "Plot")

**kde-kernal density estimation** - Kde plot has relationship with rugplot. It is based on the rug counts.
```python
sns.kdeplot(tips['total_bill'])
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/kde.png?raw=true "Plot")

### Categorical Plots

```python
import seaborn as sns
%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
```

Categorical plots helps plot categories data against a estimated function like avg, sum, std.

**barplot** - Plot for categorical data against numerical data. We can think of it as visualization of groupBy action.
Default estimator function is average. We can change it to anything, we want to set.
```python
# Average by Default
sns.barplot(x='sex',y='total_bill',data=tips, estimator=np.std)

# Change the Estimator function
import numpy as np
sns.barplot(x='sex',y='total_bill',data=tips, estimator=np.std)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/barplot.png?raw=true "Plot")

**countplot** - Countplot is similar to barplot, but estimator is explicitly counting the number of occurances. We need to set only x values.
```python
sns.countplot(x='sex',data=tips)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/countplot.png?raw=true "Plot")

**boxplot** - It shows distribution of data across category. Here the dots are outliers, and it shows the total bill on each day.
```python
sns.boxplot(x='day',y='total_bill',data=tips)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/boxplot.png?raw=true "Plot")

Below example shows the box plot with another variations using hue. It shows that the total_bill and smoker relations
```python
sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/boxplot_hue.png?raw=true "Plot")

**violinplot** - Violin plot are similar to boxplot, they also show the distribution of data across categories.
```python
sns.violinplot(x='day',y='total_bill',data=tips)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/violinplot.png?raw=true "Plot")

**stripplot** - Stripplot will draw a scatter graph where one variable is categorical. We will not easily know the number of points stacked on eachother. To show more robust points, we need to add one argument - `jitter=True`. We also have `hue` and `split` parameters.
```python
sns.stripplot(x='day',y='total_bill',data=tips)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/stripplot.png?raw=true "Plot")

```python
sns.stripplot(x='day',y='total_bill',data=tips,jitter=True)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/stripplot_jitter.png?raw=true "Plot")

**swarmplot** - Swarmplot is combination of violin plot and stripplot. It shows more clear distribution of the data. Swarmplots are not good for large number of datasets, as points might go haywire and will not show correct distribution.
```python
sns.swarmplot(x='day',y='total_bill',data=tips)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/swarmplot.png?raw=true "Plot")

**factorplot** - Factor plot will show plot based on the `kind` parameter. Kind paramters will tell which type of plot to disply.
kind - bar,violin,strip, box etc.
```python
sns.factorplot(x='day',y='total_bill',data=tips,kind='bar')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/factorplot_bar.png?raw=true "Plot")

### Matrix Plots
Matrix plots helps us plot the matrix data, primararily heatmaps of the data.

**heatmap** -  

```python
import seaborn as sns
%matplotlib inline
# LOAD TIPS DATA
tips = sns.load_dataset('tips')
tips.head(5)
>>>
	total_bill	tip	sex	smoker	day	time	size
0	16.99	1.01	Female	No	Sun	Dinner	2
1	10.34	1.66	Male	No	Sun	Dinner	3
2	21.01	3.50	Male	No	Sun	Dinner	3
3	23.68	3.31	Male	No	Sun	Dinner	2
4	24.59	3.61	Female	No	Sun	Dinner	4

#LOAD FLIGHTS DATA
flights = sns.load_dataset('flights')
flights.head(5)
>>>
    year	month	    passengers
0	1949	January	    112
1	1949	February	118
2	1949	March	    132
3	1949	April	    129
4	1949	May	        121
```
In order to show the correct heatmaps, the data should be in the correct matrix format. Means, it should have relationship between column and rows. So, we will first convert tips data to `pivot_table` form. Once data is in the matrix form, we can call heatmaps on the data, so it will show the related data with some coeficiant color, so we know which data in the matrix colsely relates to eachother.
```python
tc = tips.corr()
tc
>>>
	        total_bill	tip	        size
total_bill	1.000000	0.675734	0.598315
tip	        0.675734	1.000000	0.489299
size	    0.598315	0.489299	1.000000

# CALL HEATMAP
sns.heatmap(tc)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/heatmap.png?raw=true "Plot")

We can also pass the annotation, which will annotate the color with actual numerical values
```python
sns.heatmap(tc, annot=True)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/heatmap_annot.png?raw=true "Plot")

Lets convert the flight data into a `pivot_table` of month/year with passengers as the values.
```python
fp = flights.pivot_table(index='month',columns='year',values='passengers')
fp
>>>
year	    1949    1950	1951	1952	1953	1954	1955	1956	1957	1958	1959	1960
month												
January	    112 115	145	171	196	204	242	284	315	340	360	417
February    118	126	150	180	196	188	233	277	301	318	342	391
March	    132	141	178	193	236	235	267	317	356	362	406	419
April	    129	135	163	181	235	227	269	313	348	348	396	461
May	        121	125	172	183	229	234	270	318	355	363	420	472

# PLOT HEATMAP
sns.heatmap(fp)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/flight_heatmap.png?raw=true "Plot")

This heatmap shows, as the year go by more people travel by plane and number of passengers increased. The most popular months are Jun, July, Aug. We can also add linecolor, linewidths to have more prominent display.

**clustermap** - Uses hierarchical clustering to produce a cluster version of the heatmap. It tries to cluster rows and columns based on the similarity.
```python
sns.clustermap(fp)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/clustermap.png?raw=true "Plot")

In above graph, we can see months and year are out of order and are clustered together.

### Grids
We will use iris data. Iris data is data related to the bunch of different flowers.
```python
import seaborn as sns
%matplotlib inline
iris = sns.load_dataset('iris')
iris.head()
>>>
	sepal_length	sepal_width	petal_length	petal_width	species
0	5.1	            3.5	        1.4	            0.2	        setosa
1	4.9	            3.0	        1.4	            0.2	        setosa
2	4.7	            3.2	        1.3	            0.2	        setosa
3	4.6	            3.1	        1.5	            0.2	        setosa
4	5.0	            3.6	        1.4	            0.2	        setosa

iris['species'].unique()
>>> array(['setosa', 'versicolor', 'virginica'], dtype=object)
```

**PairGrid** - Pariplot shows the distribution pair of the data across all possible combinations. Here we will create a `PairGrid`. Pariplot is basically a simplified version of pairgrid in which many of the things are automated. In PairGrid, we have more control over the disply.
We can choose what type of plot we want on the grid.

```python
sns.PairGrid(iris)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/PairGrid_init.png?raw=true "Plot")

Lets add bunch of different plots.
```python
g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/PairGrid_multiple.png?raw=true "Plot")

**FacetGrid** -
```python
tips = sns.load_dataset('tips') 
g = sns.FacetGrid(data=tips,col='time',row='smoker')
g.map(sns.distplot,'total_bill')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/facetgrid.png?raw=true "Plot")

### Regression Plots
Regression plots show the regression of data. **lmplot** will helps us show the data on the plots. Seaborn under the hood uses matplot libs regression plot. We can customized it by passing the keyward arguments.
```python
import seaborn as sns
%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','v'])
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/05.Python-for-Data-Visualization-Seaborn/plots/lmplot.png?raw=true "Plot")


# Python for Data Visualization - Pandas Built-in Data Viz
Pandas built in data visualization capablities are built on the top of matplotlib and allows to call data visualation on DataFrames.
Styles of these graphs are not as good as seaborn, so we should import seaborn, so they will look like seaborn styles.

```python
import numpy as np
import seaborn as sns
import pandas as pd

%matplotlib inline

df1 = pd.read_csv('df1',index_col=0)
df1.head()
>>>
	        A	        B	        C	        D
2000-01-01	1.339091	-0.163643	-0.646443	1.041233
2000-01-02	-0.774984	0.137034	-0.882716	-2.253382
2000-01-03	-0.921037	-0.482943	-0.417100	0.478638
2000-01-04	-1.738808	-0.072973	0.056517	0.015085
2000-01-05	-0.905980	1.778576	0.381918	0.291436

df2 = pd.read_csv('df2')
df2.head()
>>>
	a	        b	        c	        d
0	0.039762	0.218517	0.103423	0.957904
1	0.937288	0.041567	0.899125	0.977680
2	0.780504	0.008948	0.557808	0.797510
3	0.672717	0.247870	0.264071	0.444358
4	0.053829	0.520124	0.552264	0.190008

# PLOT HISTOGRAM - We can also pass matplotlib arguments
df1['A'].hist(bins=30)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/hist.png?raw=true "Plot")

We can also plot by using `plot` method and passing the `kind` argument.
```python
df1['A'].plot(kind='hist')
# OR
df1['A'].plot.hist() # WE WILL USE THIS METHOD THROUGHTOUT

# AREA PLOT
df2.plot.area()
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/area.png?raw=true "Plot")

```python
# BAR - INDEX(ROW) SHOULD be CATEGORICAL
df2.plot.bar()
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/bar.png?raw=true "Plot")

```python
# LINE PLOT
df1.plot.line()
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/line.png?raw=true "Plot")

```python
# SCATTER
df1.plot.scatter(x='A', y='B')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/scatter.png?raw=true "Plot")

```python
# Show 3 Dimensional scattered plot
df1.plot.scatter(x='A', y='B',c='C',cmap='coolwarm')
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/scatter_3D.png?raw=true "Plot")

```python
# BOX PLOT
df2.plot.box()
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/box.png?raw=true "Plot")

```python
# HEX PLOT
df = pd.DataFrame(np.random.randn(1000,2),columns=['a','b'])
df.head()
>>>
    a	        b
0	0.651909	1.989180
1	-1.128242	1.012460
2	1.837604	-0.324401
3	-0.101816	0.475093
4	-0.319067	-0.085784

df.plot.hexbin(x='a',y='b',gridsize=15)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/hexplot.png?raw=true "Plot")

```python
# KERNAL DENSITY ESTIMATION
df2.plot.kde()
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/06.Python-for-Data-Visualization-Pandas-Built-in-Data-Viz/plots/kde.png?raw=true "Plot")

# Python for Data Visualization - Plotly and Cufflinks
Plotly is a library that allows us to create interactive plots that we can use in dashboards or websites (you can save them as html files or static images). Cufflinks connects plotly with pandas.

Install plotly and cufflinks
```python
pip install plotly
pip install cufflinks
```

Plotly works offline as well as online. Please refer to the code notebook for more details.

# Python for Data Visualization - Geographical Plotting
Geographical plotting is usually challenging due to the various formats of the data come in. We will focus on `plotly` for plotting.Matplotlib also has `basemap` extension.

### USA and Nationwide data maps

We will plot the geographical map of USA states. We will need to construct the data and layout.

```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

# CREATE DATA
data = dict(type = 'choropleth',
           locations = ['AZ','CA','NY'],
           locationmode = 'USA-states',
           colorscale = 'Portland',
           text = ['text1','text2','text3'],
           z = [1.0,2.0,3.0],
           colorbar = {'title':'Colorbar Title Goes Here'})

# CREATE LAYOUT
layout = dict(geo = {'scope':'usa'})

# CREATE FIGURE AND INTERACTIVELY PLOT IT
choromap = go.Figure(data=[data],layout=layout)
iplot(choromap)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/08.Python-for-Data-Visualization-Geographical-Plotting/plots/usa-states.png?raw=true "Plot")

Lets use the USA 2011 export data to plot interactive data on the choreplot.

```python
import pandas as pd
df = pd.read_csv('2011_US_AGRI_Exports')

df.head()

	code	state	category	total exports	beef	pork	poultry	dairy	fruits fresh	fruits proc	total fruits	veggies fresh	veggies proc	total veggies	corn	wheat	cotton	text
0	AL	Alabama	state	1390.63	34.4	10.6	481.0	4.06	8.0	17.1	25.11	5.5	8.9	14.33	34.9	70.0	317.61	Alabama<br>Beef 34.4 Dairy 4.06<br>Fruits 25.1...
1	AK	Alaska	state	13.31	0.2	0.1	0.0	0.19	0.0	0.0	0.00	0.6	1.0	1.56	0.0	0.0	0.00	Alaska<br>Beef 0.2 Dairy 0.19<br>Fruits 0.0 Ve...

data = dict(type='choropleth',
           colorscale = 'YlOrRd',
           locations = df['code'],
           locationmode = 'USA-states',
           z = df['total exports'],
           text = df['text'],
           marker = dict(line = dict(color ='rgb(255,255,255)',width=2)),
           colorbar = {'title':'Millions USD'})

layout = dict(title='2011 US Agriculture Exports by State',
             geo = dict(scope='usa', showlakes = True, lakecolor = 'rgb(85,173,240)'))

choromap2 = go.Figure(data = [data], layout=layout)
iplot(choromap2)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/08.Python-for-Data-Visualization-Geographical-Plotting/plots/us-exports-2011.png?raw=true "Plot")

### International data maps
We can also plot the `world data` using the choropleth plotting. Lets plot data of world GDP.

```python
df = pd.read_csv('2014_World_GDP')
df.head()
>>>
    COUNTRY	        GDP (BILLIONS)	CODE
0	Afghanistan	    21.71	        AFG
1	Albania	        13.40	        ALB
2	Algeria	        227.80	        DZA
3	American Samoa	0.75	        ASM
4	Andorra	        4.80	        AND

data = dict(type='choropleth',
           locations = df['CODE'],
           z = df['GDP (BILLIONS)'],
           text = df['COUNTRY'],
           colorbar = {'title':'GDP in Billions USD'})

layout = dict(title = '2014 Global GDP',
             geo = dict(showframe = False,
                       projection = {'type':'mercator'}))

choromap3 = go.Figure(data=[data],layout=layout)
iplot(choromap3)
```
![Alt text](https://github.com/vaibhavpatilai/DataScience/blob/master/DataScience-MachineLearning-Py/Code/08.Python-for-Data-Visualization-Geographical-Plotting/plots/world-gdp.png?raw=true "Plot")

# Data Capstone Project
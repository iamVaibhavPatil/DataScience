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
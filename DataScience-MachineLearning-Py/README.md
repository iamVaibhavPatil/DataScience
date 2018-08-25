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

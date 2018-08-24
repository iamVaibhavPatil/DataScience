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
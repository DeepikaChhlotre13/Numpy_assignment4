#Q1.Explain the purpose and advantages of numpy in Scientific computing and data analysis.
# How does it enhance Python's capabilities for numerical operations?

'''NumPy, which stands for Numerical Python, is a powerful library in Python that is essential for
scientific computing and data analysis. Here’s an overview of its purpose and advantages:

Purpose of NumPy:

1.Array Management: NumPy introduces a new data structure called the ndarray (n-dimensional array),
                    which allows for efficient storage and manipulation of large datasets.
2.Numerical Computations: It provides a wide range of mathematical functions to perform operations
                          on arrays, enabling efficient numerical calculations.
3.Interoperability: NumPy serves as a foundation for many other scientific libraries (like SciPy,
                    pandas, and Matplotlib), enhancing their capabilities.

Advantages of NumPy:

1.Performance: NumPy operations are implemented in C and optimized for performance, making 
               significantly faster than native Python lists, especially for large datasets.
2.Vectorization: NumPy allows for vectorized operations, meaning you can perform operations on 
                 entire arrays without writing explicit loops. This leads to more concise and
                 readable code.
3.Multidimensional Arrays: It supports multi-dimensional arrays, enabling complex data structures
                           and mathematical operations that are not feasible with standard Python
                           lists.
4.Memory Efficiency: It uses less memory compared to traditional Python lists, making it better
                     suited for large datasets.

Enhancing Python’s Capabilities:

NumPy enhances Python’s capabilities for numerical operations by:

1.Efficient Computation: It allows for the efficient execution of complex calculations that would 
                         be cumbersome and slow in standard Python.

2.Ease of Use: It provides an intuitive syntax for array operations, making it accessible for
               users transitioning from other programming languages with similar functionalities
               (like MATLAB).

3.Integration with : Many scientific libraries depend on NumPy, allowing seamless 
  other libraries    integration and the ability to handle diverse types of data analyses, from 
                    statistics to machine learning.'''

#Q2.Compare and contrast np.mean()and np.average()functions in NumPy. When would you 
# use one over the other?

'''In NumPy, both np.mean() and np.average() are used to compute averages, but they serve slightly
different purposes and have distinct functionalities. Here’s a comparison:

np.mean()
Purpose: Computes the arithmetic mean of the input array elements.

Syntax: np.mean(a, axis=None, dtype=None, out=None)

np.average()
Purpose: Computes the weighted average of the input array elements. It can also compute the simple
         mean if no weights are provided.

Syntax: np.average(a, axis=None, weights=None, returned=False)

Use np.mean():

When you want a simple, fast calculation of the mean.
When all elements are equally significant.

Use np.average():
When you need to consider weights in your calculation.
When you want to get additional information about the weighting.'''

import numpy as np

data = np.array([1, 2, 3, 4, 5])

# Using np.mean()
mean_value = np.mean(data)  # Returns 3.0

# Using np.average() without weights
average_value = np.average(data)  # Also returns 3.0

# Using np.average() with weights
weights = np.array([1, 1, 1, 1, 5])  # Last element has higher weight
weighted_average = np.average(data, weights=weights)  # Returns 4.0


#Q3.Describe the methods of reversing a NumPy array along different axes. Provide example for 1D 
#   and 2D arrays.

'''Reversing a NumPy array can be done along different axes using slicing techniques. Let’s break
down the methods for reversing a 1D and 2D array along different axes.

1D Array Reversal
In a 1D array, there is only one axis (axis 0). You can reverse the entire array using slicing.

Example: Reversing a 1D Array

import numpy as np

# Create a 1D NumPy array
arr_1d = np.array([1, 2, 3, 4, 5])

# Reverse the array
reversed_1d = arr_1d[::-1]
print(reversed_1d)

Output:

[5 4 3 2 1]
2D Array Reversal
In a 2D array, you can reverse along the rows (axis 0) or along the columns (axis 1).

Reversing Along Axis 0 (Rows)
Reversing along axis 0 means flipping the rows in the array. This is done using slicing on the
first axis.

Example: Reversing Along Axis 0

# Create a 2D NumPy array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

#Reverse the array along rows (axis 0)
reversed_rows = arr_2d[::-1, :]
print(reversed_rows)

Output:

[[7 8 9]
 [4 5 6]
 [1 2 3]]

Reversing Along Axis 1 (Columns)

Reversing along axis 1 means flipping the columns in the array. This is done using slicing on
the second axis.

Example: Reversing Along Axis 1

# Reverse the array along columns (axis 1)
reversed_columns = arr_2d[:, ::-1]
print(reversed_columns)

Output:

[[3 2 1]
 [6 5 4]
 [9 8 7]]

Reversing Both Axes (Flipping Entire Array)

You can also reverse the array along both axes simultaneously, effectively flipping the array
both vertically and horizontally.

Example: Reversing Along Both Axes

# Reverse the array along both axes
reversed_both = arr_2d[::-1, ::-1]
print(reversed_both)

Output:

[[9 8 7]
 [6 5 4]
 [3 2 1]]'''


#Q4.How can you determine the data type of elements in NumPy array? Discuss the importance
#   of data types in memory management and performance. 

'''In NumPy, the data type of elements in an array can be determined using the .dtype attribute.
This provides the type of data stored in the array, which could be a type like int64, float32, 
complex128, etc. You can access it like this:


import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4])

# Determine the data type of the elements
print(arr.dtype)  # Output: int64 (or other type depending on the system)

Importance of Data Types in Memory Management

Memory Efficiency:

Different data types consume different amounts of memory. For example, an int32 uses 4 bytes of
memory, whereas an int64 uses 8 bytes. Choosing the appropriate data type helps in reducing memory
usage, which is especially important when working with large datasets or on systems with limited
memory.
For example, using float64 for an array that only needs float32 will unnecessarily double the 
memory usage.

Alignment and Storage:

NumPy arrays are designed to store homogeneous data, meaning all elements in an array have the 
same data type. This homogeneity ensures efficient storage and memory alignment, improving cache 
utilization.
Data types also determine the size of elements in the array and, consequently, the overall size of
the array in memory.'''


#Q5.Define ndarray in NumPy and explain their Key features. How do they differ from standard python
#   list.

'''An ndarray (short for n-dimensional array) is the core data structure in NumPy that represents a
grid of values, all of the same type, indexed by a tuple of non-negative integers. It is a
multidimensional container for homogeneous data, meaning that all elements in an ndarray must 
have the same data type. The ndarray is optimized for performance and memory efficiency, allowing you to perform mathematical operations on large datasets with minimal overhead.

Key Features of ndarray in NumPy

Homogeneous Data:

All elements in an ndarray have the same data type (e.g., all int, float, etc.). This ensures 
efficient memory usage and computational speed.
For example, you can create an array of integers using np.array([1, 2, 3]), and the dtype is int32
or int64 depending on your system.

Multidimensional:

An ndarray can represent arrays of any number of dimensions, including 1D, 2D, 3D, or higher-dimen
sional arrays. For example:

1D array: np.array([1, 2, 3])
2D array (matrix): np.array([[1, 2], [3, 4]])
3D array: np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

The shape of an ndarray (i.e., the number of elements along each dimension) is defined by the shape
attribute.

Efficient Memory Layout:

Ndarrays are stored in contiguous memory blocks, unlike Python lists which are arrays of pointers
to objects. This memory layout allows for faster access and manipulation of elements.
the elements are laid out in memory in a way that facilitates efficient looping and operations on
the array.


Example of ndarray in NumPy:

import numpy as np

# Create a 2D array (matrix)
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)         # Output: [[1 2 3] [4 5 6]]
print(arr.shape)   # Output: (2, 3) - 2 rows and 3 columns
print(arr.ndim)    # Output: 2 - 2 dimensions (rows and columns)
print(arr.size)    # Output: 6 - total number of elements
print(arr.dtype)   # Output: int64 (or another type depending on your system)

Performance: NumPy arrays are much faster for numerical computations due to their contiguous 
memory layout and optimized operations.

Memory Efficiency: Since NumPy arrays store elements of the same type, they use less memory 
compared to Python lists, which store references to objects.

Advanced Features: NumPy provides features like broadcasting, vectorized operations, and advanced 
indexing that make working with large datasets and performing complex calculations more 
efficient and concise.

Mathematical Operations: With NumPy, you can perform operations on entire arrays at once (e.g.,
matrix multiplication, element-wise addition), whereas with Python lists, you'd need to manually 
iterate over the elements.'''

#Q6.Analyze the performance benefits of NumPy arrays over Python lists for large-scale numerical
#   operations. 

'''Performance Benefits of NumPy Arrays Over Python Lists for Large-Scale Numerical Operations
NumPy arrays provide significant performance benefits over Python lists, especially when it comes
to large-scale numerical operations. These benefits stem from several key factors related to memory
layout, data type uniformity, vectorization, advanced functionality, and optimization for numerical tasks. Let's break down these performance advantages in detail.

1. Efficient Memory Layout

NumPy Arrays:

Contiguous memory storage: NumPy arrays are stored in a contiguous block of memory, meaning the 
elements are laid out sequentially without gaps between them. This enables the CPU to cache large 
chunks of data efficiently and access elements quickly.

Fixed-size data type: Each element in a NumPy array is of a fixed size (determined by the data
type), which allows NumPy to allocate a precise amount of memory for the entire array. This 
reduces memory overhead and fragmentation.

Python Lists:
Non-contiguous memory storage: Python lists are actually arrays of pointers (references) to
objects stored elsewhere in memory. This introduces extra overhead because the list itself is just
a collection of references, and each element is stored independently.
Variable-sized objects: Since Python lists can hold elements of different data types, each element
is a Python object, which has additional overhead (like reference counts, type information, etc.).
Performance Impact:
NumPy: Faster data access due to contiguous memory layout and lower memory overhead. It can access
large arrays more efficiently because of better cache locality.
Python Lists: Slower access and more memory usage, as each element is a separate Python object 
that requires additional storage and pointers.

2. Vectorized Operations (Broadcasting)

NumPy Arrays:

Vectorized operations: NumPy leverages vectorization, which means operations (like addition,
multiplication, or dot products) can be applied to entire arrays or subarrays at once without the
need for explicit Python loops.These operations are implemented in C and optimized for performance.

Broadcasting: NumPy also supports broadcasting, which allows arrays of different shapes to be 
operated on together, automatically expanding the smaller array to match the size of the larger 
one. This eliminates the need for looping or reshaping arrays manually.

Example: Vectorized addition of two arrays in NumPy


import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = arr1 + arr2  # Element-wise addition

Python Lists:

No native vectorized operations: Python lists do not support vectorized operations directly. To
perform operations like element-wise addition or multiplication, you would need to use explicit 
loops or list comprehensions, which are not optimized for performance and can be significantly 
slower for large arrays.

Example: Adding two lists with loops

arr1 = [1, 2, 3]
arr2 = [4, 5, 6]
result = [arr1[i] + arr2[i] for i in range(len(arr1))]  # Manual loop-based addition
Performance Impact:
NumPy: Operations on entire arrays (e.g., element-wise addition, matrix multiplication) are orders
of magnitude faster because they are implemented in low-level C, without the need for Python loops.

Python Lists: Using Python loops or list comprehensions to perform element-wise operations is much
slower due to Python’s dynamic nature and interpreter overhead.'''


#Q7.Compare vstack() and hstack() functions in NumPy.Provide example demonstrating their usage 
#   and output.


'''The functions vstack() and hstack() in NumPy are used to stack arrays along different axes, but 
they differ in the axis along which they combine the arrays.

vstack(): Stacks arrays vertically (row-wise). It stacks arrays along axis 0 (the first axis), 
which means it concatenates arrays by adding rows.

hstack(): Stacks arrays horizontally (column-wise). It stacks arrays along axis 1 (the second axis)
, which means it concatenates arrays by adding columns.

Key Differences:

vstack():

Stacks along axis 0 (vertically).
The number of columns (shape along axis 1) must be the same in all arrays.
Adds rows to the existing array.

hstack():

Stacks along axis 1 (horizontally).
The number of rows (shape along axis 0) must be the same in all arrays.
Adds columns to the existing array.


Example of vstack():

Let's look at how vstack() works with two arrays:


import numpy as np

# Create two 2D arrays with the same number of columns
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])

# Stack arrays vertically
result = np.vstack((arr1, arr2))

print("Result of vstack:")
print(result)

#output:

Result of vstack:

[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]


Example of hstack():
Now let's see how hstack() works with two arrays:


# Stack arrays horizontally
result = np.hstack((arr1, arr2))

print("Result of hstack:")
print(result)

#output:

Result of hstack:
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]'''


#Q8.Explain the difference between fliplr() and flipud() methods in NumPy, including their effects
#   on various array dimensions.

'''Difference Between fliplr() and flipud() in NumPy

The functions fliplr() and flipud() in NumPy are used to flip arrays along different axes. They
are specifically designed to flip arrays horizontally or vertically, depending on the function:

fliplr(): Flips an array left to right (horizontally).
flipud(): Flips an array up to down (vertically).

1. fliplr() (Flip Left to Right)

Effect: The fliplr() function flips an array along its left-right axis (axis 1 for 2D arrays),
meaning the order of columns is reversed.

Applicable to: 1D and 2D arrays. In the case of a 1D array, it simply reverses the order of 
elements.

Example 1: 2D Array (Matrix)


import numpy as np

arr = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])

flipped_arr = np.fliplr(arr)
print(flipped_arr)

#output:

[[3 2 1]
 [6 5 4]
 [9 8 7]]


Example 2: 1D Array


arr = np.array([1, 2, 3, 4, 5])

flipped_arr = np.fliplr(arr)
print(flipped_arr)


#output:

[5 4 3 2 1]


2. flipud() (Flip Up to Down)

Effect: The flipud() function flips an array along its up-down axis (axis 0 for 2D arrays), 
meaning the order of rows is reversed.

Applicable to: 1D and 2D arrays. For a 1D array, flipud() behaves the same as fliplr() because it 
reverses the order of the elements, essentially flipping the array upside down.

Example 1: 2D Array (Matrix)

arr = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])

flipped_arr = np.flipud(arr)
print(flipped_arr)


#output:

[[7 8 9]
 [4 5 6]
 [1 2 3]]


Example 2: 1D Array

arr = np.array([1, 2, 3, 4, 5])

flipped_arr = np.flipud(arr)
print(flipped_arr)


#output:

[5 4 3 2 1]'''

#Q9.Discuss the functionality of the array_split() method in NumPy.How does it handle uneven 
#   splits?


'''array_split() in NumPy

The array_split() function in NumPy is used to split an array into multiple sub-arrays along a
specified axis. Unlike split(), which requires that the array be evenly divisible, array_split() 
allows splitting the array even when the array size isn't perfectly divisible by the number of 
splits. It handles uneven splits by distributing the remaining elements across the sub-arrays.

Syntax:
numpy.array_split(ary, indices_or_sections, axis=0)


Key Features of array_split():

Flexible Number of Splits: array_split() can split the array into any number of sub-arrays, even
if the array size is not perfectly divisible by the number of splits.

Handling Uneven Splits: If the number of elements in the array is not evenly divisible by the 
number of desired splits, array_split() will distribute the remaining elements among the sub-
arrays. Some sub-arrays may have one more element than others.

Returns: It returns a list of sub-arrays, not a single concatenated result.


When the array size is not perfectly divisible by the number of splits, array_split() handles this
by distributing the "extra" elements across the sub-arrays. The general approach is:

Extra elements are given to the earlier sub-arrays. If you want to split an array of size n into k
parts, each part will contain at least n // k elements.
Any remaining elements (n % k elements) will be evenly distributed among the first few sub-arrays.
For example, splitting an array of 9 elements into 4 parts:

Base size for each split: 9 // 4 = 2
Remainder: 9 % 4 = 1
Thus, the first sub-array will get 3 elements (2 + 1 extra), and the remaining sub-arrays will get 
2 elements each.

Edge Cases:

Empty arrays: If you pass an empty array, array_split() will return an empty list of sub-arrays.

Splitting into larger parts than array size: If you try to split into more parts than the number 
of elements in the array, NumPy will create empty sub-arrays for the remaining splits.

Example:

arr = np.array([1, 2, 3])
sub_arrays = np.array_split(arr, 5)
for sub in sub_arrays:
    print(sub)


#output:

[1]
[2]
[3]
[]
[]'''


#Q10.Explain the concept of vectorization and broadcasting in NumPy. How do they contribute to
#    efficient array operation. 


'''1. Vectorization

Vectorization refers to the ability to apply operations to entire arrays (or vectors) at once, 
instead of performing operations element by element. In essence, vectorized operations are high-
level, optimized, and often written in low-level languages (like C or Fortran), making them signi
ficantly faster than equivalent Python loops.

How Vectorization Works:

Instead of looping through each element of an array and performing the operation manually, vectori
zed operations allow you to apply the operation directly on the entire array.
NumPy operations such as addition, subtraction, multiplication, and other mathematical functions
are vectorized, meaning they are designed to work on entire arrays without explicit Python loops.

Example of Vectorization:


import numpy as np

arr = np.array([1, 2, 3, 4, 5])
result = arr * 2  # This is a vectorized operation
print(result)

#output:

[2 4 6 8 10]


2. Broadcasting

Broadcasting is a technique in NumPy that allows operations between arrays of different shapes. 
When performing an operation between arrays of different dimensions or shapes, NumPy "broadcasts" 
the smaller array to match the shape of the larger array. This enables operations to be applied 
without explicitly reshaping the arrays, making code more concise and efficient.

Broadcasting Rules:

Dimensions: If two arrays have different numbers of dimensions, the smaller array is padded with 
ones on the left side of its shape to match the number of dimensions of the larger array.

Shape Compatibility: The arrays are compatible if, for each dimension, the size of the dimension 
is either the same or one of the arrays has size 1 along that dimension.

Expansion: Once the arrays are compatible, NumPy implicitly "expands" the smaller array along the
dimensions where it has size 1, effectively matching the shape of the larger array.

Example of Broadcasting:

import numpy as np

arr1 = np.array([[1], [2], [3]])  # Shape (3, 1)
arr2 = np.array([[10, 20, 30, 40], [10, 20, 30, 40], [10, 20, 30, 40]])  # Shape (3, 4)

result = arr1 + arr2  # Broadcasting happens here
print(result)


#output:

[[11 21 31 41]
 [12 22 32 42]
 [13 23 33 43]]'''




#Practical Question

#Q1.Create a 3*3 NumPy array with random integers between 1 and 100. Then interchange the rows 
#   and columns.

'''import numpy as np

# Create a 3x3 array with random integers between 1 and 100
arr = np.random.randint(1, 101, size=(3, 3))

# Print the original array
print("Original Array:")
print(arr)

# Interchange the rows and columns (Transpose the array)
transposed_arr = arr.T

# Print the transposed array
print("\nTransposed Array (Rows and Columns Interchanged):")
print(transposed_arr)


#output:
Original Array:
[[23 56 78]
 [12 67 89]
 [34 45 67]]

Transposed Array (Rows and Columns Interchanged):
[[23 12 34]
 [56 67 45]
 [78 89 67]]'''


 #Q2.Generate a 1D NumPy array with 10 elements .Reshape it into 2*5 array,then into 5*2 array. 


'''import numpy as np

# Create a 1D array with 10 elements (values from 0 to 9)
arr = np.arange(10)

# Reshape the 1D array into a 2x5 array
reshaped_2x5 = arr.reshape(2, 5)
print("Reshaped to 2x5 array:")
print(reshaped_2x5)

# Reshape the 1D array into a 5x2 array
reshaped_5x2 = arr.reshape(5, 2)
print("\nReshaped to 5x2 array:")
print(reshaped_5x2)

#output:
Reshaped to 2x5 array:
[[0 1 2 3 4]
 [5 6 7 8 9]]

Reshaped to 5x2 array:
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]'''


#Q3.Create a 4*4 NumPy array with random float values .Add a border of Zeros around 
#   it,resulting in a 6*6.

'''import numpy as np

# Step 1: Create a 4x4 array with random float values between 0 and 1
arr = np.random.rand(4, 4)
print("Original 4x4 Array:")
print(arr)

# Step 2: Add a border of zeros around the array, resulting in a 6x6 array
arr_with_border = np.pad(arr, pad_width=1, mode='constant', constant_values=0)

print("\nArray with Border (6x6):")
print(arr_with_border)

#output:
Original 4x4 Array:
[[0.57469078 0.51584802 0.70554046 0.66430964]
 [0.71294626 0.46949693 0.2923382  0.3874257 ]
 [0.98293307 0.64733769 0.77141434 0.15926675]
 [0.39329673 0.69288503 0.04219552 0.68524662]]

Array with Border (6x6):
[[0.         0.         0.         0.         0.         0.        ]
 [0.         0.57469078 0.51584802 0.70554046 0.66430964 0.        ]
 [0.         0.71294626 0.46949693 0.2923382  0.3874257  0.        ]
 [0.         0.98293307 0.64733769 0.77141434 0.15926675 0.        ]
 [0.         0.39329673 0.69288503 0.04219552 0.68524662 0.        ]
 [0.         0.         0.         0.         0.         0.        ]]'''


 #Q4. Using NumPy, create an array of integers from 10 to 60 with a step of 5.

'''import numpy as np

# Create an array of integers from 10 to 60 with a step of 5
arr = np.arange(10, 61, 5)

# Print the array
print(arr)

#output:
[10 15 20 25 30 35 40 45 50 55 60]'''

#Q5. Create a NumPy array of strings['python','numpy','pandas'].Apply different cases 
#    transformations (uppercase,lowercase,titlecase,etc) to each element.

'''import numpy as np

# Create a NumPy array of strings
arr = np.array(['python', 'numpy', 'pandas'])

# Apply different case transformations
uppercase_arr = np.char.upper(arr)    # Convert all to uppercase
lowercase_arr = np.char.lower(arr)    # Convert all to lowercase
titlecase_arr = np.char.title(arr)    # Convert each string to title case
capitalize_arr = np.char.capitalize(arr)  # Capitalize the first letter of each word

# Print the original and transformed arrays
print("Original Array:", arr)
print("Uppercase:", uppercase_arr)
print("Lowercase:", lowercase_arr)
print("Titlecase:", titlecase_arr)
print("Capitalize:", capitalize_arr)

#output:

Original Array: ['python' 'numpy' 'pandas']
Uppercase: ['PYTHON' 'NUMPY' 'PANDAS']
Lowercase: ['python' 'numpy' 'pandas']
Titlecase: ['Python' 'Numpy' 'Pandas']
Capitalize: ['Python' 'Numpy' 'Pandas']'''

#Q6.Generate a NumPy array of words. Insert a space between each character of every word 
#   in an array.

import numpy as np

# Create a NumPy array of words
words = np.array(['python', 'numpy', 'pandas'])

# Insert a space between each character of every word
words_with_spaces = np.char.join(' ', words)

# Print the resulting array
print(words_with_spaces)


#output:

['p y t h o n' 'n u m p y' 'p a n d a s']


#Q7.Create two 2D NumPy arrays and perform element-wise addition, substraction, multiplication,
#   and division.

'''import numpy as np

# Create two 2D NumPy arrays
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[6, 5, 4], [3, 2, 1]])

# Perform element-wise addition
addition_result = arr1 + arr2

# Perform element-wise subtraction
subtraction_result = arr1 - arr2

# Perform element-wise multiplication
multiplication_result = arr1 * arr2

# Perform element-wise division
division_result = arr1 / arr2

# Print the results
print("Array 1:")
print(arr1)

print("\nArray 2:")
print(arr2)

print("\nElement-wise Addition:")
print(addition_result)

print("\nElement-wise Subtraction:")
print(subtraction_result)

print("\nElement-wise Multiplication:")
print(multiplication_result)

print("\nElement-wise Division:")
print(division_result)

#output:
Array 1:
[[1 2 3]
 [4 5 6]]

Array 2:
[[6 5 4]
 [3 2 1]]

Element-wise Addition:
[[7 7 7]
 [7 7 7]]

Element-wise Subtraction:
[[-5 -3 -1]
 [ 1  3  5]]

Element-wise Multiplication:
[[ 6 10 12]
 [12 10  6]]

Element-wise Division:
[[0.16666667 0.4        0.75      ]
 [1.33333333 2.5        6.        ]]'''

 #Q8.Use NumPy to create 5*5 identity matrix,then extract its diagonal elements.

'''import numpy as np

# Create a 5x5 identity matrix
identity_matrix = np.eye(5)

# Extract the diagonal elements
diagonal_elements = np.diagonal(identity_matrix)

# Print the results
print("5x5 Identity Matrix:")
print(identity_matrix)

print("\nDiagonal Elements:")
print(diagonal_elements)


#output:

5x5 Identity Matrix:
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

Diagonal Elements:
[1. 1. 1. 1. 1.]'''

#Q9. Generate a NumPy array of 100 random integers between 0 and 1000.Find and display all
#    prime numbers in this array. 

'''import numpy as np

# Step 1: Generate a NumPy array of 100 random integers between 0 and 1000
arr = np.random.randint(0, 1001, size=100)

# Step 2: Define a function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Step 3: Find and display all prime numbers in the array
prime_numbers = arr[arr[0:] & [is_prime(x) for x in arr]]

# Print the original array and prime numbers
print("Original Array of 100 Random Integers:")
print(arr)

print("\nPrime Numbers in the Array:")
print(prime_numbers)


#output:

Original Array of 100 Random Integers:
[983 464 251 302 203 468 195 417 320 618 452 753 259 672 407 877 870 546 175 778 615 520 139 899
563 862 759 425 870 767 118 372 203 425 120 345 595 506  58 334 190 139 131 334 115 429 241  0 
994 376 604 753 278 544 635  97 249 111 168 482 276 426 321 784 767 418 125 480 848 102  78 121
251 824  94 872 809 539 944 274 270  16 144 118 591 315 102 196 352 428 133 283 598 473  27 209 
280 234 477 824 312 223 370 963 396 274 523 494 100 225 128 909]

Prime Numbers in the Array:
[983 251 139 563 131 139 139 137 167]'''

#Q10. Create a NumPy array representing daily temperature for a month.Calculate and Display 
#     the weekly average.


'''import numpy as np

#  Generate a NumPy array of daily temperatures for a month (30 days)
# For demonstration, let's assume the temperatures are random integers between 10 and 35 (in Celsius).
daily_temperatures = np.random.randint(10, 36, size=30)

#  Reshape the array into 4 weeks (5 days per week, as a 30-day month can be divided into 4 weeks)
weekly_temperatures = daily_temperatures.reshape(4, 7)

#  Calculate the weekly average temperature
weekly_average = weekly_temperatures.mean(axis=1)

#  Display the results
print("Daily Temperatures for the Month:")
print(daily_temperatures)

print("\nTemperatures Reshaped into Weeks (4 Weeks, 7 Days per Week):")
print(weekly_temperatures)

print("\nWeekly Average Temperatures:")
print(weekly_average)


#output:

Daily Temperatures for the Month:
[33 20 13 29 12 21 29 34 27 27 16 28 21 17 12 24 25 15 23 32 17 32 18 12 34 29 30 34 24 13]

Temperatures Reshaped into Weeks (4 Weeks, 7 Days per Week):
[[33 20 13 29 12 21 29]
 [34 27 27 16 28 21 17]
 [12 24 25 15 23 32 17]
 [32 18 12 34 29 30 34]]

Weekly Average Temperatures:
[22.42857143 22.14285714 22.42857143 26.57142857]'''

# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 
import numpy as np

'''
Intitial Total time: 1.43673 s
Improved Total time: 0.003961 s
Speedup = 362.7
'''
def add(x,y):
    """
    Add two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    if x.shape == y.shape:
        return x+y
    else:
        raise Exception('Incompatible arrays')


'''
Intitial Total time: 1.45457 s
Improved Total time: 0.004516 s
Speedup = 684.8
'''
def multiply(x,y):
    """
    Multiply two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    if x.shape == y.shape:
        return x*y
    else:
        raise Exception('Incompatible arrays')


'''
Intitial Total time: 1.39358 s
Improved Total time: 0.004094 s
Speedup = 340.4
'''
def sqrt(x):
    """
    Take the square root of the elements of an arrays using a Python loop.
    """
    return np.sqrt(x)


'''
Intitial Total time: 3.92057 s
Improved Total time: 0.013878 s
Speedup = 282.5
'''
def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = multiply(x,x)
    yy = multiply(y,y)
    zz = add(xx, yy)
    return sqrt(zz)
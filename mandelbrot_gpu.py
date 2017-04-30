# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
    Got the blockdim and griddim values, calucate the start and end positions of the block.
    Since not using memory, whole image would be available.
    '''
    ### YOUR CODE HERE
    dxB, dyB = cuda.blockDim.x, cuda.blockDim.y
    dxG, dyG = cuda.gridDim.x, cuda.gridDim.y
    (x_index, y_index) = cuda.grid(2)
    
    (x_start, y_start) = (x_index * int(image.shape[0]/(dxG*dxB)), y_index *int(image.shape[1]/(dyG*dyB)))
    (x_end, y_end) = (x_start + int(image.shape[0]/(dxG*dxB)), y_start + int(image.shape[1]/(dyG*dyB))) 
    
    
    pixel_size_x = (max_x - min_x) / image.shape[1]
    pixel_size_y = (max_y - min_y) / image.shape[0]
    

    for x in range(y_start, y_end):
        real = min_x + x * pixel_size_x
        for y in range(x_start, x_end):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)
            
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)
    
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()
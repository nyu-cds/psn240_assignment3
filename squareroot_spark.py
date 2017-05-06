from pyspark import SparkContext
from operator import add

if __name__ == '__main__':
    sc = SparkContext("local", "squareroot")
    # Create an RDD of numbers from 0 to 1,000
    nums = sc.parallelize(range(1,1001))
    #Finds the square root of all the values
    squroot_nums = nums.map(lambda x: (x**0.5))
    #Finds the number of elements
    n = nums.count()
    #Prints the average of the square root of numbers
    print(squroot_nums.fold(0, add)/n)
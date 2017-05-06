from pyspark import SparkContext
from operator import mul

if __name__ == '__main__':
    sc = SparkContext("local", "products")
    # Create an RDD of numbers from 0 to 1,000
    nums = sc.parallelize(range(1,1001))
    #Prints the product of all the values in RDD
    print(nums.fold(1, mul))
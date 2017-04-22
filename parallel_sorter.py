'''
    Creates a unsorted array(size is entered by user) and sorts it
'''    

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

'''
    Merge two sorted array, maintaining the sort order
'''    
def Merge(A,B):
    result = []
    len_A = len(A)
    len_B = len(B)
    ref = 0
    while len(A)>0 or len(B)>0:
        if len(A) == 0:
            result.extend(B)
            return result
        if len(B) == 0:
            result.extend(A)
            return result
        if A[ref] > B[ref]:
            result.append(B[ref])
            B = B[1:]
        else:
            result.append(A[ref])
            A = A[1:]
    return result
            
#Rank zero would create n sized array and split it to number of process
if rank == 0:
    #Enter the array size to be sorted
    while True:
        n = input("Enter array size to be sorted(integer value only): ")
        try:
            n = int(n)
            break
        except ValueError:
            print('Enter only integer values')
    unsorted = np.random.randint(1,10000,n)
    print('Unsorted array:', unsorted)
    #Partition the array
    data = np.array_split(unsorted, size)
    partition_data = None
else:
    data = None
    partition_data = None
# scatter the partition to different process 
partition_data = comm.scatter(data, root=0)
# Sort the array
partition_sorted_data = np.sort(partition_data)
sorted_data = np.array([])
#Gather the sorted partitions, this would have value for only for root or zero rank
temp = comm.gather(partition_sorted_data, root=0)
if temp is not None:
    #Merge the sorted partitions
    for s in temp:
        sorted_data = Merge(sorted_data, s)
    print('Sorted array:',sorted_data)        
#MPI program for printing greetings based on rank is even or odd
import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
# Get the rank of the process
rank = comm.Get_rank()
#Print message based on the rank number
if rank % 2 == 0:
    print("Hello from process ", rank)
else :
    print("Goodbye from process ", rank)
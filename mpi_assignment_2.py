#MPI program for sending incremental values to orderely process
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
input_var = np.array([0])

# Handle all ranks except rank zero
if rank != 0:
        prev = rank -1
        req = comm.Irecv(input_var, source=rank-1)
        req.Wait()
        input_var[0] = input_var[0] * rank
        if rank+1 < size:
            print("Process", rank, "sends the value", input_var[0], "to process ", rank+1)
            comm.Isend(input_var, dest=rank+1)
        else:
            #Sends the final value to rank zero to print
            comm.Isend(input_var, dest=0)
            
if rank == 0:    
    # Checking for integer values(Ref: http://stackoverflow.com/questions/25794490/how-do-i-check-a-value-entered-python-3-4)
    while True:
        inputValue = input("Enter value(<100): ")
        try:
            input_var[0] = int(inputValue)
            # Checking value is less than 100
            if(input_var[0] <= 100):
                if rank+1 < size:
                    print("Process", rank, "sends the value", input_var[0], "to process 1")
                    comm.Isend(input_var, dest=1)
                else:
                    #Sends the final value to rank zero to print
                    comm.Isend(input_var, dest=0)
                break
            else:
                raise ValueError
        except ValueError:
            print('Enter only integer values less than 100')
    # Receive values from any source to print
    req = comm.Irecv(input_var, source=MPI.ANY_SOURCE)
    req.Wait()
    print("Final value", input_var[0])
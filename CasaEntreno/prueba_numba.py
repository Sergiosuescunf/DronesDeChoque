# SuperFastPython.com
# example of a concurrent for loop
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor
 
# task to execute in another thread
def task(arg1,arg2):
    print(f"Arg1 is {arg1}")
    print(f"Arg2 is {arg2}")
    # generate a value between 0 and 1
    value = random()
    # block for a fraction of a second to simulate work
    sleep(value)
    # return the generated value
    return value
 
# entry point for the program
if __name__ == '__main__':
    # create the thread pool
    with ThreadPoolExecutor() as tpe:
        # call the same function with different data concurrently
        for result in tpe.map(task, range(10), range(6)):
            # report the value to show progress
            print(result)
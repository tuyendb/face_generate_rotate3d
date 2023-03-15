
from memory_profiler import profile
class A:
    def __init__(self, x):
        self.x = x

@profile
def my_func():
    for i in range(10000):
        a = A(x=i)
    return a

if __name__ == '__main__':
    my_func()
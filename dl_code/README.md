## Deep Q Learning


# Structure
- pdf : An introduction to RL learning
- test.py : an easy python script
- embedded_python_in_cpp.cpp : a cpp file that invoking python code

# Embedded a python code in C++
There are three ways to invoke python codes. In `embedded_python_in_cpp.cpp`, the three ways
are demonstrated as case 1, 2, and 3.
Case 1 simply executes a python line.
Case 2 executes test.py.
Case 3 is a generalized way.
To build `embedded_python_in_cpp.cpp`, simply run the instruction where python3.x is the python binary in your system:
```
g++ -std=c++17 embedded_python_in_cpp.cpp -I/path/to/python/headers -lpython3.x
```

To run the executable, simply run the instruction:
```
./a.out
```

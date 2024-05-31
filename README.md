This is a simple framework for automatic differentiation in C++ with CUDA. There are several issues currently present:

1. Possible memory leaks.
2. Memory being deallocated multiple times.
3. Incomplete functions and features.
4. Poor optimization and code style.

**Compilation methods:**

*Compiler: nvcc or g++*
```
cd auto_grad_cuda
nvcc ./src/main.cu -o ./build/main_cuda -lm -lstdc++ -I include/
# or
g++ ./src/main.cpp -o ./build/main_g++ -lm -lstdc++ -I include/
```

**Current performance comparison:**

The program accelerated with CUDA runs approximately 0.5 times faster than the program without CUDA.

Without rigorous testing, the code compiled with nvcc is `main.cu`, and the code compiled with g++ as `main.cpp` is a symbolic link of `main.cu`.


**NOTE:** 

This is just a "toy code" for practicing CUDA. I do not guarantee the accuracy of the results due to its experimental nature.

#pragma once

#include <cmath>
#include <map>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstring>

//#define DEVICE_GPU
#define ERROR_HANDLE

#ifdef DEVICE_GPU
    #ifdef ERROR_HANDLE
        cudaError_t inner_err;
    #endif
#else
    #undef ERROR_HANDLE
#endif

#ifdef DEVICE_GPU
    #define CALL_FUNC(func_name, ...) func_name<<<BLOCK_SIZE, MAX_THREAD>>>(__VA_ARGS__);
#else
    #define CALL_FUNC(func_name, ...) func_name(__VA_ARGS__);
#endif


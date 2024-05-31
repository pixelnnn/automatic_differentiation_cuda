#ifdef MACRODEF_H

#else
#define MACRODEF_H



#include "config.h"



#ifdef NULL
#undef NULL
#endif
#define NULL 0
#define BLOCK_SIZE 30*64
#define MAX_THREAD 1024


#ifdef DEVICE_GPU
#define MEM_MALLOC(ptr, size)      \
    cudaMallocManaged(&ptr, size); \
    ERR_CUDA(inner_err);
#define MEM_FREE(ptr)  \
    if (ptr != NULL)   \
    {                  \
        cudaFree(ptr); \
        ptr = NULL;    \
    };                 \
    ERR_CUDA(inner_err);
#define SYNC                 \
    cudaDeviceSynchronize(); \
    ERR_CUDA(inner_err);
#define INDEX (blockDim.x * blockIdx.x + threadIdx.x)
#define STRIKE (blockDim.x * gridDim.x)
#define MEM_SET(ptr, value, size)                           \
    valueset<<<BLOCK_SIZE, MAX_THREAD>>>(ptr, value, size); \
    SYNC;
#define MEM_CPY(dst, src, size)                           \
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); \
    SYNC;
#else
#define MEM_MALLOC(ptr, size) ptr = (decltype(ptr))malloc(size)
#define MEM_FREE(ptr) \
    if (ptr != NULL)  \
    {                 \
        free(ptr);    \
        ptr = NULL;   \
    }
#define MEM_SET(ptr, value, size)     \
    for (size_t i = 0; i < size; ++i) \
    {                                 \
        (ptr)[i] = value;             \
    }
#define SYNC ;
#define MEM_CPY(dst, src, size) memcpy(dst, src, size);
#endif

// 判断ptr是不是NULL，如果是NULL，就分配内存，不然先释放内存再分配内存
#define MEM_AUTO_MALLOC(ptr, size) \
    if (ptr == NULL)               \
    {                              \
        MEM_MALLOC(ptr, size);     \
    }                              \
    else                           \
    {                              \
        MEM_FREE(ptr);             \
        MEM_MALLOC(ptr, size);     \
    }
#define PTR_TYPE(type, ptr) type *ptr = NULL

#define GEN_VAR_WITH_VALUE(type, ptr, value, size) \
    PTR_TYPE(type, ptr);                           \
    MEM_MALLOC(ptr, sizeof(type) * size);          \
    MEM_SET((type *)ptr, value, size);

#define GEN_VAR(type, ptr, size)                   \
    PTR_TYPE(type, ptr);                           \
    MEM_MALLOC(ptr, sizeof(type) * size);          


#ifdef ERROR_HANDLE
#define ERR_CUDA(err)                                                             \
    err = cudaGetLastError();                                                     \
    if (err != cudaSuccess)                                                       \
    {                                                                             \
        printf("ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    }
#else
    #define ERR_CUDA(err) ;
#endif

#define MAKE_1OP(func_name) Ntensor func_name(Ntensor& a)


#define SIMFUNC_FOR_CUDA_1OP(function_name, apply_func_for_each_value) \
    __global__ void function_name(double *a, size_t N)              \
    {                                                               \
        int strike = STRIKE;                                        \
        for (int i = INDEX; i < N; i += strike)                     \
        {                                                           \
            a[i] = apply_func_for_each_value;                       \
        }                                                           \
    }


#define SIMFUNC_FOR_CUDA_2OP(function_name, apply_func_for_each_value)          \
    __global__ void function_name(double *a, double *b, double *c, size_t N) \
    {                                                                        \
        int strike = STRIKE;                                                 \
        for (int i = INDEX; i < N; i += strike)                              \
        {                                                                    \
            c[i] = apply_func_for_each_value;                                \
        }                                                                    \
    }
#define SIMFUNC_FOR_CPU_1OP(function_name, apply_func_for_each_value) \
    void function_name(double *a, size_t N)                        \
    {                                                              \
        for (int i = 0; i < N; ++i)                                \
        {                                                          \
            a[i] = apply_func_for_each_value;                      \
        }                                                          \
    }

#define SIMFUNC_FOR_CPU_2OP(function_name, apply_func_for_each_value) \
    void function_name(double *a, double *b, double *c, size_t N)  \
    {                                                              \
        for (int i = 0; i < N; ++i)                                \
        {                                                          \
            c[i] = apply_func_for_each_value;                      \
        }                                                          \
    }

#ifdef DEVICE_GPU
    #define CONSTRUCT_SIMFUNC_1OP(func_name, apply_func_for_each_value) SIMFUNC_FOR_CUDA_1OP(func_name, apply_func_for_each_value)
    #define CONSTRUCT_SIMFUNC_2OP(func_name, apply_func_for_each_value) SIMFUNC_FOR_CUDA_2OP(func_name, apply_func_for_each_value)
#else
    #define CONSTRUCT_SIMFUNC_1OP(func_name, apply_func_for_each_value) SIMFUNC_FOR_CPU_1OP(func_name, apply_func_for_each_value)
    #define CONSTRUCT_SIMFUNC_2OP(func_name, apply_func_for_each_value) SIMFUNC_FOR_CPU_2OP(func_name, apply_func_for_each_value)
#endif


#ifdef DEVICE_GPU
__global__ void valueset(double *ptr, double value, size_t size)
{

    int strike = STRIKE;
    for (int i = INDEX; i < size; i += strike)
    {
        ptr[i] = value;
    }
}

// + - * /

__global__ void addNtensor(double *a, double *b, double *c, size_t N)
{ // c=a+b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] + b[i];
    }
}
__global__ void addNtensor(double a, double *b, double *c, size_t N)
{ // c=a+b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a + b[i];
    }
}
__global__ void addNtensor(double *a, double b, double *c, size_t N)
{ // c=a+b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] + b;
    }
}
__global__ void subNtensor(double *a, double *b, double *c, size_t N)
{ // c=a-b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] - b[i];
    }
}
__global__ void subNtensor(double a, double *b, double *c, size_t N)
{ // c=a-b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a - b[i];
    }
}
__global__ void subNtensor(double *a, double b, double *c, size_t N)
{ // c=a-b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] - b;
    }
}
__global__ void mulNtensor(double *a, double *b, double *c, size_t N)
{ // c=a*b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] * b[i];
    }
}
__global__ void mulNtensor(double a, double *b, double *c, size_t N)
{ // c=a*b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a * b[i];
    }
}
__global__ void mulNtensor(double *a, double b, double *c, size_t N)
{ // c=a*b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] * b;
    }
}
__global__ void divNtensor(double *a, double *b, double *c, size_t N)
{ // c=a/b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] / b[i];
    }
}
__global__ void divNtensor(double a, double *b, double *c, size_t N)
{ // c=a/b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a / b[i];
    }
}
__global__ void divNtensor(double *a, double b, double *c, size_t N)
{ // c=a/b
    int strike = STRIKE;
    for (int i = INDEX; i < N; i += strike)
    {
        c[i] = a[i] / b;
    }
}
#else
// + - * /

void addNtensor(double *a, double *b, double *c, size_t N)
{ // c=a+b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] + b[i];
    }
}
void addNtensor(double a, double *b, double *c, size_t N)
{ // c=a+b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a + b[i];
    }
}
void addNtensor(double *a, double b, double *c, size_t N)
{ // c=a+b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] + b;
    }
}
void subNtensor(double *a, double *b, double *c, size_t N)
{ // c=a-b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] - b[i];
    }
}
void subNtensor(double a, double *b, double *c, size_t N)
{ // c=a-b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a - b[i];
    }
}
void subNtensor(double *a, double b, double *c, size_t N)
{ // c=a-b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] - b;
    }
}
void mulNtensor(double *a, double *b, double *c, size_t N)
{ // c=a*b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] * b[i];
    }
}
void mulNtensor(double a, double *b, double *c, size_t N)
{ // c=a*b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a * b[i];
    }
}
void mulNtensor(double *a, double b, double *c, size_t N)
{ // c=a*b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] * b;
    }
}
void divNtensor(double *a, double *b, double *c, size_t N)
{ // c=a/b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] / b[i];
    }
}
void divNtensor(double a, double *b, double *c, size_t N)
{ // c=a/b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a / b[i];
    }
}
void divNtensor(double *a, double b, double *c, size_t N)
{ // c=a/b
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] / b;
    }
}
#endif
#define MID_CONSTRUCTION(op, func_name)                       \
    inline void op(double *a, double *b, double *c, size_t N) \
    {                                                         \
        CALL_FUNC(func_name, a, b, c, N);                     \
    }                                                         \
    inline void op(double a, double *b, double *c, size_t N)  \
    {                                                         \
        CALL_FUNC(func_name, a, b, c, N);                     \
    }                                                         \
    inline void op(double *a, double b, double *c, size_t N)  \
    {                                                         \
        CALL_FUNC(func_name, a, b, c, N);                     \
    }

#define MAKE_SIMPLE_OPR_INLINE(op) MID_CONSTRUCTION(op, op##Ntensor)


#endif
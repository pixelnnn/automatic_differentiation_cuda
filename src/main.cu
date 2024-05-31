
#include "macrodef.h"
#include "functional.h"
#include "ntensor.h"


int main(){
    GEN_VAR_WITH_VALUE(double, a_value, 1.0, (unsigned)2<<24);
    GEN_VAR_WITH_VALUE(double, b_value, 2.0, (unsigned)2<<24);
    GEN_VAR_WITH_VALUE(double, c_value, 1.5, (unsigned)2<<24);
    GEN_VAR_WITH_VALUE(double, root_grad, 1.0, (unsigned)2<<24);
    int id=0;
    #ifdef DEVICE_GPU
        cudaGetDevice(&id);
    #endif
    std::cout<<"CUDA ID: "<<id<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    Ntensor X(a_value,{1,(unsigned)2<<24},NULL,NULL,std::string(), NULL), 
        W(b_value,{1,(unsigned)2<<24},NULL,NULL,std::string(),NULL), 
        B(c_value,{1,(unsigned)2<<24},NULL,NULL,std::string(),NULL),
        C(c_value,{1,(unsigned)2<<24},NULL,NULL,std::string(),NULL),
        Y(c_value,{1,(unsigned)2<<24},NULL,NULL,std::string(),NULL),
        P_Y(c_value,{1,(unsigned)2<<24},NULL,NULL,std::string(),NULL),
        L(c_value,{1,(unsigned)2<<24},NULL,NULL,std::string(),NULL)
        ;
    C=W*X;
    Y=sigmoid(C)+B;
    L=P_Y-Y;
    L.backward(root_grad);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " ms" << std::endl;
    return 0;
}

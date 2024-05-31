#ifdef FUNCTIONAL_H

#else
#define FUNCTIONAL_H
#include "macrodef.h"





CONSTRUCT_SIMFUNC_1OP(n_exp, exp(a[i]));
CONSTRUCT_SIMFUNC_1OP(n_tan, tan(a[i]));
CONSTRUCT_SIMFUNC_1OP(n_tanh, tanh(a[i]));
CONSTRUCT_SIMFUNC_1OP(n_sin, sin(a[i]));
CONSTRUCT_SIMFUNC_1OP(n_cos, cos(a[i]));
CONSTRUCT_SIMFUNC_1OP(n_relu, ((a[i]>0.0)?(a[i]):(0.0)));
CONSTRUCT_SIMFUNC_2OP(n_max, ((a[i]>b[i])?(a[i]):(b[i])));
CONSTRUCT_SIMFUNC_2OP(n_min, ((a[i]>b[i])?(b[i]):(a[i])));



#endif

#ifdef NTENSOT_H

#else
#define NTENSOT_H

#pragma once
#include "macrodef.h"
#include "functional.h"


MAKE_SIMPLE_OPR_INLINE(add);
MAKE_SIMPLE_OPR_INLINE(sub);
MAKE_SIMPLE_OPR_INLINE(mul);
MAKE_SIMPLE_OPR_INLINE(div);

class Ntensor
{
public:
    PTR_TYPE(double, data);
    PTR_TYPE(double, grad);
    PTR_TYPE(Ntensor, left);
    PTR_TYPE(Ntensor, right);
    std::vector<size_t> shape;
    std::string op;
    size_t size() const
    {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    }
    size_t bytesize() const
    {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * sizeof(double);
    }

public:
    Ntensor()
    {
        data = NULL;
        grad = NULL;
        left = NULL;
        right = NULL;
    }

    Ntensor(double *_data, std::vector<size_t> _shape, Ntensor *_left, Ntensor *_right, std::string _op, double *_grad)
    {

        MEM_AUTO_MALLOC(data, std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        MEM_CPY(data, _data, std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        if (_grad == NULL)
        {
            MEM_AUTO_MALLOC(grad, std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
            MEM_SET(grad, 0, std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()));
        }
        else
        {
            MEM_AUTO_MALLOC(grad, std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
            MEM_CPY(grad, _grad, std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        }

        shape = _shape;
        left = _left;
        right = _right;
        op = _op;
    }
    // 重载=
    Ntensor &operator=(const Ntensor &other)
    {
        MEM_AUTO_MALLOC(data, std::accumulate(other.shape.begin(), other.shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        MEM_CPY(data, other.data, std::accumulate(other.shape.begin(), other.shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        MEM_AUTO_MALLOC(grad, std::accumulate(other.shape.begin(), other.shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        MEM_CPY(grad, other.grad, std::accumulate(other.shape.begin(), other.shape.end(), 1, std::multiplies<size_t>()) * sizeof(double));
        shape = other.shape;
        left = other.left;
        right = other.right;
        op = other.op;
        return *this;
    }

    ~Ntensor()
    {
        MEM_FREE(this->data);
        MEM_FREE(this->grad);
    }

    void print()
    {
        // 使用printf而非cout
        printf("shape: ");
        for (int i = 0; i < shape.size(); ++i)
        {
            printf("%lu ", shape[i]);
        }
        printf("\n");
        printf("data: ");
        for (int i = 0; i < this->size(); ++i)
        {
            printf("%.3lf ", data[i]);
        }
        printf("\n");
    }

    Ntensor operator+(const Ntensor &other) const
    {
        PTR_TYPE(double, newData);
        MEM_AUTO_MALLOC(newData, this->bytesize());
        add(data, other.data, newData, this->size());
        return Ntensor(newData, shape, const_cast<Ntensor *>(this), const_cast<Ntensor *>(&other), "+", grad);
    }
    Ntensor operator-(const Ntensor &other) const
    {
        PTR_TYPE(double, newData);
        MEM_AUTO_MALLOC(newData, this->bytesize());
        sub(data, other.data, newData, this->size());
        return Ntensor(newData, shape, const_cast<Ntensor *>(this), const_cast<Ntensor *>(&other), "-", grad);
    }
    Ntensor operator*(const Ntensor &other) const
    {
        PTR_TYPE(double, newData);
        MEM_AUTO_MALLOC(newData, this->bytesize());
        mul(data, other.data, newData, this->size());
        return Ntensor(newData, shape, const_cast<Ntensor *>(this), const_cast<Ntensor *>(&other), "*", grad);
    }
    Ntensor operator/(const Ntensor &other) const
    {
        PTR_TYPE(double, newData);
        MEM_AUTO_MALLOC(newData, this->bytesize());
        div(data, other.data, newData, this->size());
        return Ntensor(newData, shape, const_cast<Ntensor *>(this), const_cast<Ntensor *>(&other), "/", grad);
    }

    void backward(double *upper_grad)
    {
        PTR_TYPE(double, temp);
        MEM_AUTO_MALLOC(temp, this->bytesize());
        
        if (this->left != NULL)
        {
            if(this->op=="+")
            {
                add(this->left->grad, upper_grad, this->left->grad, this->size());
            }else if(this->op=="-"){
                add(this->left->grad, upper_grad, this->left->grad, this->size());
            }else if(this->op=="*"){
                mul(this->right->data, upper_grad, temp, this->size());
                add(this->left->grad, temp, this->left->grad, this->size());
            }else if(this->op=="/"){
                div(1.0, this->right->data, temp, this->size());
                mul(temp, upper_grad, temp, this->size());
                add(this->left->grad, temp, this->left->grad, this->size());
            }else if(this->op=="sigmoid"){
                GEN_VAR(double, left, this->size());
                GEN_VAR(double, right, this->size());
                MEM_CPY(left, this->data, this->bytesize());
                CALL_FUNC(n_exp, left, this->size());
                MEM_CPY(right, left, this->bytesize());
                SYNC;
                div(1, right, right, this->size());
                SYNC;
                mul(left, right, left, this->size());
                SYNC;
                mul(left, upper_grad, left, this->size());
                SYNC;
                add(left, this->grad, this->grad, this->size());
                SYNC;
                MEM_FREE(left);
                MEM_FREE(right);
            }else{
                
            }
        }

        if (this->right != NULL)
        {
            if(this->op=="+")
            {
                add(this->right->grad, upper_grad, this->right->grad, this->size());
            }else if(this->op=="-"){
                sub(this->right->grad, upper_grad, this->right->grad, this->size());
            }else if(this->op=="*"){
                mul(this->left->data, upper_grad, temp, this->size());
                add(this->right->grad, temp, this->right->grad, this->size());
            }else if(this->op=="/"){
                mul(this->right->data, this->right->data, temp, this->size());
                div(this->left->data, temp, temp, this->size());
                sub(0.0, temp, temp, this->size());
                mul(temp, upper_grad, temp, this->size());
                add(this->right->grad, temp, this->right->grad, this->size());
            }else{
                
            }
        }
        MEM_FREE(temp);
    }
};



MAKE_1OP(sigmoid)
{
    Ntensor res(a);
    size_t N=a.size();
    res.right=NULL;
    res.left=&a;
    res.op="sigmoid";
    sub(0.0,res.data,res.data,N);
    SYNC;
    CALL_FUNC(n_exp, res.data, N);
    SYNC;
    div(1.0, res.data, res.data, N);
    SYNC;
    return res;
}

MAKE_1OP(tanh)
{
    Ntensor res(a);
    size_t N=a.size();
    res.right=NULL;
    res.left=&a;
    res.op="tanh";
    CALL_FUNC(n_tanh, res.data, N);
    SYNC;
    return res;
}


#endif
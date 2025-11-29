#include<cuda_runtime.h>
#include<numeric>
#include<algorithm>
#include<chrono>
#include<iostream>
#include "Kernel.h"
#include "Matrix.cpp"


void run_test_spmm_kernel()
{

    //===========================================================SPMM=================================================
    tensor::Matrix<float> B(4096,4096);
    tensor::Matrix<float> C(4096,4096);
    B.InitMatrixRandom();
    C.filldata(0);

    tensor::BSR_Matrix<float> A(0.1,8,4096,4096,false);

    std::cout<<A.n_rows<<std::endl;

    CUDA_Kernel::SPMM_cuSparse(A.GetData(),A.GetRowoffset(),A.GetColIndex(),B.GetPtr(),C.GetPtr(),4096,4096,4096,A.n_rows);

    CUDA_Kernel::SPMM_88(A.GetData(),A.GetRowoffset(),A.GetColIndex(),A.GetSort(),B.GetPtr(),C.GetPtr(),4096,4096,4096,A.n_rows);

}

void run_test_flash_attention_kernel()
{

 //=======================================================flash Attention==============================================
    int N=196;
    int d=1024;
    tensor::Matrix<float>Q(N,d);
    tensor::Matrix<float>K(d,N);
    tensor::Matrix<float>V(N,d);
    tensor::Matrix<float>O(N,d);

    Q.InitMatrixRandom();
    K.InitMatrixRandom();
    V.InitMatrixRandom();
    O.filldata(0);
    
    CUDA_Kernel::FlashAttention(Q.GetPtr(),K.GetPtr(),V.GetPtr(),O.GetPtr(),N,d,1);

}

void run_test_sgemv_kernel()
{

    // =====================================================================SGMV
    int dim = 4096;
    tensor::Matrix<float>A(dim,dim);
    tensor::ArrayVec<float>x(dim);
    tensor::ArrayVec<float>y(dim);
    A.InitMatrixRandom();x.InitRandom();y.fill(0.0);

    CUDA_Kernel::SGEMV(A.GetPtr(),x.GetPtr(),y.GetPtr(),dim,dim);

}


void run_test_flesible_spmm_kernel()
{


    //=================================================FLES_SPMM=================================================
    int M = 4096;
    int N = 4096;
    int K = 4096;
    tensor::Matrix<float> B(K,N);
    tensor::Matrix<float> C(M,N);
    B.InitMatrixRandom();
    C.filldata(0);

    tensor::BSR_Matrix<float> A(0.1,4,M,K,false);

    std::cout<<A.n_rows<<std::endl;

    CUDA_Kernel::SPMM_Flesible(A.GetData(),A.GetRowoffset(),A.GetColIndex(),A.GetSort(),B.GetPtr(),C.GetPtr(),M,N,K,A.n_rows,4,4);
}


int main()
{
    run_test_flash_attention_kernel();
    return 0;
}
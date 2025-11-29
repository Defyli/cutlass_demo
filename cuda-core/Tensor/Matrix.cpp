#ifndef MATIRX_H
#define MATIRX_H

#include<memory>
#include<vector>
#include<utility>
#include<algorithm>
#include<numeric>
#include<cassert>
#include <random>
#include<iostream>
using namespace std;

namespace tensor{

    template<class T> class BSR_Matrix;

    template<class T>
    class Matrix{
    protected:
        shared_ptr<T[]>data;
        size_t padding_row = 0;
        size_t padding_col = 0;
    public:
        friend class BSR_Matrix<T>;
        static size_t PADDING_SIZE;
        size_t Col = 0;
        size_t Row = 0;

        Matrix()=default;
        Matrix(size_t N_ROW,size_t M_COL)
        {
            padding_row = N_ROW+(PADDING_SIZE-N_ROW%PADDING_SIZE);
            padding_col = M_COL+(PADDING_SIZE-M_COL%PADDING_SIZE);
            Col = M_COL;
            Row = N_ROW;
            data.reset(new T[padding_col*padding_row]);
        }
        Matrix(const Matrix&other)
        {
            if(this!=&other)
            {
                padding_col = other.padding_col;
                padding_row = other.padding_row;
                Col = other.Col;
                Row = other.Row;

                data.reset(new T[padding_col*padding_row]);
                //copy
                for(int i=0;i<Row;++i)
                {
                    for(int j=0;j<Col;++j)
                    {
                        data[i*padding_col+j] = other(i,j);
                    }
                }
            }
        }

        Matrix(Matrix&&other)
        {
            if(this!=&other)
            {
                data = std::move(other.data);
                padding_col = other.padding_col;
                padding_row = other.padding_row;
                Col = other.Col;
                Row = other.Row;

                other.Col = 0;other.Row = 0;
                other.padding_col = 0;other.padding_row = 0;
            }
        }

        void filldata(T value)
        {
            for(size_t i=0;i<Row;++i)
            {
                for(size_t j=0;j<Col;++j)
                    data[i*padding_col+j] = value;
            }
        }
        void InitMatrixRandom()
        {
            assert(data!=nullptr);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-100,100);
            for (size_t i = 0; i < Row; ++i)
            {
                for(size_t j =0;j < Col; ++j)
                    data[i*padding_col+j] = static_cast<T>(dist(gen));
            }
        }

        T&operator()(size_t i,size_t j)
        {
            if(i<Row&&j<Col)
            {
                assert(data!=nullptr);
                return data[i*padding_col+j];
            }
            else
                throw "index out of range";
        }

        T*GetPtr()
        {
            return data.get();
        }
    };

    template<class T>
    size_t Matrix<T>::PADDING_SIZE = 128;

    template<class T>
    class BSR_Matrix{
        private:
            shared_ptr<T[]>data;
            shared_ptr<size_t[]>col_index;
            shared_ptr<size_t[]>row_offset;  //row offset is based on the blocksize like data[row_offset[i]*blocksize*blocksize]
            shared_ptr<size_t[]>sorted_index; // sorted index is the real index
        public:
            size_t row,col;
            size_t num_elements;
            size_t num_block;
            int BlockSize;
            size_t n_rows;

            BSR_Matrix()=default;
            BSR_Matrix(Matrix<T>&dense,int block_size,vector<pair<size_t,size_t>>&block_index)
            {
                BlockSize = block_size;
                num_elements = block_index.size()*block_size*block_size;
                data.reset(new T[num_elements]);
                col_index.reset(new size_t[BlockSize*block_index.size()]);
                row_offset.reset(new size_t[(dense.Row)/block_size+1]);
                sorted_index.reset(new size_t[dense.Row/block_size]);
                row = dense.Row;
                col  = dense.Col;
                num_block = block_index.size();

                //copy data
                size_t count = 0;
                size_t col_count = 0;
                size_t row_count =0;
                size_t block_count = 0;
                size_t last_row = block_index[0].first;
                row_offset[0] = 0;
                sorted_index[0] = block_index[0].first;
                n_rows = 0;
                for(auto&index:block_index)
                {
                    size_t startrow = index.first;
                    size_t startcol = index.second;
                    for(size_t i = startrow;i<block_size;++i)
                    {
                        for(size_t j = startcol;j<block_size;++j)
                            data[count++] = dense(i,j);
                    }

                    //col index
                    for(int i=startcol;i<block_size;++i)col_index[col_count++] = i;
                    //row offset
                    if(index.first!=last_row){
                        ++row_count;
                        row_offset[row_count] = block_count;
                        sorted_index[row_count] = index.first;
                        ++n_rows;
                    }
                    last_row = index.first;
                    block_count++;
                }
                ++row_count;
                while(row_count<dense.Row){
                    row_offset[row_count] = block_count;
                    sorted_index[row_count] = 0;  //invaild
                    ++row_count;
                }
                row_offset[row_count] = block_count;
            }
            BSR_Matrix(float rate,int blocksize,size_t rowsize,size_t colsize,bool is_sort =true)
            {
                // according to sparse rate randomly init
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float>num_dist(-256.0,256.0);
                int block_bound = colsize/blocksize;


                num_block = rate*rowsize*colsize/(blocksize*blocksize);
                num_elements = num_block*blocksize*blocksize;
                row = rowsize;
                col = colsize;
                BlockSize = blocksize;
                n_rows = rowsize/blocksize;

                data.reset(new T[num_elements]);
                row_offset.reset(new size_t[n_rows+1]);
                col_index.reset(new size_t[num_block]);
                sorted_index.reset(new size_t[n_rows]);
                //fill data first
                for(int i=0;i<num_elements;++i)
                    data[i] = static_cast<T>(num_dist(gen));

                
                //fill  sort_index
                std::vector<size_t>possiable_row;
                for(size_t i=0;i<rowsize;i+=blocksize)
                    possiable_row.push_back(i);
                random_shuffle(possiable_row.begin(),possiable_row.end());
                for(int i=0;i<n_rows;++i)
                    sorted_index[i] = possiable_row[i];
                
                //fill row and col_index
                std::vector<size_t>possiable_col;
                for(size_t i=0;i<colsize;i+=blocksize)
                    possiable_col.push_back(i);

                row_offset[0] = 0;
                int gen_block = 0;
                size_t col_ptr=0;
                for(int i=1;i<n_rows;++i)
                {
                    int res = num_block-gen_block;
                    if(res<=0)
                    {
                        row_offset[i] = gen_block;
                        continue;
                    }
                    
                    std::uniform_int_distribution blcok(1+(block_bound-1)*3/4,block_bound);
                    int block_num = min(max(blcok(gen),0),res);
                    gen_block += block_num;
                    row_offset[i] = gen_block;

                    //col_index;
                    random_shuffle(possiable_col.begin(),possiable_col.end());
                    for(int k = 0;k<block_num;++k)
                    {
                        size_t start = possiable_col[k];
                        col_index[col_ptr++] = start; 
                        // for(int j=0;j<blocksize;++j)
                        // {
                        //     col_index[col_ptr++] = start+j;
                        // }
                    }
                    if(is_sort)block_bound = block_num;
                    else block_bound = possiable_col.size();                 
                }
                row_offset[rowsize/blocksize] = gen_block;
            }
            T*GetData()
            {
                return data.get();
            }
            size_t*GetRowoffset()
            {
                return row_offset.get();
            }
            size_t*GetColIndex()
            {
                return col_index.get();
            }
            size_t*GetSort()
            {
                return sorted_index.get();
            }
    };


    template<class T>
    class ArrayVec{
        private:
        shared_ptr<T[]>data;
        size_t padding_size;
        public:
        static int PADDING;
        size_t size;

        ArrayVec()=default;
        ArrayVec(size_t n)
        {
            padding_size = n + (PADDING-n%PADDING);
            data.reset(new T[padding_size]);
            size = n;
        }
        ArrayVec(const ArrayVec&other)
        {
            if(this!=&other)
            {
                data.reset(new T[other.padding_size]);
                padding_size = other.padding_size;
                std::copy(other.data,other.data+padding_size,data);
                size = other.size;
            }
        }
        ArrayVec(ArrayVec&&other)
        {
            if(this!=other)
            {
                data = other.data;
                padding_size = other.padding_size;
                size = other.size;
            }
        }
        T&operator()(size_t index)
        {
            assert(index<size);
            return data[index];
        }
        T*GetPtr()
        {
            return data.get();
        }
        void InitRandom()
        {
            assert(data!=nullptr);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-100,100);
            for(int i=0;i<size;++i)
            {
                data[i] = static_cast<T>(dist(gen));
            }
        }
        void fill(T v)
        {
            assert(data!=nullptr);
            for(int i=0;i<size;++i)
                data[i] = v;
        }
    };

    template<class T>
    int ArrayVec<T>::PADDING =16;

};





#endif
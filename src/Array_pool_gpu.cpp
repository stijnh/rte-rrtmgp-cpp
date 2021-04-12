/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#include "Array_pool_gpu.h"
#include <tuple>

template<typename TF>
Array_pool_gpu<TF>::Array_pool_gpu(
    const std::vector<int>& block_sizes_):lock(false),alloc_counter(0),
    alloc_bytes(0),reuse_counter(0),reuse_bytes(0)
{
    std::vector<int> input_blocks(block_sizes_);
    std::sort(input_blocks.begin(), input_blocks.end());
    for(int i=0;i<input_blocks.size() - 1; ++i)
    {
        if(input_blocks[i] < 0.7 * input_blocks[i + 1])
        {
            block_sizes.push_back(input_blocks[i]);
        }
    }
    block_sizes.push_back(input_blocks.back());
    blocks.resize(block_sizes.size());
}

template<typename TF>
Array_pool_gpu<TF>::~Array_pool_gpu()
{
    for(typename memory_storage_type::iterator it = blocks.begin(); it != blocks.end(); ++it)
    {
#ifdef __CUDACC__
        for(typename std::list<TF*>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
        {
            cuda_safe_call(cudaFree(*it2));
        }
#endif
        it->clear();
    }
    blocks.clear();
}

template<typename TF>
std::tuple<typename Array_pool_gpu<TF>::memory_storage_type::iterator, int> Array_pool_gpu<TF>::lookup_block_list(int size)
{
    auto it1 = blocks.begin();
    auto it2 = block_sizes.begin();
    while(size > *it2 and it2 != block_sizes.end())
    {
        it1++;
        it2++;
    }
    return std::make_tuple(it1, *it2);
}


template<typename TF>
void Array_pool_gpu<TF>::acquire_memory(TF*& block_, int size_)
{
    auto lookup = lookup_block_list(size_);
    
    if(std::get<0>(lookup) == blocks.end())
    {
        throw std::out_of_range("requested block size not supported by pool");
    }

    auto block_it = std::get<0>(lookup);
    int block_size = std::get<1>(lookup);

    bool alloc = (block_it->size() == 0);
    if(alloc)
    {
        auto it = block_it;
        auto it2 = block_sizes.begin() + (block_it - blocks.begin());
//TODO: Review whether this is a responsible (and necessary) thing to do...
        while(it != blocks.end() and it->size()==0)
        {
            ++it;
            ++it2;
        }
        if(it != blocks.end())
        {
            alloc = false;
            block_it = it;
            block_size = *it2;
        }
    }

    if(alloc)
    {
//        std::cerr<<"alloc "<<size_<<"<=>"<<block_size<<std::endl;
        if(lock)
        {
            throw std::bad_alloc();
        }
        
#ifdef __CUDACC__
        cuda_safe_call(cudaMalloc((void **) &block_, block_size*sizeof(TF)));
#endif
        alloc_counter += 1;
        alloc_bytes += (block_size * sizeof(TF));
    }
    else
    {
//        std::cerr<<"re-use "<<size_<<"<=>"<<block_size<<std::endl;
        block_ = block_it->back();
        block_it->pop_back();
        reuse_counter += 1;
        reuse_bytes += (block_size * sizeof(TF));
    }
}

template<typename TF>
void Array_pool_gpu<TF>::release_memory(TF*& block_, int size_)
{
    auto lookup = lookup_block_list(size_);

    if(std::get<0>(lookup) == blocks.end())
    {
        throw std::out_of_range("requested block size not supported by pool");
    }
    
    auto it = std::get<0>(lookup);

    it->push_back(block_);
    block_ = nullptr;
}

template<typename TF>
void Array_pool_gpu<TF>::print_stats(std::ostream& os) const
{
    os<<"Allocations:\t"<<alloc_counter<<"\t"<<alloc_bytes<<" bytes"<<std::endl;
    os<<"Recycled:   \t"<<reuse_counter<<"\t"<<reuse_bytes<<" bytes"<<std::endl;
}


#ifdef FLOAT_SINGLE_RRTMGP
template class Array_pool_gpu<float>;
#else
template class Array_pool_gpu<double>;
#endif

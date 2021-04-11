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

#include "Block_vector_pool.h"
#include <tuple>

template<typename TF>
Block_vector_pool<TF>::Block_vector_pool(
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
std::tuple<typename Block_vector_pool<TF>::memory_storage_type::iterator, int> Block_vector_pool<TF>::lookup_block_list(int size)
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
void Block_vector_pool<TF>::acquire_memory(std::vector<TF>& block_, int size_)
{
    if(block_.size() >= size_) return;

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
        while(it != blocks.end() and it->size()==0)
        {
            ++it;
        }
        if(it != blocks.end())
        {
            alloc = false;
            block_it = it;
            block_size = block_it->begin()->size();
        }
    }

    if(alloc)
    {
//        std::cerr<<"alloc "<<size_<<"<=>"<<block_size<<std::endl;
        if(lock)
        {
            throw std::bad_alloc();
        }
        block_.resize(block_size);
        alloc_counter += 1;
        alloc_bytes += (block_size * sizeof(TF));
    }
    else
    {
//        std::cerr<<"re-use "<<size_<<"<=>"<<block_size<<std::endl;
        block_ = std::move(block_it->back());
        block_it->pop_back();
        reuse_counter += 1;
        reuse_bytes += (block_size * sizeof(TF));
    }
}

template<typename TF>
void Block_vector_pool<TF>::release_memory(std::vector<TF>& block_, int size_)
{
    if(block_.empty()) return;

    auto lookup = lookup_block_list(block_.size());

    if(std::get<0>(lookup) == blocks.end())
    {
        throw std::out_of_range("requested block size not supported by pool");
    }
    
    std::list<std::vector<TF>>& block_list = *(std::get<0>(lookup));

    block_list.push_back(std::vector<TF>());
    block_list.back() = std::move(block_);
}

template<typename TF>
void Block_vector_pool<TF>::print_stats(std::ostream& os) const
{
    os<<"Allocations:\t"<<alloc_counter<<"\t"<<alloc_bytes<<" bytes"<<std::endl;
    os<<"Recycled:   \t"<<reuse_counter<<"\t"<<reuse_bytes<<" bytes"<<std::endl;
}


#ifdef FLOAT_SINGLE_RRTMGP
template class Array_memory_pool<float>;
#else
template class Block_vector_pool<double>;
#endif

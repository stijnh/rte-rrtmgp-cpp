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
 *z
 */

#ifndef ARRAY_MEM_POOL_H
#define ARRAY_MEM_POOL_H

#include <list>
#include <vector>
#include <iostream>
#include "Array.h"


template<typename TF>
class Block_vector_pool: public Pool_base<std::vector<TF>>
{
    public:

        bool lock;

        typedef std::vector<std::list<std::vector<TF>>> memory_storage_type;

        Block_vector_pool(const std::vector<int>& block_sizes_);

        void acquire_memory(std::vector<TF>& block_, int size_);

        void release_memory(std::vector<TF>& block_, int size_);

        void print_stats(std::ostream& os) const;

    private:

        std::tuple<typename memory_storage_type::iterator, int> lookup_block_list(int size);

        std::vector<int> block_sizes;
        memory_storage_type blocks;

        int alloc_counter;
        int alloc_bytes;
        int reuse_counter;
        int reuse_bytes;
};

#endif

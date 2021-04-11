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

#ifndef POOL_BASE_H
#define POOL_BASE_H

#include <set>

template<typename T>
class Pool_base
{
    public:

        virtual void acquire_memory(T& storage, int num_elems)=0;

        virtual void release_memory(T& block, int num_elems)=0;

        virtual ~Pool_base(){}
};

template<typename T>
class Pool_client
{
    public:

        Pool_client():pool(nullptr){}

        Pool_client(Pool_base<T>* pool_):pool(pool_){}

        Pool_client(const Pool_client<T>& other):pool(nullptr){}

//        Pool_client(Pool_client<T>&& other):pool(std::exchange(other.pool, nullptr)){}

        virtual ~Pool_client(){}

        void acquire_memory()
        {
            if(pool == nullptr)
            {
                allocate_memory(get_memory(), get_num_elements());
            }
            else
            {
                pool->acquire_memory(get_memory(), get_num_elements());
            }
        }

        void release_memory()
        {
            if(pool == nullptr)
            {
                deallocate_memory(get_memory(), get_num_elements());
            }
            else
            {
                pool->release_memory(get_memory(), get_num_elements());
            }
        }

        bool is_pooled() const
        {
            return pool != nullptr;
        }

    protected:

        virtual T& get_memory()=0;

        virtual int get_num_elements() const=0;

        virtual void allocate_memory(T& memory, int num_elems) const=0;

        virtual void deallocate_memory(T& memory, int num_elems) const=0;

    private:

        Pool_base<T>* pool;
};

template<typename T>
class Pool_client_group
{
    public:

        Pool_client_group():pool(nullptr){}

        Pool_client_group(Pool_base<T>* pool_):pool(pool_){}

        Pool_client_group(const Pool_client_group<T>& other):pool(other.pool),clients(other.clients){}

        virtual ~Pool_client_group(){}

        void add_client(Pool_client<T>& client)
        {
            clients.insert(&client);
        }

        void remove_client(Pool_client<T>& client)
        {
            clients.erase(&client);
        }

        void acquire_memory()
        {
            for(auto it = clients.begin(); it != clients.end(); ++it)
            {
                if((*it)->is_pooled())
                {
                    (*it)->acquire_memory();
                }
            }
        }

        void release_memory()
        {
            for(auto it = clients.begin(); it != clients.end(); ++it)
            {
                if((*it)->is_pooled())
                {
                    (*it)->release_memory();
                }
            }
        }

    protected:

        Pool_base<T>* pool;

    private:

        std::set<Pool_client<T>*> clients;
};

#endif

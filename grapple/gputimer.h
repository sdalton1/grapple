/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// A simple timer class

#include <cuda.h>

class timer
{
  private:
    cudaEvent_t start;
    cudaEvent_t end;

  public:
    timer(void)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    }

    ~timer(void)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    void Start(void)
    {
        cudaEventRecord(start, 0);
    }

    void Stop(void)
    {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
    }

    float milliseconds_elapsed(void)
    {
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, end);

        return elapsed_time;
    }

    float seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};


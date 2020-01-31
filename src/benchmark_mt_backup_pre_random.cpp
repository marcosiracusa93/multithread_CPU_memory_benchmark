#include <omp.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <time.h>
#include <chrono>
#include <cstdlib>
#include <exception>

#define TYPE unsigned int
#define READ 1
#define WRITE 1
#define READWRITE 1

int main(int argc, char* argv[]) 
{
    if (argc != 4) {
        printf("wrong usage: specify num thread, access length\n");
        exit(-1);
    }
    
    //clock_t start1, end1;
    //time_t start2, end2;
    //struct timespec start, end;
    
    unsigned long long max_num_threads = strtol(argv[1], NULL, 10);
    unsigned long long max_num_transactions = strol(argv[2], NULL, 10);
    unsigned long long max_access_length = strtol(argv[2], NULL, 10);

    printf("Running with MNT=%llu and MAL= %llu\n", max_num_threads, max_access_length);

    printf("mquanta[B], num_threads, length, memsize[B], r_time[s], w_time[s], rw_time[s], r_bw[B/s], w_bw[B/s], rw_bw[B/s]\n");
   
    double read_time = 0;
    double write_time = 0;
    double readwrite_time = 0;

    for (unsigned long long num_threads = max_num_threads; num_threads <= max_num_threads; num_threads = num_threads << 1) {
        for (unsigned long long access_length = 1; access_length <= max_access_length; access_length = access_length << 1) {
             
            TYPE *mat = (TYPE *)malloc(num_threads * access_length * sizeof(TYPE));
            double memsize = (double)num_threads * (double)access_length * (double)sizeof(TYPE);
            //TYPE **mat = (TYPE **)malloc(num_threads * sizeof(TYPE*));
            //for(int i = 0; i < num_threads; i++) mat[i] = (TYPE *)malloc(access_length * sizeof(TYPE));
            //double memsize = (double)num_threads * (double)access_length * (double)sizeof(TYPE);
            
	    TYPE val = std::rand();

            #pragma omp parallel for shared(mat) num_threads(num_threads) schedule(dynamic,1)
            for (unsigned long long i = 0; i < num_threads; ++i) {
                for (unsigned long long j = 0; j < access_length; ++j) {
		    TYPE init = 0;
		    unsigned long long idx = i * access_length + j;
		    if (idx % val == 0) {
		        init = val;
		    }
                    mat[idx] = init;
                }
            }
            
            //omp_set_dynamic(0);     		// Explicitly disable dynamic teams
            //omp_set_num_threads(num_threads); 	// Use # threads for all consecutive parallel regions    
            
#if defined READ
#if READ != 0
            {
                std::chrono::system_clock::time_point begin, end;
                
                //printf("Reading...\n");
                begin = std::chrono::high_resolution_clock::now();
                //clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                
                #pragma omp parallel for shared(mat) num_threads(num_threads) schedule(dynamic,1)
                for (unsigned long long i = 0; i < num_threads; ++i) {
		    unsigned long long row_idx = i * access_length;

                    TYPE v = 0;
                    for (unsigned long long j = 0; j < access_length; ++j) {
                        v += mat[row_idx + j];
                    }
                    
                    if (v == 0) {
                        val = v;
                    }
                }
                
                end = std::chrono::high_resolution_clock::now();
                //clock_gettime(CLOCK_MONOTONIC_RAW, &end);
                
                auto delta_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin); 
                read_time = (double)delta_us.count() * 1e-9;
            }
#endif
#endif
            
#if defined WRITE
#if WRITE != 0
            {
                std::chrono::system_clock::time_point begin, end;
                
                //printf("Writing...\n");
                begin = std::chrono::high_resolution_clock::now();
                //clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                
                #pragma omp parallel for shared(mat) num_threads(num_threads) schedule(dynamic,1)
                for (unsigned long long i = 0; i < num_threads; ++i) {
		    unsigned long long row_idx = i * access_length;

                    for (unsigned long long j = 0; j < access_length; ++j) {
                        mat[row_idx + j] = 1;
                    }
                }
                
                end = std::chrono::high_resolution_clock::now();
                //clock_gettime(CLOCK_MONOTONIC_RAW, &end);
                
                auto delta_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin); 
                write_time = (double)delta_us.count() * 1e-9;
            }
#endif
#endif

#if defined READWRITE
#if READWRITE != 0
            {
                std::chrono::system_clock::time_point begin, end;
                
                //printf("ReadingWriting...\n");
                begin = std::chrono::high_resolution_clock::now();
                //clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                
                #pragma omp parallel for shared(mat) num_threads(num_threads) schedule(dynamic,1)
                for (unsigned long long i = 0; i < num_threads; ++i) {
		    unsigned long long row_idx = i * access_length;
                    for (unsigned long long j = 0; j < access_length; ++j) {
                        mat[row_idx + j] += 1;
                    }
                }
                
                end = std::chrono::high_resolution_clock::now();
                //clock_gettime(CLOCK_MONOTONIC_RAW, &end);
                
                auto delta_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin); 
                readwrite_time = (double)delta_us.count() * 1e-9;
            }
#endif
#endif
            
	    printf("%llu %llu %llu %.3e %.3e %.3e %.3e %.3e %.3e %.3e\n", sizeof(TYPE), num_threads, access_length, memsize, read_time, write_time, readwrite_time, memsize / read_time, memsize / write_time, memsize / readwrite_time);
   
            val += mat[(val % num_threads) * access_length + (val % access_length)];
            if (val == 0) {
                sleep(1);
            }
            
            free(mat);
        }
    }
    
    //printf("CYCLES %.3e\n", clock_cycles);
    //printf("TIME %.3e\n", cpu_time_used);
    //printf("BANDWIDTH [B/cc] %.3e\n", memsize / clock_cycles);
} 

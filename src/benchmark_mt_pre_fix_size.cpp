#include <omp.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <time.h>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <fstream>

#define TYPE double
#define ENABLE_READ 1
#define ENABLE_WRITE 1
#define ENABLE_READWRITE 1
#define SHUFFLE_THREADS 0
#define REVERSE_THREADS 0
#define SHUFFLE_TRANSACTIONS 0
#define REVERSE_TRANSACTIONS 0
#define SHUFFLE_ACCESSES 0
#define REVERSE_ACCESSES 0
#define SHUFFLE_ALL 0
#define TEST_RANDOMICITY 0

#define STR(x)   #x
#define PRINT_DEFINE(x) printf("%s=%s\n", #x, STR(x))
#define get_bw(b,t) (t == 0 ? "x" : std::to_string(b / t))

typedef enum {TEST, READ, WRITE, READWRITE} mode_ty;

void print_configuration() {
    PRINT_DEFINE(TYPE);
    PRINT_DEFINE(ENABLE_READ);
    PRINT_DEFINE(ENABLE_WRITE);
    PRINT_DEFINE(ENABLE_READWRITE);
    PRINT_DEFINE(SHUFFLE_THREADS);
    PRINT_DEFINE(REVERSE_THREADS);
    PRINT_DEFINE(SHUFFLE_TRANSACTIONS);
    PRINT_DEFINE(REVERSE_TRANSACTIONS);
    PRINT_DEFINE(SHUFFLE_ACCESSES);
    PRINT_DEFINE(REVERSE_ACCESSES);
    PRINT_DEFINE(TEST_RANDOMICITY);
}

inline unsigned int get_idx(unsigned int rnd, unsigned long tid, unsigned long ii, unsigned long jj, 
		unsigned long num_threads, unsigned long num_transactions, unsigned long access_length) {
	#if defined SHUFFLE_THREADS and SHUFFLE_THREADS != 0
	tid = _rotl(rnd, tid) % num_threads;
	#endif

	#if defined SHUFFLE_TRANSACTIONS and SHUFFLE_TRANSACTIONS != 0
	ii = _rotl(rnd, ii) % num_transactions;
	#endif

	#if defined SHUFFLE_ACCESSES and SHUFFLE_ACCESSES != 0
	jj = _rotl(rnd, jj) % access_length;
	#endif

	#if defined SHUFFLE_ALL and SHUFFLE_ALL != 0
	return tid * num_transactions * access_length + ii * access_length + jj; // TODO add randomize all
	#else
	return tid * num_transactions * access_length + ii * access_length + jj;
	#endif
}

#pragma optimization_level 0
inline void read_fun(TYPE *mat, unsigned long idx, TYPE &v) {
    v += mat[idx];
}

#pragma optimization_level 0
inline void write_fun(TYPE *mat, unsigned long idx, TYPE &v) {
    mat[idx] = 1;
}

#pragma optimization_level 0
inline void readwrite_fun(TYPE *mat, unsigned long idx, TYPE &v) {
    mat[idx] += 1;
}

inline void multithread_benchmark(mode_ty mode, TYPE *mat, TYPE &val,
				  unsigned long num_threads, unsigned long num_transactions, unsigned long access_length,
				  double &mode_max_time, double &mode_avg_time, double &mode_var_time) {

    double max_time = 0;
    double time_sum = 0;
    double time_sum_of_squares = 0;

    #pragma omp parallel shared(mat) shared(val) num_threads(num_threads) reduction(max : max_time) reduction(+ : time_sum) reduction(+ : time_sum_of_squares)
    {
    //for (unsigned int tid = 0; tid < num_threads; ++tid) {
	unsigned long tid = omp_get_thread_num();
	#if defined SHUFFLE_THREADS and SHUFFLE_THREADS != 0
	//tid = _rotl(rnd, tid) % num_threads;
	#endif

	std::chrono::system_clock::time_point begin2, end2;
	begin2 = std::chrono::high_resolution_clock::now();

	register TYPE v = val;

	#if defined REVERSE_TRANSACTIONS and REVERSE_TRANSACTIONS != 0
  	for (unsigned long i = num_transactions; i > 0; --i) {
	    unsigned long ii = i - 1;
	#else
	for (unsigned long i = 0; i < num_transactions; ++i) {
	    unsigned long ii = i;
	#endif

	    #if defined ENABLE_WRITE and ENABLE_WRITE != 0
	    #pragma vector nontemporal(mat)
            #endif

	    #if defined SHUFFLE_TRANSACTIONS and SHUFFLE_TRANSACTIONS != 0
	    //ii = _rotl(rnd, ii) % num_transactions;
	    #endif
	    //unsigned long row_idx = tid * num_transactions * access_length + ii * access_length;

	    #if defined REVERSE_ACCESSES and REVERSE_ACCESSES != 0
	    for (unsigned long j = access_length; j > 0; --j) {
		unsigned long jj = j - 1;
	    #else
	    for (unsigned long j = 0; j < access_length; ++j) {
		unsigned long jj = j;
	    #endif
		#if defined SHUFFLE_ACCESSES and SHUFFLE_ACCESSES != 0
		//jj = _rotl(rnd, jj) % access_length;
		#endif

		unsigned long idx = get_idx(0, tid, ii, jj, num_threads, num_transactions, access_length);

		#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
		std::stringstream stream;
		stream << sizeof(TYPE) << " " << num_threads << " " << num_transactions << " " << access_length << " " << "x" << " " << omp_get_thread_num() << " " << tid << " " << i << " " << ii << " " << j << " " << jj << " " << idx << std::endl;
		std::cout << stream.str();
		#else
		
		switch (mode) {
		    case READ:
		    	v += mat[idx];
			//read_fun(mat, idx, v);
		    break;
		    case WRITE:
		        mat[idx] = v;
			//write_fun(mat, idx, v);
		    break;
		    case READWRITE:
		    	mat[idx] += v;
			//readwrite_fun(mat, idx, v);
		    break;
		}
		
		#endif
	    }
	}
	end2 = std::chrono::high_resolution_clock::now();
	
	val = v;

	auto delta_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
	double read_time = (double)delta_us.count() * 1e-9;

	max_time = std::max(max_time, read_time);
	time_sum += read_time;
	time_sum_of_squares += std::pow(read_time, 2);
    } // END omp parallel

    mode_max_time = max_time;
    mode_avg_time = time_sum / num_threads;
    mode_var_time = (time_sum_of_squares - std::pow(time_sum, 2) / num_threads) / num_threads;
} // END multithreaded_benchmark

inline void initialize(TYPE *mat, unsigned long rnd_input, TYPE val, unsigned long num_threads, unsigned long num_transactions, unsigned long access_length) {
    #pragma omp parallel shared(mat) num_threads(num_threads)
    {
        //for (unsigned int tid = 0; tid < num_threads; ++tid) {
    	unsigned long tid = omp_get_thread_num();
	unsigned int mod = (val < 2 ? 2 : (unsigned int)val);
	for (unsigned long j = 0; j < num_transactions; ++j) {
	    for (unsigned long k = 0; k < access_length; ++k) {
	        unsigned long idx = tid * num_transactions * access_length + j * access_length + k ;
		TYPE init = (idx*tid*j*k) % mod;

	        if (/*rnd_input != 0 and*/ idx % (unsigned int)val == 0) {
		    init = val*tid*j*k;
	        } 
	    	mat[idx] = init;
		//std::string toprint = std::to_string(idx) + " " + std::to_string(init) + "\n";
		//std::cout << toprint; // TODO remove it
	    }
        }
    }
}

void read_file(std::vector<unsigned int> &words) {
    std::ifstream infile("input.txt", std::ios::in);
    if (infile.is_open()) {
	std::string word;
        while (infile >> word) {
            words.push_back(std::stoi(word));
        }
        infile.close();
    } else {
	std::cout << "Unable to open file"; 
	exit(-1);
    }
}

void write_file(std::vector<unsigned int> &values) {
    std::ofstream outfile("output.txt", std::ios::out);
    if (outfile.is_open()) {
        for (unsigned int v : values) {
            outfile << v;
        }
        outfile.close();
    } else {
        std::cout << "Unable to open file";
        exit(-1);
    }
}

int main(int argc, char* argv[]) 
{
    if (argc != 5) {
        printf("wrong usage: specify num thread, num_transactions per thread, access length per transaction, rnd_input\n");
        exit(-1);
    }
    
    print_configuration();
    //clock_t start1, end1;
    //time_t start2, end2;
    //struct timespec start, end;
    
    srand(time(NULL));
    unsigned int rnd = std::rand();
    
    unsigned long max_num_threads = strtol(argv[1], NULL, 10);
    unsigned long max_num_transactions = strtol(argv[2], NULL, 10);
    unsigned long max_access_length = strtol(argv[3], NULL, 10);
    unsigned long rnd_input = strtol(argv[4], NULL, 10);

    std::string running_mode = "Running";
#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
    running_mode = "Testing randomicity";
#endif

    std::cout << running_mode << " with MNTh=" << max_num_threads << " and MNTr=" << max_num_transactions << " and MAL=" << max_access_length << " and Ri=" << rnd_input << std::endl;

#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
    std::cout << "mquanta[B], num_threads, num_transactions, length, memsize[B], thread_num, tid, i, ii, j, jj, access_idx" << std::endl;
#else
    std::cout << "mquanta[B], num_threads, num_transactions, length, memsize[B], "
	      << "read_max_time[s], read_avg_time[s], read_var_time, "
	      << "write_max_time[s], write_avg_time[s], write_var_time, "
	      << "readwrite_max_time[s], readwrite_avg_time[s], readwrite_var_time, "
	      << "read_min_bw[B/s], read_avg_bw[B/s], "
              << "write_min_bw[B/s], write_avg_bw[B/s], "
              << "readwrite_min_bw[B/s], readwrite_avg_bw[B/s], " << std::endl;
#endif

    double read_max_time = 0;
    double read_avg_time = 0;
    double read_var_time = 0;
    double write_max_time = 0;
    double write_avg_time = 0;
    double write_var_time = 0;
    double readwrite_max_time = 0;
    double readwrite_avg_time = 0;
    double readwrite_var_time = 0;

    std::cout << "Allocating matrix." << std::endl;
    TYPE *mat = (TYPE *)malloc(max_num_threads * max_num_transactions * max_access_length * sizeof(TYPE));
    std::vector<unsigned int> words, values;
    std::cout << "Loading the file." << std::endl;
    read_file(words);
    std::cout << "Initializing matrix..." << std::endl;
    for (unsigned long i = 0; i < max_num_threads * max_num_transactions * max_access_length; ++i) {
        unsigned long idx = i % words.size();
        mat[i] = words[idx];
    }
    std::cout << "Initialization done." << std::endl;

    for (unsigned long num_threads = max_num_threads; num_threads <= max_num_threads; num_threads = num_threads << 1) {
        for (unsigned long num_transactions = 1; num_transactions <= max_num_transactions; num_transactions = num_transactions << 1) {
            for (unsigned long access_length = 1; access_length <= max_access_length; access_length = access_length << 1) {
                //TYPE *mat = (TYPE *)malloc(num_threads * num_transactions * access_length * sizeof(TYPE));
                double memsize = (double)num_threads * (double)num_transactions * (double)access_length * (double)sizeof(TYPE);

                TYPE val = std::rand();

		#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
                    multithread_benchmark(TEST, mat, rnd, rnd_input, val, num_threads, num_transactions, access_length, read_max_time, read_avg_time, read_var_time);
                #else

			#if defined ENABLE_READ and ENABLE_READ != 0
			    multithread_benchmark(READ, mat, val, num_threads, num_transactions, access_length, read_max_time, read_avg_time, read_var_time);
			#endif

			#if defined ENABLE_WRITE and ENABLE_WRITE != 0
			    multithread_benchmark(WRITE, mat, val, num_threads, num_transactions, access_length, write_max_time, write_avg_time, write_var_time);
			#endif

			#if defined ENABLE_READWRITE and ENABLE_READWRITE != 0
			    multithread_benchmark(READWRITE, mat, val, num_threads, num_transactions, access_length, readwrite_max_time, readwrite_avg_time, readwrite_var_time);
			#endif

			//#if not defined TEST_RANDOMICITY or TEST_RANDOMICITY == 0
				std::cout << sizeof(TYPE) << " " << num_threads << " " << num_transactions << " " << access_length << " " << memsize << " " 
				   << read_max_time << " " << read_avg_time << " " << read_var_time << " " 
				   << write_max_time << " " << write_avg_time << " " << write_var_time << " " 
				   << readwrite_max_time << " " << readwrite_avg_time << " " << readwrite_var_time << " "
				   << get_bw(memsize, read_max_time) << " " << get_bw(memsize, read_avg_time) << " " 
				   << get_bw(memsize, write_max_time) << " " << get_bw(memsize, write_avg_time) << " " 
				   << get_bw(2*memsize, readwrite_max_time) << " " << get_bw(2*memsize, readwrite_avg_time) << std::endl;
			//#endif
			
			for(unsigned int i = 0; i < (unsigned int)val % 3; ++i) {
			    val += mat[((unsigned int)val % num_threads) * access_length + ((unsigned int)val % access_length)];
			}
			if (val == 0) {
			    values.push_back(val);
			    std::cerr << "P";
			}
            	#endif
            }
        }
    }

    free(mat);
    std::cout << "Writing results..." << std::endl;
    write_file(values);
    std::cout << "Results written." << std::endl;
} 

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

/*
 *
 * The benchmark reads/writes/copies a dataset subdivided in memory blocks, memory segments and memory accesses to first class type elements.
 * Each THREAD accesses to a MEMORY BLOCK which can be assigned deterministically or randomly.
 * If no randomization is involved on these thread-memory_block mapping, each thread accesses to a single memory block. 
 * If memory blocks are assigned randomly, this assumption falls. Future works will control the maximum number of threads conflicting measuring the overhead.
 * Each memory block is then divided in MEMORY SEGMENTS composed of continuous ELEMENTS.
 * Memory segments represent the spatial locality we want to measure. Each thread can acces memory segments within its memory block in a sequential or random order.
 * If these indexes are randomized, there is no particular order the memory segments within a memory block are accessed.
 * In case no randomization is applied, the order the segments are accessed one next to another (forward or backward).
 * Moreover, the randomization could be index based (compie-time resolvable) or data based (run-time resolvable).
 * Each segment is composed by contiguous ELEMENTS to be accessed randomly or sequentially (forward or backward).
 *
 * With this structure we measure the MEMORY BANDWIDTH when varying the SPATIAL LOCALITY the access pattern has.
 *
 */

#define TYPE double                     // defines the MEMORY QUANTA: type of each memory element we want to access
#define ENABLE_READ 1                   // enables the execution of the read-only benchmark
#define ENABLE_WRITE 1                  // enables the execution of the write-only benchmark
#define ENABLE_READWRITE 1              // enables the execution of the read-write benchmark
#define SHUFFLE_THREADS 0               // NOT USED BY NOW
#define REVERSE_SEGMENT_IDXS 0          // Access the segments in reverse order (backward)
#define SHUFFLE_SEGMENT_IDXS 0          // Access the segments in random index-based order
#define GATHER_SCATTER_SEGMENT_IDXS 0   // Access the segments in random data-based order
#define REVERSE_ELEMENT_IDXS 0          // Access the elements in reverse order (backward)
#define SHUFFLE_ELEMENT_IDXS 0          // Access the elements in random index-based order
#define SHUFFLE_ALL 0                   // NOT USED BY NOW
#define TEST_RANDOMICITY 0              // Prints out the used indexes in order to test randomicity

// Utility macros for formatted printing
#define STR(x)   #x
#define PRINT_DEFINE(x) printf("%s=%s\n", #x, STR(x))
#define get_bw(b,t) (t == 0 ? 0 : (double)b / (double)t)

// Enumerates the running mode
typedef enum {TEST, READ, WRITE, READWRITE} mode_ty;

// Prints the variables defining the running configuration
void print_configuration() {
    PRINT_DEFINE(TYPE);
    PRINT_DEFINE(ENABLE_READ);
    PRINT_DEFINE(ENABLE_WRITE);
    PRINT_DEFINE(ENABLE_READWRITE);
    PRINT_DEFINE(SHUFFLE_THREADS);
    PRINT_DEFINE(REVERSE_SEGMENT_IDXS);
    PRINT_DEFINE(SHUFFLE_SEGMENT_IDXS);
    PRINT_DEFINE(GATHER_SCATTER_SEGMENT_IDXS);
    PRINT_DEFINE(REVERSE_ELEMENT_IDXS);
    PRINT_DEFINE(SHUFFLE_ELEMENT_IDXS);
    PRINT_DEFINE(SHUFFLE_ALL);
    PRINT_DEFINE(TEST_RANDOMICITY);
}

/*
// Compute the final index given the provided indexes
inline unsigned int get_idx(unsigned int rnd, unsigned long tid, unsigned long ii, unsigned long jj, 
		unsigned long num_threads, unsigned long num_segments, unsigned long num_elements) {
	#if defined SHUFFLE_THREADS and SHUFFLE_THREADS != 0
	tid = _rotl(rnd, tid) % num_threads;
	#endif

	#if defined SHUFFLE_SEGMENT_IDXS and SHUFFLE_SEGMENT_IDXS != 0
	ii = _rotl(rnd, ii) % num_segments;
	#endif

	#if defined SHUFFLE_ELEMENT_IDXS and SHUFFLE_ELEMENT_IDXS != 0
	jj = _rotl(rnd, jj) % num_elements;
	#endif

	#if defined SHUFFLE_ALL and SHUFFLE_ALL != 0
	return tid * num_segments * num_elements + ii * num_elements + jj; // TODO add randomize all
	#else
	return tid * num_segments * num_elements + ii * num_elements + jj;
	#endif
}

// Core operations optimizations should be avoided for
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
*/

void multithread_benchmark(mode_ty mode, TYPE *mat, TYPE &val,
				  unsigned long num_threads, unsigned long num_segments, unsigned long num_elements,
				  double &mode_max_time, double &mode_avg_time, double &mode_var_time, unsigned long rnd_input) {

    double max_time = 0;
    double time_sum = 0;
    double time_sum_of_squares = 0;

    #pragma omp parallel shared(mat) shared(val) num_threads(num_threads) reduction(max : max_time) reduction(+ : time_sum) reduction(+ : time_sum_of_squares)
    {
    	unsigned long tid = omp_get_thread_num();
#if defined SHUFFLE_THREADS and SHUFFLE_THREADS != 0
    	//tid = _rotl(rnd, tid) % num_threads;
#endif

    	std::chrono::system_clock::time_point begin_time, end_time;
    	begin_time = std::chrono::high_resolution_clock::now();

    	register TYPE v = val;

#if defined GATHER_SCATTER_SEGMENT_IDXS and GATHER_SCATTER_SEGMENT_IDXS != 0
    	register unsigned long next_segment_idx = 0;
#endif

#if defined REVERSE_SEGMENT_IDXS and REVERSE_SEGMENT_IDXS != 0
      	for (unsigned long i = num_segments; i > 0; --i) {
    	    unsigned long ii = i - 1;
#elif defined GATHER_SCATTER_SEGMENT_IDXS and GATHER_SCATTER_SEGMENT_IDXS != 0
    	for (unsigned long i = 0; i < num_segments; ++i) {
            unsigned long ii = next_segment_idx;
#else
    	for (unsigned long i = 0; i < num_segments; ++i) {
    	    unsigned long ii = i;
#endif

#pragma vector nontemporal(mat)

#if defined SHUFFLE_SEGMENT_IDXS and SHUFFLE_SEGMENT_IDXS != 0
    	    //ii = _rotl(rnd, ii) % num_segments;
#endif

#if defined REVERSE_ELEMENT_IDXS and REVERSE_ELEMENT_IDXS != 0
    	    for (unsigned long j = num_elements; j > 0; --j) {
    		unsigned long jj = j - 1;
#else
    	    for (unsigned long j = 0; j < num_elements; ++j) {
    		unsigned long jj = j;
#endif
        
#if defined SHUFFLE_ELEMENT_IDXS and SHUFFLE_ELEMENT_IDXS != 0
    		//jj = _rotl(rnd, jj) % num_elements;
#endif

#if defined SHUFFLE_ALL and SHUFFLE_ALL != 0
    		    //unsigned long idx = get_idx(0, tid, ii, jj, num_threads, num_segments, num_elements);
#else
                unsigned long idx = tid * num_segments * num_elements + ii * num_elements + jj;
#endif
		

#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
    		    std::stringstream stream;
    		    stream << sizeof(TYPE) << " " << num_threads << " " << num_segments << " " << num_elements << " " << "x" << " " << omp_get_thread_num() << " " << tid << " " << i << " " << ii << " " << j << " " << jj << " " << idx << std::endl;
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
    		    	    TYPE r = mat[idx];
    			        v += r;
    		    	    mat[idx] += v;
    			        //readwrite_fun(mat, idx, v);
    		            break;
    		    }
#endif
    	    } // END NUM ELEMENTS FOR LOOP
            
#if defined GATHER_SCATTER_SEGMENT_IDXS and GATHER_SCATTER_SEGMENT_IDXS != 0
    	    next_segment_idx = (unsigned long)v % num_segments;
#endif
            
    	} // END NUM SEGMENTS FOR LOOP
    
    	end_time = std::chrono::high_resolution_clock::now();
	
    	val = v;

    	auto delta_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time);
    	double read_time = (double)delta_us.count() * 1e-9;

    	max_time = std::max(max_time, read_time);
    	time_sum += read_time;
    	time_sum_of_squares += std::pow(read_time, 2);
    } // END omp parallel

    mode_max_time = max_time;
    mode_avg_time = time_sum / num_threads;
    mode_var_time = (time_sum_of_squares - std::pow(time_sum, 2) / num_threads) / num_threads;
} // END multithreaded_benchmark

/*
void initialize(TYPE *mat, unsigned long rnd_input, TYPE val, unsigned long num_threads, unsigned long num_segments, unsigned long num_elements) {
    #pragma omp parallel shared(mat) num_threads(num_threads)
    {
        //for (unsigned int tid = 0; tid < num_threads; ++tid) {
    	unsigned long tid = omp_get_thread_num();
	unsigned int mod = (val < 2 ? 2 : (unsigned int)val);
	for (unsigned long j = 0; j < num_segments; ++j) {
	    for (unsigned long k = 0; k < num_elements; ++k) {
	        unsigned long idx = tid * num_segments * num_elements + j * num_elements + k ;
		TYPE init = ((idx+1)*(tid+1)*(j+1)*(k+1)) % mod;

	        if (rnd_input != 0 and idx % (unsigned int)val == 0) {
		    init = (val+1)*(tid+1)*(j+1)*(k+1);
	        } 
	    	mat[idx] = init;
		//std::string toprint = std::to_string(idx) + " " + std::to_string(init) + "\n";
		//std::cout << toprint; // TODO remove it
	    }
        }
    }
}
*/

// read the input file for initializing the matrix
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

// Write the output matrix in the output file
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
    if (argc != 4) {
        printf("wrong usage, please specify: num_threads max num elements per segment (spatial_locality/memory_quanta), rnd_input\n");
        exit(-1);
    }
    
    // Print the compile-time configuration (reversing/shuffling segments/elements ...)
    print_configuration();
    
    // Initialize a random value
    srand(time(NULL));
    unsigned int rnd = std::rand();

    // Get the program arguments
    unsigned long arg_num_threads = strtol(argv[1], NULL, 10);
    unsigned long max_num_elements = strtol(argv[2], NULL, 10);
    unsigned long rnd_input = strtol(argv[3], NULL, 10);

    unsigned long omp_num_threads = arg_num_threads;//omp_get_num_threads();

    // Print runtime info
    std::string running_mode = "Running";
#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
    running_mode = "Testing randomicity";
#endif
    std::cout << running_mode << " with NThreads=" << arg_num_threads << "(" << omp_get_num_threads << ")" << " and MNElements=" << max_num_elements << " and Rinput=" << rnd_input << std::endl;

    /// Declare time keepers
    double read_max_time = 0;
    double read_avg_time = 0;
    double read_var_time = 0;
    double write_max_time = 0;
    double write_avg_time = 0;
    double write_var_time = 0;
    double readwrite_max_time = 0;
    double readwrite_avg_time = 0;
    double readwrite_var_time = 0;

    // Launch the benchmark allocating a big enough matrix
    std::cout << "Allocating matrix." << std::endl;
    TYPE *mat = (TYPE *)malloc(omp_num_threads * max_num_elements * sizeof(TYPE));
    std::vector<unsigned int> values;
    {
        std::vector<unsigned int> words;
        std::cout << "Loading the file." << std::endl;
        read_file(words);
        std::cout << "Initializing matrix..." << std::endl;
        for (unsigned long i = 0; i < omp_num_threads *  max_num_elements; ++i) {
            unsigned long idx = i % words.size();
            mat[i] = words[idx];
        }
    }
    std::cout << "Initialization done." << std::endl;

    // Print report header
#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
    std::cout << "mquanta[B], num_threads, num_segments, num_elements, memsize[B], thread_num, tid, i, ii, j, jj, access_idx" << std::endl;
#else
    std::cout << "mquanta[B], num_threads, num_segments, num_elements, memsize[B], "
	      << "read_max_time[s], read_avg_time[s], read_var_time, "
	      << "write_max_time[s], write_avg_time[s], write_var_time, "
	      << "readwrite_max_time[s], readwrite_avg_time[s], readwrite_var_time, "
	      << "read_min_bw[B/s], read_avg_bw[B/s], "
          << "write_min_bw[B/s], write_avg_bw[B/s], "
          << "readwrite_min_bw[B/s], readwrite_avg_bw[B/s], " << std::endl;
#endif

    double memsize = (double)omp_num_threads * (double)max_num_elements * (double)sizeof(TYPE);

    // Increase the spatial locality and see what happens
    for (unsigned long num_elements = 1; num_elements <= max_num_elements; num_elements = num_elements << 1) {
        unsigned long num_segments = max_num_elements / num_elements;

        TYPE val = std::rand();

#if defined TEST_RANDOMICITY and TEST_RANDOMICITY != 0
        multithread_benchmark(TEST, mat, val, omp_num_threads, num_segments, num_elements, read_max_time, read_avg_time, read_var_time, rnd_input);
#else

#if defined ENABLE_READ and ENABLE_READ != 0
	    multithread_benchmark(READ, mat, val, omp_num_threads, num_segments, num_elements, read_max_time, read_avg_time, read_var_time, rnd_input);
#endif

#if defined ENABLE_WRITE and ENABLE_WRITE != 0
	    multithread_benchmark(WRITE, mat, val, omp_num_threads, num_segments, num_elements, write_max_time, write_avg_time, write_var_time, rnd_input);
#endif

#if defined ENABLE_READWRITE and ENABLE_READWRITE != 0
	    multithread_benchmark(READWRITE, mat, val, omp_num_threads, num_segments, num_elements, readwrite_max_time, readwrite_avg_time, readwrite_var_time, rnd_input);
#endif

		std::cout << std::fixed << sizeof(TYPE) << " " << omp_num_threads << " " << num_segments << " " << num_elements << " " << std::scientific <<  memsize << " " 
		   << read_max_time << " " << read_avg_time << " " << read_var_time << " " 
		   << write_max_time << " " << write_avg_time << " " << write_var_time << " " 
		   << readwrite_max_time << " " << readwrite_avg_time << " " << readwrite_var_time << " "
		   << get_bw(memsize, read_max_time) << " " << get_bw(memsize, read_avg_time) << " " 
		   << get_bw(memsize, write_max_time) << " " << get_bw(memsize, write_avg_time) << " " 
		   << get_bw(2*memsize, readwrite_max_time) << " " << get_bw(2*memsize, readwrite_avg_time) << std::endl;

        // Do random things for preventing compiler optimizations
	for(unsigned int i = 0; i < (unsigned int)val % 3; ++i) {
	    val += mat[((unsigned int)val % omp_num_threads) * num_elements + ((unsigned int)val % num_elements)];
	}
	if (val == 0) {
	    values.push_back(val);
	    std::cerr << "P";
	}
#endif
    }

    free(mat);
    std::cout << "Writing results..." << std::endl;
    write_file(values);
    std::cout << "Results written." << std::endl;
} 


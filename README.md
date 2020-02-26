# multithread_CPU_memory_benchmark

This repo contains a C++ benchmark for measuring memory bandwidth of multicore architectures on specific NUMA levels and access patterns.<br><br>
In particular, we are interested in measuring _read-only/write-only/read-write_ __memory bandiwdth__ while varying:<br>
* __memory quanta__ the byte size of the data type we access memory with
* __spatial locality__ the amount of subsequent bytes of sequential access
* __memory concurrency__ the amount of in-flight bytes in the memory subsystem<br>
For this reason, the allocated __memory space__ is subdivided in __blocks__, __segments__ and __elements__ in the following way:<br>

![alt text](https://github.com/marcosiracusa93/multithread_CPU_memory_benchmark/blob/master/CPU_benchmark_memory_layout.png)

Each thread accesses a certain __memory block__ with a mapping assigned at runtime according to the running mode. 
If the mapping is randomized, each memory block might be assigned from none to multiple threads. 
Future works would control the amount of conflicts between threads.<br>
Each memory block can be accessed in subsequent __memory segments__. 
These segments represent the spatial locality we want to control.<br>
In particular, memory segments can be accessed:
* __sequentialy__ the next accessed segment is stored in memory before/after the current one
  * __forward__ the next accessed segment is stored in memory after the current one
  * __backward__ the next accessed segment is stored in memory before the current one
* __randomly__ the next segment to be accessed is picked randomly during compile time (index dependent)
* __gather-scatter__ the next segment to be accessed is picked randomly during run time (data dependent)<br>
The listed access patterns allow to characterize memory bandwidth at the different NUMA levels of the multicore architecture.<br>
Memory segments are composed of __contiguous elements__ to be accessed:
* __sequentialy__ the next accessed element is stored in memory before/after the current one
  * __forward__ the next accessed element is stored in memory after the current one
  * __backward__ the next accessed element is stored in memory before the current one

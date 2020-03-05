#!/bin/bash
CC -c -fPIC libbutils.cpp -o libbutils.o
ar rcs libbutils.a libbutils.o
CC ./benchmark_mt.cpp -o ../bin/readwrite_benchmark_mt -fopenmp -O3 -v -L. -lbutils -DENABLE_READ=0 -DENABLE_WRITE=0 -DENABLE_READWRITE=1

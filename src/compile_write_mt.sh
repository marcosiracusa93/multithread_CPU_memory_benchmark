#!/bin/bash
CC -c -fPIC -O3 libbutils.cpp -o libbutils.o
ar rcs libbutils.a libbutils.o
CC ./benchmark_mt.cpp -o ../bin/write_benchmark_mt -fopenmp -O3 -v -L. -lbutils -DENABLE_READ=0 -DENABLE_WRITE=1 -DENABLE_READWRITE=0

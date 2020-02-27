#!/bin/bash
CC -c -fPIC libbutils.cpp -o libbutils.o
ar rcs libbutils.a libbutils.o
CC ./benchmark_mt.cpp -o ../bin/benchmark_mt -fopenmp -O3 -v -L. -lbutils

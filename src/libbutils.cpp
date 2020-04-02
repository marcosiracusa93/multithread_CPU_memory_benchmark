#include<stdio.h>
#include<iostream>

//template<typename T>
unsigned long increment_lcd_idx(unsigned long idx, unsigned long num_segments, unsigned long cache_lines_per_segment, unsigned long rnd,  double val) {
	unsigned long zero = (unsigned long)val % 2; // this should be always 0
	idx += zero;
	return (idx % 2 == 0 ? idx / 2 : num_segments - (idx + 1) / 2);
}

void print_cond(double val, bool cond) {
	if (cond) std::cout << val;
}

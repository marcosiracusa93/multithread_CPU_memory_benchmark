#include<stdio.h>
#include<iostream>

//template<typename T>
unsigned long increment_lcd_idx(unsigned long idx, unsigned long rnd,  double val) {
	return idx + 1;// + rnd * (unsigned long)val;
}

void print_cond(double val, bool cond) {
	if (cond) std::cout << val;
}

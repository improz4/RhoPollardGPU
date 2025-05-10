
#include <iostream>
#include <curand_kernel.h>
#include <mpir.h>

#define NUM_WORDS 32
#define BITS_PER_WORD 32
#define NUM_BLOCKS 2000

#define _E(x) { myCheckFunction((x), __FILE__, __LINE__, 1); }
#define _E_CT(x) { myCheckFunction((x), __FILE__, __LINE__, 0); }

//funzioni device
__device__ inline bool geq(const uint32_t* x, const uint32_t* y, size_t tid, size_t* pos);//CHECK
__device__ inline void ciosMonPro(const uint32_t* a, const uint32_t* b, const uint32_t* N, uint32_t* res, uint32_t n_prime, uint32_t* TMP, size_t tid, uint32_t*G , uint32_t* P, uint32_t* TMP2, size_t* pos);
__device__ inline void modAdd(const uint32_t* x, const uint32_t* y, const uint32_t* N, uint32_t* res, size_t tid, uint32_t* G, uint32_t* P, uint32_t* overflow, uint32_t* TMP, size_t* pos);//
__device__ inline void sub(const uint32_t* x,  const uint32_t* y, uint32_t* res, size_t tid, uint32_t* G, uint32_t* P, uint32_t* TMP);
__device__ inline void mcd(const uint32_t* x, const uint32_t* y, uint32_t* res, size_t tid, uint32_t* d, uint32_t* tmp_x, uint32_t* tmp_y, uint32_t* is_zero, size_t* pos, uint32_t* G, uint32_t* P, uint32_t* TMP);
__device__ inline bool nonZero(const uint32_t* x, size_t tid, uint32_t* result); 
__device__ inline void shiftLd(uint32_t* x, uint32_t* d, size_t tid); //CHECK
__device__ inline void shiftR1(uint32_t* x,size_t tid); //CHECK
__device__ inline void addAmount(uint32_t* x,uint32_t amount, size_t tid, uint32_t* G, uint32_t* P, size_t* pos); //CHECK
__device__ inline bool isN(const uint32_t* x, const uint32_t* n, size_t tid, uint32_t* result);
__device__ inline bool isOne(const uint32_t* x, size_t tid, uint32_t* result);
__global__ void rhoPollard(const uint32_t* N, uint32_t* DIV, uint32_t* n_prime, uint32_t* r2_mod_n, uint32_t leading_zeros, uint32_t* N_minus_4, uint32_t* N_minus_3);

//funzioni host
void rhoPollardGPU(const uint32_t* N, uint32_t* divider);
int is_divisor(const uint32_t* N, const uint32_t* div);
void print_num(const char* label, const uint32_t* num);
uint32_t rand32(void);
void generate_random_odd_number(uint32_t *output);
void generate_N(uint32_t* N);
uint32_t invMul(uint32_t n);
void compute_r2_mod_n(uint32_t* r2_mod_n, const uint32_t* N);
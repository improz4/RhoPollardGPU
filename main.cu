#include <stdio.h>
#include "PollardRhoGpu.h"

int is_divisor(const uint32_t* N, const uint32_t* div){
    // Convertiamo N e div in numeri GMP
    mpz_t mpz_N, mpz_divider, mpz_remainder;
    
    // Inizializziamo i numeri GMP
    mpz_init(mpz_N);
    mpz_init(mpz_divider);
    mpz_init(mpz_remainder);
    
    // Importiamo i numeri dagli array di uint32_t (assumendo little-endian)
    mpz_import(mpz_N, NUM_WORDS, -1, sizeof(uint32_t), 0, 0, N);
    mpz_import(mpz_divider, NUM_WORDS, -1, sizeof(uint32_t), 0, 0, div);
    
    // Calcoliamo il resto N % div
    mpz_mod(mpz_remainder, mpz_N, mpz_divider);
    
    // Verifichiamo se il resto è zero
    int is_divisor = (mpz_cmp_ui(mpz_remainder, 0) == 0);
    
    // Puliamo le risorse GMP
    mpz_clear(mpz_N);
    mpz_clear(mpz_divider);
    mpz_clear(mpz_remainder);
    
    // Restituisce true se div è un divisore (N % div == 0)
    return is_divisor;
}

int main(){

	uint32_t divider[NUM_WORDS], N[NUM_WORDS]={1};
	srand(time(NULL));

	generate_N(N);
	print_num("N: ",N);

	rhoPollardGPU(N, divider);
	print_num("Divider: ", divider);

	if(is_divisor(N,divider)) printf("Divisore non banale trovato");

    return 0;
}
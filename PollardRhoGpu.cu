
#include "PollardRhoGpu.h"

__device__ int global_flag;

inline void myCheckFunction(cudaError_t cd, const char *fl, int ln, int ab){
	if (cd != cudaSuccess){
		  fprintf(stderr,"Error! - File: %s Line: %d Details: %s\n", fl, ln, cudaGetErrorString(cd));
		  if (ab) exit(cd);
	}
}
//funzioni device
__device__ inline bool geq(const uint32_t* x, const uint32_t* y, size_t tid, size_t* pos) {

	if(tid == 0) *pos=0;
	__syncthreads(); //necessaria affinché tutti vedano pos inizializzata a 0

	if(x[tid] != y[tid]) atomicMax(pos,tid); // Salva l'indice della word più significativa in cui x e y differiscono
	__syncthreads();  //attesa completamento confronti

	return x[*pos] >= y[*pos];

}

__device__ inline void ciosMonPro(const uint32_t* a, const uint32_t* b, const uint32_t* N, uint32_t* res, uint32_t n_prime, uint32_t* TMP, size_t tid, uint32_t*G , uint32_t* P, uint32_t* TMP2, size_t* pos)
{
	uint32_t carry,m;
	uint64_t cs;
	size_t i,j;


	//TMP è di dimensione pari a NUM_WORDS+2

	if(tid == 0)
	{
		TMP[0] = 0;
		TMP[NUM_WORDS] = 0;
		TMP[NUM_WORDS+1] = 0;
	}
	else
	{
		TMP[tid] = 0;
	}
	__syncthreads();

	if(tid == 0)
	{
		for (i = 0; i < NUM_WORDS; i++) {
			carry = 0;
			for (j = 0; j < NUM_WORDS; ++j) {
				cs = TMP[j] + (uint64_t)a[j] * b[i] + carry;
				TMP[j] = (uint32_t)cs;
				carry = (uint32_t)(cs >> 32);
			}
			cs = (uint64_t)TMP[NUM_WORDS] + carry;
			TMP[NUM_WORDS] = (uint32_t) cs;
			TMP[NUM_WORDS + 1] = (uint32_t)(cs >> 32);
	
			m = (uint32_t)((uint64_t)TMP[0] * n_prime);
			cs = TMP[0] + (uint64_t)m * N[0]; 
			carry = (uint32_t)(cs>>32);
	
			for (j = 1; j < NUM_WORDS; ++j) {
				cs = TMP[j] + (uint64_t)m * N[j] + carry;
				TMP[j-1]= (uint32_t) cs;
				carry = (uint32_t)(cs >> 32);
			}
			cs = (uint64_t)TMP[NUM_WORDS] + carry;
			TMP[NUM_WORDS-1] = (uint32_t)(cs);
			TMP[NUM_WORDS] = TMP[NUM_WORDS+1] + (uint32_t)(cs>>32);
		}
	}
	__syncthreads();

	if(TMP[NUM_WORDS] !=0)
	{
		sub(TMP, N, TMP, tid, G, P, TMP2);
	}
	else if (geq(TMP, N, tid, pos))
	{
		sub(TMP, N, TMP, tid, G, P, TMP2);
	}
	
	res[tid] = TMP[tid];
	__syncthreads();
}

__device__ inline void modAdd(const uint32_t* x, const uint32_t* y, const uint32_t* N, uint32_t* res, size_t tid, uint32_t* G, uint32_t* P, uint32_t* overflow, uint32_t* TMP, size_t* pos)
{
	uint32_t g_tmp,p_tmp;
	uint64_t cs = (uint64_t)x[tid] + (uint64_t)y[tid];

	G[tid] = (uint32_t)(cs>>32);
	P[tid] = ((uint32_t)(cs) == 0xFFFFFFFF);

	__syncthreads();

	for(size_t offset=1;offset < NUM_WORDS; offset*=2)
	{
		if(tid >= offset)
		{
			g_tmp = G[tid-offset]; 
			p_tmp = P[tid-offset];
		}

		__syncthreads();

		if(tid >= offset)
		{
			G[tid] += P[tid] * g_tmp; 
			P[tid] *= p_tmp;
		}

		__syncthreads();
	}

	if(tid==0)
	{
		res[tid] = (uint32_t)(cs);
		*overflow = G[NUM_WORDS-1];
	}
	else res[tid] = (uint32_t)(cs) + G[tid-1];

	__syncthreads();

	if(*overflow != 0)
	{
		sub(res, N, res, tid, G, P, TMP);
	}
	else if(geq(res, N,tid,pos))
	{
		sub(res, N, res, tid, G, P, TMP);
	}
}

__device__ inline void sub(const uint32_t* x,  const uint32_t* y, uint32_t* res, size_t tid, uint32_t* G, uint32_t* P, uint32_t* TMP)
{
	uint32_t g_tmp,p_tmp;
	TMP[tid] = ~y[tid];
	__syncthreads();

	uint64_t cs = (uint64_t)x[tid] + (uint64_t)TMP[tid];

	G[tid] = uint32_t(cs >> 32);
	P[tid] = ((uint32_t)(cs) == 0xFFFFFFFF);
	__syncthreads();

	for(size_t offset=1;offset < NUM_WORDS; offset*=2)
	{
		if(tid >= offset)
		{
			g_tmp = G[tid-offset]; 
			p_tmp = P[tid-offset];
		}

		__syncthreads();

		if(tid >= offset)
		{
			G[tid] += P[tid] * g_tmp; 
			P[tid] *= p_tmp;
		}

		__syncthreads();
	}

	if(tid==0) res[tid] = (uint32_t)(cs) + 1;
	else res[tid] = (uint32_t)(cs) + ( G[tid-1]+ P[tid-1]) ;

	__syncthreads();
}

__device__ inline void mcd(const uint32_t* x, const uint32_t* y, uint32_t* res, size_t tid, uint32_t* d, uint32_t* tmp_x, uint32_t* tmp_y, uint32_t* is_zero, size_t* pos, uint32_t* G, uint32_t* P, uint32_t* TMP)
{

	
	res[tid] = 0;
	tmp_x[tid] = 0;
	tmp_y[tid] = 0;
	__syncthreads();
	
	if(!nonZero(y, tid, is_zero)) 
	{
        res[tid] = x[tid];
        __syncthreads();
    }
    else {

        if(tid == 0) *d = 0;
        tmp_x[tid] = x[tid];
        tmp_y[tid] = y[tid];
        __syncthreads();

        while(((tmp_x[0] & 1)==0) && ((tmp_y[0] & 1)==0)) 
		{
            shiftR1(tmp_x, tid);
            shiftR1(tmp_y, tid);
    
            if(tid == 0) (*d)++;
            __syncthreads();
        }
    
		while(nonZero(tmp_x, tid, is_zero)) 
		{            
            while((tmp_x[0] & 1) == 0) {
                shiftR1(tmp_x, tid);
            }
    
            while((tmp_y[0] & 1) == 0) {
                shiftR1(tmp_y, tid);
            }
            
            if(geq(tmp_x, tmp_y, tid, pos)) 
			{
                sub(tmp_x, tmp_y, tmp_x, tid, G, P, TMP);
                shiftR1(tmp_x, tid);
            }
            else{	
                sub(tmp_y, tmp_x, tmp_y, tid, G, P, TMP);	
                shiftR1(tmp_y, tid);
            }
			__syncthreads();
        }
		res[tid] = tmp_y[tid];
        __syncthreads();

		if(*d > 0) shiftLd(res, d, tid);
    }
}

__device__ inline bool nonZero(const uint32_t* x, size_t tid, uint32_t* result)
{
	if(tid==0) *result=0;
	__syncthreads();
	if(x[tid]!=0) atomicOr(result,1);
	__syncthreads();
	return *result;
}

__device__ inline void shiftLd(uint32_t* x, uint32_t* d, size_t tid)
{
	uint32_t word_shift,x1;

	word_shift = *d / BITS_PER_WORD;

	if(tid==0) *d -= (word_shift*BITS_PER_WORD);
	__syncthreads();

	if(word_shift > 0)
	{
		x1 = (tid >= word_shift) ? x[tid-word_shift] : 0;
		__syncthreads();
		x[tid] = x1;
		__syncthreads();
	}

	if(*d>0)
	{
		x1 = (tid > 0) ? x[tid-1] : 0;
		__syncthreads();
		x[tid] = (x[tid] << *d) | (x1 >> (BITS_PER_WORD - *d));
		__syncthreads();
	}
}

__device__ inline void shiftR1(uint32_t* x,size_t tid)
{
	uint32_t x1;
	x1 = (tid < (NUM_WORDS-1)) ? x[tid+1] : 0;
	__syncthreads();
	x[tid] = (x[tid] >> 1) | (x1 << 31);
	__syncthreads();
}

__device__ inline void addAmount(uint32_t* x,uint32_t amount, size_t tid, uint32_t* G, uint32_t* P, size_t* pos)
{
	if(tid == 0) *pos=0;
	__syncthreads();

	G[tid] = (tid==0) ? (0xFFFFFFFF-amount) : 0xFFFFFFFF;

	if(x[tid] != G[tid]) atomicMax(pos,tid);

	__syncthreads();

	//se aggiungendo il numero amount non vado in overflow
	if(x[*pos] <= G[*pos])
	{
		uint32_t g_tmp,p_tmp;
		g_tmp = (tid==0) ? amount : 0;
		uint64_t cs = (uint64_t)x[tid] + g_tmp;
		P[tid] = ((uint32_t)(cs) == 0xFFFFFFFF);
		__syncthreads();

		G[tid] = (uint32_t)(cs>>32);
		__syncthreads();

		for(size_t offset=1;offset < NUM_WORDS; offset*=2)
		{
			if(tid >= offset)
			{
				g_tmp = G[tid-offset]; 
				p_tmp = P[tid-offset];
			}

			__syncthreads();

			if(tid >= offset)
			{
				G[tid] += P[tid] * g_tmp; 
				P[tid] *= p_tmp;
			}

			__syncthreads();
		}

		if(tid==0)
			x[tid] = (uint32_t)(cs);
		else 
			x[tid] = (uint32_t)(cs) + G[tid-1];
	}
}

__device__ inline bool isOne(const uint32_t* x, size_t tid, uint32_t* result)
{
    if(tid == 0) *result = 1;
    __syncthreads();

    if(tid == 0)
	{
		if(x[tid]!=1) atomicAnd(result,0);
	}
	else
		if(x[tid]!=0) atomicAnd(result,0);    
    __syncthreads();

    return (*result != 0);
}

__device__ inline bool isN(const uint32_t* x, const uint32_t* n, size_t tid, uint32_t* result)
{
    if(tid == 0) *result = 1;
    __syncthreads();
    
    if(x[tid] != n[tid]) {
        atomicAnd(result, 0); 
    }
    __syncthreads(); 

    return (*result != 0);
}

__global__ void rhoPollard(const uint32_t* N, uint32_t* DIV, uint32_t* n_prime, uint32_t* r2_mod_n, uint32_t leading_zeros, uint32_t* N_minus_4, uint32_t* N_minus_3) 
{
	__shared__ uint32_t x[NUM_WORDS], y[NUM_WORDS], n[NUM_WORDS], c[NUM_WORDS], div[NUM_WORDS], diff[NUM_WORDS];
	__shared__ uint32_t G[NUM_WORDS], P[NUM_WORDS], TMP[NUM_WORDS], ciosTMP[NUM_WORDS+2], one[NUM_WORDS], tmp_x[NUM_WORDS], tmp_y[NUM_WORDS];
	__shared__ size_t pos;
	__shared__ uint32_t tmp1, tmp2;
	__shared__ uint32_t old;

	size_t tid = threadIdx.x;
	curandState state;

	//copia del numero da fattorizzare in shared memory
	n[tid] = N[tid];
	//inizializzazione 1 e divisore
	one[tid] = (tid==0) ? 1 : 0;
	div[tid] = (tid==0) ? 1 : 0;
	//generazione casuale dei numeri x, c e y
	//inizializzazione stato generatore
	curand_init(blockIdx.x, tid, 0, &state);
	__syncthreads();  //necesaria affinché tutti i thread aggiornino il loro stato

	do
	{
		x[tid] = (tid < NUM_WORDS - leading_zeros) ? curand(&state) : 0;
		__syncthreads(); //necessaria affinché tutti i thread generino la propria porzione di word
	}while(geq(x , N_minus_4 , tid, &pos));
	addAmount(x, 2, tid, G, P, &pos);

	do
	{
		c[tid] = (tid < NUM_WORDS - leading_zeros) ? curand(&state) : 0;
		__syncthreads();
	}while(geq(c, N_minus_3, tid, &pos));
	addAmount(c, 1, tid, G, P, &pos);

	//rappresentazione in forma di montgomery
	ciosMonPro(x, r2_mod_n, n, x, *n_prime, ciosTMP, tid, G, P, TMP, &pos);
	ciosMonPro(c, r2_mod_n, n, c, *n_prime, ciosTMP, tid, G, P, TMP, &pos);
	//inizializzazione y come copia di x
	y[tid] = x[tid]; 
	__syncthreads();

	//global_found_flag è un flag in memoria global volatile, il primo blocco che troverà un fattore ci scriverà per indicare che l'ha troato
	//quindi entro nel while solo se il divisore è 1 oppure se nessun altro blocco della griglia ha trovato un fattore non banale
	while(isOne(div, tid, &tmp1))
	{
		if(tid==0)
		{
			old = atomicAdd(&global_flag , 0);
		}
		__syncthreads();
		if(old != 0) break;

		//passo della tartaruga x=x^2 mod n
		ciosMonPro(x, x, n, x, *n_prime, ciosTMP, tid, G, P, TMP, &pos);
		// x = (x^2 mod n + c) mod n
		modAdd(x, c, n, x, tid, G, P, &tmp1, TMP, &pos);

		//passo della lepre
		ciosMonPro(y, y, n, y, *n_prime, ciosTMP, tid, G, P, TMP, &pos);
		modAdd(y, c, n, y, tid, G, P, &tmp1, TMP, &pos);
		ciosMonPro(y, y, n, y, *n_prime, ciosTMP, tid, G, P, TMP, &pos);
		modAdd(y, c, n, y, tid, G, P, &tmp1, TMP, &pos);

		//calcolo della differenza
		if(geq(x, y, tid, &pos))
			sub(x, y, diff, tid, G, P, TMP);
		else
			sub(y, x, diff, tid, G, P, TMP);
		
		//riconversione in forma normale della differenza
		ciosMonPro(diff, one, n, diff, *n_prime, ciosTMP, tid, G, P, TMP, &pos);

		//calcolo del mcd tra la differenza e il numero da fattorizzare
		mcd(diff, n, div, tid, &tmp1, tmp_x, tmp_y, &tmp2, &pos, G, P, TMP);

		if(isN(div, n, tid, &tmp1))
		{
			//se il divisore trovato è uguale a N e quindi banale, si riavvia l'algoritmo rigenerando tutti gli operandi necessari
			do
			{
				x[tid] = (tid < NUM_WORDS - leading_zeros) ? curand(&state) : 0;
				__syncthreads();
			}while(geq(x , N_minus_4 , tid, &pos));
			addAmount(x, 2, tid, G, P, &pos);
		
			do{
				c[tid] = (tid < NUM_WORDS - leading_zeros) ? curand(&state) : 0;
				__syncthreads();
			}while(geq(c, N_minus_3, tid, &pos));
			addAmount(c, 1, tid, G, P, &pos);
		
			ciosMonPro(x, r2_mod_n, n, x, *n_prime, ciosTMP, tid, G, P, TMP, &pos);
			ciosMonPro(c, r2_mod_n, n, c, *n_prime, ciosTMP, tid, G, P, TMP, &pos); 
			y[tid] = x[tid]; 
			div[tid] = (tid==0) ? 1 : 0;
		}
		else if(!(isOne(div, tid, &tmp1)))
		{
			//se il fattore non è Né 1 né n, ho trovato un fattore non banale. 
			//Scrivo in atomicAdd. Se ero il primo a scrivere, allora entro in una sezione critica dove scrivo il risultato trovato
			//altrimenti esco
			if (tid==0) 
			{	
				old = atomicCAS(&global_flag, 0, 1);  // Scrittura atomica
				
			}
			__syncthreads();
			if(old == 0) 
			{
				DIV[tid] = div[tid];
			}
        }
	}
}

//funzioni host
uint32_t rand32(void)
{    //RAND MAX restituisce al massimo 2^15 - 1, di conseguenza per fare un numero da 32 bit sono necessari 3 chiamate (15+15+2)
	return ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 2) ^ (rand() & 0x3);
}

void generate_random_odd_number(uint32_t *output)
{
	//inizializza numero dispari di grandezza NUM_WORDS/2
	for(int i=0; i<(NUM_WORDS/2); i++)
	{
		output[i] = rand32();
	}

	//se pari
	if((output[0] & 1) == 0)
		output[0] |= 1;  // Imposta direttamente il bit meno significativo a 1
}

void generate_N(uint32_t* N)
{    
	//Per generare un N che si possa applicare all'algortimo di fattorizzazione non può essere semplicemente generato randomicamente
	//potrebbe essere primo, allora si generano due numeri dispari definiti su NUM_WORDS/2 che moltiplicano generano un N non primo
	// dispari e su NUM_WORDS parole
	
	uint32_t p[NUM_WORDS/2], q[NUM_WORDS/2];

	generate_random_odd_number(p);
	generate_random_odd_number(q);

	mpz_t mpz_p, mpz_q, mpz_N;
    mpz_inits(mpz_p, mpz_q, mpz_N, NULL);
    
    mpz_import(mpz_p, NUM_WORDS/2, -1, sizeof(uint32_t), 0, 0, p);
    mpz_import(mpz_q, NUM_WORDS/2, -1, sizeof(uint32_t), 0, 0, q);
	//moltiplico i due numeri per ottenere N
    mpz_mul(mpz_N, mpz_p, mpz_q);

	mpz_export(N, NULL, -1, sizeof(uint32_t), 0, 0, mpz_N);

	mpz_clears(mpz_p, mpz_q, mpz_N, NULL);   
}

uint32_t invMul(uint32_t n) {

    uint32_t x = n;  

    // 5 iterazioni per coprire 32 bit (2^5 = 32)
    x *= 2 - n * x;  // k=2,
    x *= 2 - n * x;  // k=4
    x *= 2 - n * x;  // k=8
    x *= 2 - n * x;  // k=16
    x *= 2 - n * x;  // k=32

    return -x; 
}

void compute_r2_mod_n(uint32_t* r2_mod_n, const uint32_t* N) {
    mpz_t mpz_N, mpz_R_squared;

    // Inizializza le variabili GMP
    mpz_inits(mpz_N, mpz_R_squared, NULL);

    // Converti N (little-endian) in mpz_t
    mpz_import(mpz_N, NUM_WORDS, -1, sizeof(uint32_t), 0, 0, N);

    // Calcola R² = 2^{64*NUM_WORDS} (perché R = 2^{32*NUM_WORDS})
    mpz_set_ui(mpz_R_squared, 1);                     // Inizializza a 1
    mpz_mul_2exp(mpz_R_squared, mpz_R_squared, 64 * NUM_WORDS);  // R² = 2^{64*NUM_WORDS}

    // Calcola R² mod N
    mpz_mod(mpz_R_squared, mpz_R_squared, mpz_N);     // R² mod N

    // Esporta il risultato in r2_mod_n (azzera prima per sicurezza)
    memset(r2_mod_n, 0, NUM_WORDS * sizeof(uint32_t));
    size_t count;
    mpz_export(r2_mod_n, &count, -1, sizeof(uint32_t), 0, 0, mpz_R_squared);

    // Pulisci le variabili GMP
    mpz_clears(mpz_N, mpz_R_squared, NULL);
}

void subtract_from_N(uint32_t* N_subtracted, const uint32_t* N, uint8_t q) {
    // Inizializza gli interi GMP dai puntatori uint32_t
    mpz_t n, result;
    mpz_init(n);
    mpz_init(result);

    mpz_import(n, NUM_WORDS, -1, sizeof(uint32_t), 0, 0, N);

    mpz_sub_ui(result, n, q); // Sostituisci 4 con 3 se vuoi sottrarre 3

    // Esporta il risultato in r2_mod_n (in formato little-endian)
    size_t written;
    mpz_export(N_subtracted, &written, -1, sizeof(uint32_t), 0, 0, result);
	
    // Pulisci le variabili GMP
    mpz_clear(n);
    mpz_clear(result);
}

void print_num(const char* label, const uint32_t* num) {
    printf("%-8s", label);
    for(int i = NUM_WORDS - 1; i >= 0; i--) {
        printf("%08x ", num[i]);
    }
    printf("\n");
}

void rhoPollardGPU(const uint32_t* N, uint32_t* divider)
{
	uint32_t leading_zeros=0;
	uint32_t *n_prime, *r2_mod_n, *N_minus_4, *N_minus_3;
	//unsigned int* global_flag;
	//r quadro modulo n
	r2_mod_n = (uint32_t*) malloc( NUM_WORDS*sizeof(uint32_t));
	for(int i=0; i<NUM_WORDS;i++) r2_mod_n[i]=0; //necessario per GMP
	//inverso molitplicativo
	n_prime = (uint32_t*) malloc(sizeof(uint32_t));
	//n-4 necessario per la generazione di x
	N_minus_4 = (uint32_t*) malloc(sizeof(uint32_t)*NUM_WORDS);
	//n-2 necessario per la generazione di c
	N_minus_3 = (uint32_t*) malloc(sizeof(uint32_t)*NUM_WORDS);

	*n_prime = invMul(N[0]);
	compute_r2_mod_n(r2_mod_n, N);
	//calcolo N-4 ed N-3
	subtract_from_N(N_minus_4, N, 4);
	subtract_from_N(N_minus_3, N, 3);
	//calcolo word più significative nulle
	for(int i=NUM_WORDS-1; i>=0; i--)
	{
		if(N[i]==0) leading_zeros++;
		else break;
	}
	size_t num_blocks_x = NUM_BLOCKS;
	size_t THREADSPERBLOCK_X = NUM_WORDS;

	cudaEvent_t start, stop;
	float streamElapsedTime;
	_E( cudaEventCreate(&start) );
	_E( cudaEventCreate(&stop) );
	_E( cudaEventRecord( start, 0 ) );
	
	uint32_t *N_d,*DIV_d, *n_prime_d, *r2_mod_n_d, *N_minus_4_d, *N_minus_3_d;
	_E( cudaMalloc((void **)&N_d,sizeof(uint32_t)*(NUM_WORDS)) );
	_E( cudaMalloc((void **)&DIV_d,sizeof(uint32_t)*(NUM_WORDS)) );
	_E( cudaMalloc((void **)&n_prime_d,sizeof(uint32_t)));
	_E( cudaMalloc((void **)&r2_mod_n_d,sizeof(uint32_t)*(NUM_WORDS)));
	_E( cudaMalloc((void **)&N_minus_4_d,sizeof(uint32_t)*(NUM_WORDS)) );
	_E( cudaMalloc((void **)&N_minus_3_d,sizeof(uint32_t)*(NUM_WORDS)) );

	_E( cudaMemcpy( N_d, N, sizeof(uint32_t)*(NUM_WORDS), cudaMemcpyHostToDevice ) );
	_E( cudaMemcpy( DIV_d, divider, sizeof(uint32_t)*(NUM_WORDS), cudaMemcpyHostToDevice ) );
	_E( cudaMemcpy( n_prime_d, n_prime, sizeof(uint32_t), cudaMemcpyHostToDevice ) );
	_E( cudaMemcpy( r2_mod_n_d, r2_mod_n, sizeof(uint32_t)*(NUM_WORDS), cudaMemcpyHostToDevice ) );
	_E( cudaMemcpy( N_minus_4_d, N_minus_4, sizeof(uint32_t)*(NUM_WORDS), cudaMemcpyHostToDevice ) );
	_E( cudaMemcpy( N_minus_3_d, N_minus_3, sizeof(uint32_t)*(NUM_WORDS), cudaMemcpyHostToDevice ) );

	//inizializazione flag
	int init_val = 0;
	_E(cudaMemcpyToSymbol(global_flag, &init_val, sizeof(int)));

	dim3 gridDims(num_blocks_x);
	dim3 blockDims(THREADSPERBLOCK_X);

	int max_threads_per_block;
cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
printf("Max threads per block: %d\n", max_threads_per_block);

	rhoPollard<<<gridDims,blockDims>>>(N_d, DIV_d, n_prime_d, r2_mod_n_d, leading_zeros, N_minus_4_d, N_minus_3_d/*,global_flag_d*/);

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, rhoPollard);
	printf("Shared memory per block: %u B (static), %u B (max)\n",
		(unsigned int) attr.sharedSizeBytes,
		(unsigned int) attr.maxDynamicSharedSizeBytes);
 
	
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
	}   // check for invalid arguments
	_E( cudaMemcpy( divider, DIV_d, sizeof(uint32_t)*(NUM_WORDS), cudaMemcpyDeviceToHost ) );
	_E( cudaEventRecord( stop, 0 ) );
	_E( cudaEventSynchronize( stop ) );
	_E( cudaEventElapsedTime( &streamElapsedTime, start, stop ) );
	_E( cudaEventDestroy( start ) );
	_E( cudaEventDestroy( stop ) );
	
	_E( cudaFree(N_d) );
	_E( cudaFree(DIV_d) );
	_E( cudaFree(n_prime_d) );
	_E( cudaFree(r2_mod_n_d) );
	_E( cudaFree(N_minus_4_d) );
	_E( cudaFree(N_minus_3_d) );
	//_E( cudaFree(global_flag_d) );

	//printf("CUDA stream elapsed time:  %f\n", streamElapsedTime);

	free(n_prime);
	free(r2_mod_n);
	free(N_minus_4);
	free(N_minus_3);
	//free(global_flag);
}

//funzione per la ground truth e determinare se il divisore è effettivamente corretto
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

    return 0;
}

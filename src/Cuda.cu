/*
 * Copyright (C)  2011  Luca Vaccaro
 *
 * TrueCrack is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */

#include "Tcdefs.h"
#include "Volumes.cuh"
#include <stdio.h>
#include <memory.h>
#include "Crypto.cuh"
#include "Core.cuh"
#include "Pkcs5.cuh"
#include "Xts.cuh"

/* Constat gpu memory data */
__device__ __constant__ unsigned char cHeaderEncrypted[TC_VOLUME_HEADER_EFFECTIVE_SIZE];
__device__ __constant__ unsigned char cSalt[SALT_LENGTH];

/* Header key size */
#define MAXPKCS5OUTSIZE 64

/* The max number of block grid; number of max parallel gpu blocks. */
int blockGridSizeMax;

/* The number of the current block grid; number of current parallel gpu blocks. */
int blockGridSizeCurrent;


/* Pointer of structures to pass to Cuda Kernel. */
unsigned char *dev_salt, *dev_blockPwd, *dev_header, *dev_headerKey;
int *dev_blockPwd_init, *dev_blockPwd_length;
short int *dev_result;
/* With Stream
#define NSTREAM 6
unsigned char *dev_salt, *dev_blockPwd[NSTREAM], *dev_header, *dev_headerKey[NSTREAM];
int *dev_blockPwd_init[NSTREAM], *dev_blockPwd_length[NSTREAM];
short int *dev_result;
*/

int getMultiprocessorCount (void){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	return prop.multiProcessorCount;
}

//#define RESIDENTTHREADS		1536
//#define NUMBLOCKS		12
#define NUMTHREADSXBLOCK	256

// Handle GPU error
static void HandleError( cudaError_t err, const char *file,  int line ) {
        if (err != cudaSuccess) {
                printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
                exit( EXIT_FAILURE );
        }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


// GPU kernel: build word from an alphabet
__global__ void cuKernel_generate(unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, int offset, uint32_t maxsize, int charsetlength, unsigned char *charset, int wordlength){
    int number=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
    if (number>=maxsize) {blockPwd_init[number]=1;return;}
	
    blockPwd_init[number]=number*wordlength;//(number==0)?0:blockPwd_init[number-1]+wordlength;
    blockPwd_length[number]=wordlength;
    
    unsigned char *word; word= &blockPwd[number*wordlength];
    unsigned short i=0;
    for (i=0;i<wordlength;i++)
        word[i]=0;
    i=0;
    number+=offset;
    while(number>0){
        word[i]=number%charsetlength;
        number=(number-word[i])/charsetlength;
        i++;
    }
    
    for (i=0;i<wordlength;i++)
    	word[i]=charset[word[i]];
}

// GPU kernel: ripemd160 hash
__global__ void cuKernel_ripemd160 (unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, unsigned char *headerKey, int max) {
	int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
	if (numData>=max) return;
	cuda_derive_key_ripemd160 (  blockPwd+blockPwd_init[numData], blockPwd_length[numData], cSalt, PKCS5_SALT_SIZE, 2000, headerKey+numData*MAXPKCS5OUTSIZE, 64);
}
// GPU kernel: sha512 hash
__global__ void cuKernel_sha512 ( unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, unsigned char *headerKey, int max) {
	int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
	if (numData>=max) return;
	cuda_derive_key_sha512 (  blockPwd+blockPwd_init[numData], blockPwd_length[numData], cSalt, PKCS5_SALT_SIZE, 1000, headerKey+numData*MAXPKCS5OUTSIZE, 64);
}
// GPU kernel: whirlpool hash
__global__ void cuKernel_whirlpool ( unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, unsigned char *headerKey, int max) {
       int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
        if (numData>=max) return;
        cuda_derive_key_whirlpool (  blockPwd+blockPwd_init[numData], blockPwd_length[numData], cSalt, PKCS5_SALT_SIZE, 1000, headerKey+numData*MAXPKCS5OUTSIZE, 64);
}
// GPU kernel: aes xts decryption
__global__ void cuKernel_aes ( unsigned char *headerKey, short int *result, int max) {
	int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
	if (numData>=max) return;
	__align__(8) unsigned char headerDecrypted[512];
	result[numData]=cuXts (AES,cHeaderEncrypted, headerKey+numData*MAXPKCS5OUTSIZE,headerDecrypted);
}
// GPU kernel: serpent xts decryption
__global__ void cuKernel_serpent ( unsigned char *headerKey, short int *result, int max) {
	int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
	if (numData>=max) return;
	__align__(8) unsigned char headerDecrypted[512];
	result[numData]=cuXts (SERPENT,cHeaderEncrypted, headerKey+numData*MAXPKCS5OUTSIZE,headerDecrypted);
}
// GPU kernel: twofish xts decryption
__global__ void cuKernel_twofish (unsigned char *headerKey, short int *result, int max) {
	int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;
	if (numData>=max) return;
	__align__(8) unsigned char headerDecrypted[512];
	result[numData]=cuXts (TWOFISH,cHeaderEncrypted, headerKey+numData*MAXPKCS5OUTSIZE,headerDecrypted);
}

// Perform the bruteforce on dictionary mode
float cuda_Core_dictionary ( int encryptionAlgorithm, int bsize, unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, short int *result, int keyDerivationFunction) {
	// Initialization
	int lengthpwd=0;
	for (int j=0;j<bsize;j++) {
		lengthpwd+=blockPwd_length[j];
		result[j]=0;
	}
	// Copy memory datas from host to gpu
	HANDLE_ERROR(cudaMemcpy(dev_blockPwd, 		blockPwd, 		lengthpwd * sizeof(unsigned char) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_blockPwd_init, 	blockPwd_init, 	bsize * sizeof(int) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_blockPwd_length,blockPwd_length,bsize * sizeof(int) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_result, 		result,			bsize * sizeof(short int) , cudaMemcpyHostToDevice));

	// Calculate the block grid and threads for block
	int numBlocks=bsize/NUMTHREADSXBLOCK+1;
	int numThreads=NUMTHREADSXBLOCK;
	if (bsize<NUMTHREADSXBLOCK)
		numThreads=bsize;

	// Start timer
	cudaEvent_t tstart,tstop;
	float time;
	cudaEventCreate(&tstart);
	cudaEventCreate(&tstop);
	cudaEventRecord(tstart, 0);

	// GPU Kernel: Key derivation function
	switch(keyDerivationFunction){
		case RIPEMD160:
			cuKernel_ripemd160 <<<numBlocks,numThreads>>>(dev_blockPwd, dev_blockPwd_init, dev_blockPwd_length, dev_headerKey, bsize);
			break;
		case SHA512:
			cuKernel_sha512 <<<numBlocks,numThreads>>>(dev_blockPwd, dev_blockPwd_init, dev_blockPwd_length, dev_headerKey,bsize);
			break;
		case WHIRLPOOL:
			cuKernel_whirlpool <<<numBlocks,numThreads>>>(dev_blockPwd, dev_blockPwd_init, dev_blockPwd_length, dev_headerKey,bsize);
			break;
	}
	
	// GPU Kernel: Encryption algorithms
	switch(encryptionAlgorithm){
		case AES:
			cuKernel_aes<<<numBlocks,numThreads>>>(dev_headerKey, dev_result, bsize);
			break;
		case SERPENT:
			cuKernel_serpent<<<numBlocks,numThreads>>>(dev_headerKey, dev_result, bsize);
			break;
		case TWOFISH:
			cuKernel_twofish<<<numBlocks,numThreads>>>(dev_headerKey, dev_result, bsize);
			break;
	}

	// Stop timer
	cudaEventRecord(tstop, 0);
	cudaEventSynchronize(tstop);
	cudaEventElapsedTime(&time, tstart, tstop);

	// Copy memory result from gpu to host
	HANDLE_ERROR(cudaMemcpy(result, dev_result,bsize* sizeof(short int) , cudaMemcpyDeviceToHost));
	return time;
}


// Perform the bruteforce on charset mode
float cuda_Core_charset ( int encryptionAlgorithm, uint64_t bsize, uint64_t start, unsigned short int charset_length, unsigned char *charset, unsigned short int password_length, short int *result, int keyDerivationFunction)
{
	// Initialization
	int numBlocks=(int)(bsize/NUMTHREADSXBLOCK)+1;
	int numThreads=NUMTHREADSXBLOCK;
	if (bsize<NUMTHREADSXBLOCK)
		numThreads=(int)bsize;
		
	// Copy memory datas from host to gpu
	unsigned char *dev_charset = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_charset, charset_length*sizeof(unsigned char)));
	HANDLE_ERROR(cudaMemcpy(dev_charset, charset, charset_length*sizeof(unsigned char), cudaMemcpyHostToDevice));
	/*
	char host_blockPwd[bsize*PASSWORD_MAXSIZE];
	int host_blockPwd_init[bsize];
	int host_blockPwd_length[bsize];
	*/
	// Start timer
    cudaEvent_t tstart,tstop;
    float time;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart, 0); 	

	// GPU Kernel: generate passwords
	cuKernel_generate <<<numBlocks,numThreads>>>(dev_blockPwd,dev_blockPwd_init,dev_blockPwd_length,(int)start,bsize,charset_length,dev_charset,password_length);
	
	// GPU Kernel: Key derivation function
	switch(keyDerivationFunction){
		case RIPEMD160:
			cuKernel_ripemd160 <<<numBlocks,numThreads>>>(dev_blockPwd, dev_blockPwd_init, dev_blockPwd_length, dev_headerKey, bsize);
			break;
		case SHA512:
			cuKernel_sha512 <<<numBlocks,numThreads>>>(dev_blockPwd, dev_blockPwd_init, dev_blockPwd_length, dev_headerKey,bsize);
			break;
		case WHIRLPOOL:
			cuKernel_whirlpool <<<numBlocks,numThreads>>>(dev_blockPwd, dev_blockPwd_init, dev_blockPwd_length, dev_headerKey,bsize);
			break;
	}
	
	// GPU Kernel: Encryption algorithms
	switch(encryptionAlgorithm){
		case AES:
			cuKernel_aes<<<numBlocks,numThreads>>>(dev_headerKey, dev_result, bsize);
			break;
		case SERPENT:
			cuKernel_serpent<<<numBlocks,numThreads>>>(dev_headerKey, dev_result, bsize);
			break;
		case TWOFISH:
			cuKernel_twofish<<<numBlocks,numThreads>>>(dev_headerKey, dev_result, bsize);
			break;
	}
	
	// Stop timer
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&time, tstart, tstop);
	/*
	HANDLE_ERROR( cudaMemcpy(host_blockPwd, dev_blockPwd, bsize*PASSWORD_MAXSIZE*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR( cudaMemcpy(host_blockPwd_init, dev_blockPwd_init, bsize*sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR( cudaMemcpy(host_blockPwd_length, dev_blockPwd_length, bsize*sizeof(int), cudaMemcpyDeviceToHost));
	printf("host_blockPwd_init: ");
	for (int i=0;i<bsize;i++)
	  printf("%d",host_blockPwd_init[i]);
	printf("\nhost_blockPwd_length: ");
	for (int i=0;i<bsize;i++)
	  printf("%d",host_blockPwd_length[i]);
	printf("\nhost_blockPwd: ");	
	for (int i=0;i<bsize*PASSWORD_MAXSIZE;i++)
	  printf("%c",host_blockPwd[i]);
	printf("\n");
	*/
	// Copy memory result from gpu to host
	HANDLE_ERROR( cudaMemcpy(result, dev_result, bsize*sizeof(short int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dev_charset));
	return time;
}
   
// GPU memory initialization: malloc anc constant symbols
void cuda_Init (int bsize, unsigned char *salt, unsigned char *header) {
	HANDLE_ERROR(cudaMalloc ( (void **)&dev_blockPwd, 		bsize*PASSWORD_MAXSIZE* sizeof(unsigned char))) ;
	HANDLE_ERROR(cudaMalloc ( (void **)&dev_blockPwd_init,		bsize * sizeof(int))) ;
	HANDLE_ERROR(cudaMalloc ( (void **)&dev_blockPwd_length, 	bsize * sizeof(int))) ;
	HANDLE_ERROR(cudaMalloc ( (void **)&dev_headerKey, 		MAXPKCS5OUTSIZE * bsize * sizeof(unsigned char))) ;
	HANDLE_ERROR(cudaMalloc ( (void **)&dev_result, 		bsize * sizeof(short int)))  ;
	HANDLE_ERROR(cudaMemcpyToSymbol( cSalt, 		salt , SALT_LENGTH* sizeof(unsigned char),0,cudaMemcpyHostToDevice)) ;
	HANDLE_ERROR(cudaMemcpyToSymbol( cHeaderEncrypted, 	header , TC_VOLUME_HEADER_EFFECTIVE_SIZE* sizeof(unsigned char),0,cudaMemcpyHostToDevice)) ;
  
}

// GPU memory free and reset
void cuda_Free(void) {
	cudaFree(dev_salt);
	cudaFree(dev_blockPwd);
	cudaFree(dev_blockPwd_init);
	cudaFree(dev_blockPwd_length);
	cudaFree(dev_result);
	cudaFree(dev_headerKey);
	cudaDeviceReset();
}


// LEGACY: 
/*
void cuda_Core_dictionary ( int block_currentsize, unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, short int *result, int keyDerivationFunction) {

	int size_block=block_currentsize;
	int size_stream=block_currentsize/NSTREAM;
	
	int numBlocks=size_stream/NUMTHREADSXBLOCK+1;
	int numThread=NUMTHREADSXBLOCK;
	if (size_stream<NUMTHREADSXBLOCK)
		numThread=size_stream;

	cudaStream_t stream[NSTREAM];
	for (int i = 0; i < NSTREAM; ++i)
		cudaStreamCreate(&stream[i]);
	
	int lengthpwd[NSTREAM]={0};
	for (int i=0;i<NSTREAM;i++){
	  for (int j=0;j<size_stream;j++) {
		lengthpwd[i]+=blockPwd_length[j+i*size_stream];
	  }
	}
	printf("1-%d 2-%d \n",lengthpwd[0],lengthpwd[1]);
	
	cudaMalloc ( &dev_result, size_block* sizeof(short int)) ;
	cudaMemcpy ( dev_result, result, size_block* sizeof(short int),cudaMemcpyHostToDevice);
	
	unsigned char *host_blockPwd[NSTREAM];
	int *host_blockPwd_init[NSTREAM];
	int *host_blockPwd_length[NSTREAM];
	short int *host_result[NSTREAM];
	
	for (int i =0; i<NSTREAM; i++){
	
		cudaMalloc ( (void **)&dev_blockPwd[i], 	size_stream * PASSWORD_MAXSIZE * sizeof(unsigned char)) ;
		cudaMalloc ( (void **)&dev_blockPwd_init[i], 	size_stream * sizeof(int)) ;
		cudaMalloc ( (void **)&dev_blockPwd_length[i], 	size_stream * sizeof(int)) ;
		cudaMalloc ( (void **)&dev_headerKey[i], 	256 * size_stream * sizeof(unsigned char)) ;
	
	        cudaHostAlloc(&host_blockPwd[i], 	lengthpwd[i]* sizeof(unsigned char), 	cudaHostAllocDefault);
		cudaHostAlloc(&host_blockPwd_init[i], 	size_stream * sizeof(int), 		cudaHostAllocDefault);
		cudaHostAlloc(&host_blockPwd_length[i], size_stream * sizeof(int),	 	cudaHostAllocDefault);
		cudaHostAlloc(&host_result[i], 		size_stream * sizeof(int),	 	cudaHostAllocDefault);
	
		memcpy(host_blockPwd[i], 	blockPwd+((i==0)?0:lengthpwd[i-1]),	lengthpwd[i]*sizeof(unsigned char));
		memcpy(host_blockPwd_init[i], 	blockPwd_init+i*size_stream, 		size_stream*sizeof(int));
		memcpy(host_blockPwd_length[i], blockPwd_length+i*size_stream, 		size_stream*sizeof(int));
			
	}
	
	for (int i = 0; i < NSTREAM; i++){
	  
		cudaMemcpyAsync(dev_blockPwd[i], 	host_blockPwd[i],		lengthpwd[i] * sizeof(unsigned char) , cudaMemcpyHostToDevice, stream[i]) ;
		cudaMemcpyAsync(dev_blockPwd_init[i], 	host_blockPwd_init[i], 		size_stream * sizeof(int) , cudaMemcpyHostToDevice,stream[i]);
		cudaMemcpyAsync(dev_blockPwd_length[i],	host_blockPwd_length[i], 	size_stream * sizeof(int) , cudaMemcpyHostToDevice,stream[i]) ;
		cudaMemcpyAsync(dev_result, 		host_result[0], 		size_stream * sizeof(short int) , cudaMemcpyHostToDevice,stream[0]) ;
		
		
		cuda_Kernel_ripemd160<<<numBlocks,numThread, 0, stream[i]>>>(dev_blockPwd[i], dev_blockPwd_init[i], dev_blockPwd_length[i], dev_headerKey[i], size_stream);
		cuda_Kernel_aes<<<numBlocks,numThread, 0, stream[i]>>>(dev_headerKey[i], dev_result, size_stream);
			
		cudaError_t err=cudaMemcpy(result+i*size_stream, dev_result,	size_stream* sizeof(short int) , cudaMemcpyDeviceToHost) ;
	//	cudaError_t err=cudaMemcpyAsync(host_result[0], dev_result,	size_stream* sizeof(short int) , cudaMemcpyDeviceToHost,stream[0]) ;
		if (err!=cudaSuccess){
			printf("->%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
		}printf("ok %d\n",i);
		cudaThreadSynchronize();
//	memcpy(result, 	host_result[0], 		size_stream*sizeof(int));
	
	
		
		//cuda_Kernel_ripemd160<<<numBlocks,numThread, 0, stream[i]>>>(dev_blockPwd+ i * size_stream, dev_blockPwd_init+ i * size_stream, dev_blockPwd_length+ i * size_stream, dev_headerKey, size_stream);
		//cuda_Kernel_aes<<<numBlocks,numThread, 0, stream[i]>>>(dev_headerKey, dev_result+ i * size_stream, size_stream);
		
		
		//cudaError_t err=cudaMemcpyAsync(host_result+i*size_stream, 	dev_result+i*size_stream,	size_stream* sizeof(short int) , cudaMemcpyDeviceToHost, stream[i]) ;
		
	}
	
	for (int i = 0; i < NSTREAM; i++)
		cudaStreamDestroy(stream[i]);
    
    
	cudaFree(dev_result);
}
*/

/*
 __global__ void cuda_Kernel_charset (
    	unsigned char *salt,
    	unsigned char *headerEncrypted,
    	unsigned short int charset_length,
    	unsigned char *charset,
    	unsigned short int password_length,
    	uint64_t maxcombination,
    	 short int *result, 
	 int keyDerivationFunction)
 {
	uint64_t numData = blockIdx.x*blockDim.x+threadIdx.x;
	__align__(8) unsigned char headerkey[192];
	__align__(8) unsigned char headerDecrypted[512];
	__align__(8) unsigned char pwd[8];

	//__device__ void computePwd (int number, int maxcombination, int charsetlength, unsigned char *charset, int wordlength, unsigned char *word){
	computePwd(numData,maxcombination,charset_length,charset,password_length,pwd);
	pwd[password_length]='\0';
	
	//__device__ void cuda_Pbkdf2_charset_ ( unsigned char *salt, unsigned char *pwd, int pwd_len, unsigned char *headerkey) {
//	cuda_Pbkdf2 ( salt, pwd, password_length, headerkey);

	int value=cuda_Xts (headerEncrypted, headerkey, headerDecrypted);
	if (value==SUCCESS)
		result[numData]=MATCH;
	else
		result[numData]=NOMATCH;
}*/

/*	
__global__ void cuda_Kernel ( unsigned char *salt, unsigned char *headerEncrypted, unsigned char *blockPwd, int *blockPwd_init, int *blockPwd_length, short int *result, int max, int keyDerivationFunction) {
	int value;
	int numData=blockIdx.x*NUMTHREADSXBLOCK+threadIdx.x;

	if (numData>=max) return;

	// Array of unsigned char in the shared memory
	__align__(8) unsigned char headerKey[192];
	__align__(8) unsigned char headerDecrypted[512];

	// Calculate the hash header key
	unsigned char *pwd=blockPwd+blockPwd_init[numData];
	int pwd_len = blockPwd_length[numData];


	if(keyDerivationFunction==RIPEMD160)
		cuda_Pbkdf2 ( salt, pwd, pwd_len, headerKey);
	else if(keyDerivationFunction==SHA512)
		cuda_derive_key_sha512 (  pwd, pwd_len, salt, PKCS5_SALT_SIZE, 1000, headerKey, 64);
	else if(keyDerivationFunction==WHIRLPOOL)
		cuda_derive_key_whirlpool (  pwd, pwd_len, salt, PKCS5_SALT_SIZE, 1000, headerKey, 64);
	else
		;
	
	// Decrypt the header and compare the key
	value=cuda_Xts (headerEncrypted, headerKey,headerDecrypted);

	if (value==SUCCESS)
		result[numData]=MATCH;
	else
		result[numData]=NOMATCH;
}
*/
/*
 * Copyright (C)  2011  Luca Vaccaro
 * Based on TrueCrypt, freely available at http://www.truecrypt.org/
 *
 * TrueCrack is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */
#include "Pkcs5.cuh"


/*
__device__ void cuda_Pbkdf2 ( unsigned char *salt, unsigned char *pwd, int pwd_len, unsigned char *headerkey) {
	SupportPkcs5 support;
	SupportPkcs5 *sup;
	sup = &support;
	int numBlock=0;
	int c, i;

	for(numBlock=0;numBlock<10;numBlock++){
		//  cuda_Pbkdf2 (salt, blockPwd, blockPwd_init, blockPwd_length, headerkey, numData, i);
		
		//INCLUDE: void derive_u_ripemd160 (char *pwd, int pwd_len, char *salt, int salt_len, int iterations, char *u, int b)		
		int b=numBlock;
		unsigned char *u=headerkey+RIPEMD160_DIGESTSIZE*b;

		// iteration 1 
		memset (sup->ccounter, 0, 4);
		sup->ccounter[3] = (char) b+1;
		memcpy (sup->cinit, salt, SALT_LENGTH);	// salt 
		memcpy (&sup->cinit[SALT_LENGTH],sup->ccounter, 4);	// big-endian block number 
		
		cuda_hmac_ripemd160 (pwd, pwd_len, sup->cinit, SALT_LENGTH + 4, sup->cj, sup);
		memcpy (u, sup->cj, RIPEMD160_DIGESTSIZE);
		
		//remaining iterations 
		for (c = 1; c < ITERATIONS; c++)
		{
			cuda_hmac_ripemd160 (pwd, pwd_len, sup->cj, RIPEMD160_DIGESTSIZE, sup->ck,sup);
			for (i = 0; i < RIPEMD160_DIGESTSIZE; i++)
			{
				u[i] ^= sup->ck[i];
				sup->cj[i] = sup->ck[i];
			}
		}
	}
}*/

__device__ void cuda_hmac_ripemd160 (unsigned char *key, int keylen, unsigned char *input, int len, unsigned char *digest)
{
    SupportPkcs5 support;
	SupportPkcs5 *sup=&support;
    int i;
    // If the key is longer than the hash algorithm block size,
    //	   let key = ripemd160(key), as per HMAC specifications.
    if (keylen > RIPEMD160_BLOCKSIZE)
	{
		//RMD160Init(&tctx);
        //RMD160Update(&tctx, (const unsigned char *) key, keylen);
        //RMD160Final(tk, &tctx);
		cuda_RMD160(&sup->ctctx,(unsigned char *) key, keylen,(unsigned char *)NULL,0,sup->ctk);
        key = (unsigned char *) sup->ctk;
        keylen = RIPEMD160_DIGESTSIZE;
		//burn (&ctctx, sizeof(ctctx));	// Prevent leaks
    }
	 /*
	 RMD160(K XOR opad, RMD160(K XOR ipad, text))
	 where K is an n byte key
	 ipad is the byte 0x36 repeated RIPEMD160_BLOCKSIZE times
	 opad is the byte 0x5c repeated RIPEMD160_BLOCKSIZE times
	 and text is the data being protected*/
	 // start out by storing key in pads
	 // XOR key with ipad and opad values
	 for (i=0; i<sizeof(sup->cpad); i++)
		 sup->cpad[i]=0x36;
	 for (i=0; i<keylen; i++)
		 sup->cpad[i] ^= key[i];
	 
	 cuda_RMD160(&sup->ccontext,sup->cpad,RIPEMD160_BLOCKSIZE,(const unsigned char *) input, len, (unsigned char *) digest);
	 
	 for (i=0; i<sizeof(sup->cpad); i++)
		 sup->cpad[i]=0x5c;
	 for (i=0; i<keylen; i++)
		 sup->cpad[i] ^= key[i];
	 cuda_RMD160(&sup->ccontext,sup->cpad,RIPEMD160_BLOCKSIZE,(const unsigned char *) digest, RIPEMD160_DIGESTSIZE, (unsigned char *) digest);
	 
}
/*
__device__ void cuda_hmac_ripemd160 (unsigned char *key, int keylen, unsigned char *input, int len, unsigned char *digest)
{
    RMD160_CTX context;
    unsigned char k_ipad[65];  //inner padding - key XORd with ipad 
    unsigned char k_opad[65];  //outer padding - key XORd with opad
    unsigned char tk[RIPEMD160_DIGESTSIZE];
    int i;
	
    // If the key is longer than the hash algorithm block size, let key = ripemd160(key), as per HMAC specifications. 
    if (keylen > RIPEMD160_BLOCKSIZE)
	{
        RMD160_CTX      tctx;
		
        RMD160Init(&tctx);
        RMD160Update(&tctx, (const unsigned char *) key, keylen);
        RMD160Final(tk, &tctx);
		
        key = ( unsigned char *) tk;
        keylen = RIPEMD160_DIGESTSIZE;
		
		burn (&tctx, sizeof(tctx));	// Prevent leaks
    }
	
	/*
	 
	 RMD160(K XOR opad, RMD160(K XOR ipad, text))
	 
	 where K is an n byte key
	 ipad is the byte 0x36 repeated RIPEMD160_BLOCKSIZE times
	 opad is the byte 0x5c repeated RIPEMD160_BLOCKSIZE times
	 and text is the data being protected 
	
	
	// start out by storing key in pads
	memset(k_ipad, 0x36, sizeof(k_ipad));
    memset(k_opad, 0x5c, sizeof(k_opad));
	
    // XOR key with ipad and opad values
    for (i=0; i<keylen; i++)
	{
        k_ipad[i] ^= key[i];
        k_opad[i] ^= key[i];
    }
	
    //perform inner RIPEMD-160
	
    RMD160Init(&context);           // init context for 1st pass
    RMD160Update(&context, k_ipad, RIPEMD160_BLOCKSIZE);  // start with inner pad
    RMD160Update(&context, (const unsigned char *) input, len); // then text of datagram
    RMD160Final((unsigned char *) digest, &context);         // finish up 1st pass
	
    // perform outer RIPEMD-160
    RMD160Init(&context);           // init context for 2nd pass
    RMD160Update(&context, k_opad, RIPEMD160_BLOCKSIZE);  // start with outer pad 
    // results of 1st hash
    RMD160Update(&context, (const unsigned char *) digest, RIPEMD160_DIGESTSIZE);
    RMD160Final((unsigned char *) digest, &context);         // finish up 2nd pass
	
	// Prevent possible leaks. 
    burn (k_ipad, sizeof(k_ipad));
    burn (k_opad, sizeof(k_opad));
	burn (tk, sizeof(tk));
	burn (&context, sizeof(context));
}
*/

__device__ void cuda_derive_u_ripemd160 (unsigned char *pwd, int pwd_len, unsigned char *salt, int salt_len, int iterations, unsigned char *u, int b)
{
	unsigned char j[RIPEMD160_DIGESTSIZE], k[RIPEMD160_DIGESTSIZE];
	unsigned char init[128];
	unsigned char counter[4];
	int c, i;
	
	/* iteration 1 */
	memset (counter, 0, 4);
	counter[3] = (char) b;
	memcpy (init, salt, salt_len);	/* salt */
	memcpy (&init[salt_len], counter, 4);	/* big-endian block number */
	cuda_hmac_ripemd160 (pwd, pwd_len, init, salt_len + 4, j);
	memcpy (u, j, RIPEMD160_DIGESTSIZE);
	
	/* remaining iterations */
	for (c = 1; c < iterations; c++)
	{
		cuda_hmac_ripemd160 (pwd, pwd_len, j, RIPEMD160_DIGESTSIZE, k);
		for (i = 0; i < RIPEMD160_DIGESTSIZE; i++)
		{
			u[i] ^= k[i];
			j[i] = k[i];
		}
	}
	
	/* Prevent possible leaks. */
	burn (j, sizeof(j));
	burn (k, sizeof(k));
}


__device__ void cuda_derive_key_ripemd160 (unsigned char *pwd, int pwd_len, unsigned char *salt, int salt_len, int iterations, unsigned char *dk, int dklen)
{
	unsigned char u[RIPEMD160_DIGESTSIZE];
	int b, l, r;
	
	if (dklen % RIPEMD160_DIGESTSIZE)
	{
		l = 1 + dklen / RIPEMD160_DIGESTSIZE;
	}
	else
	{
		l = dklen / RIPEMD160_DIGESTSIZE;
	}
	
	r = dklen - (l - 1) * RIPEMD160_DIGESTSIZE;
	
	// first l - 1 blocks 
	for (b = 1; b < l; b++)
	{
		cuda_derive_u_ripemd160 (pwd, pwd_len, salt, salt_len, iterations, u, b);
		memcpy (dk, u, RIPEMD160_DIGESTSIZE);
		dk += RIPEMD160_DIGESTSIZE;
	}
	
	// last block
	cuda_derive_u_ripemd160 (pwd, pwd_len, salt, salt_len, iterations, u, b);
	memcpy (dk, u, r);
	
	// Prevent possible leaks. 
	burn (u, sizeof(u));
	
}













__device__ void cuda_hmac_truncate
  (
	  unsigned char *d1,		/* data to be truncated */
	  unsigned char *d2,		/* truncated data */
	  int len		/* length in bytes to keep */
)
{
	int i;
	for (i = 0; i < len; i++)
		d2[i] = d1[i];
}






__device__ void cuda_hmac_sha512
(
	  unsigned char *k,		/* secret key */
	  int lk,		/* length of the key in bytes */
	  unsigned char *d,		/* data */
	  int ld,		/* length of data in bytes */
	  unsigned char *out,		/* output buffer, at least "t" bytes */
	  int t
)
{
	sha512_ctx ictx, octx;
	unsigned char isha[SHA512_DIGESTSIZE], osha[SHA512_DIGESTSIZE];
	unsigned char key[SHA512_DIGESTSIZE];
	unsigned char buf[SHA512_BLOCKSIZE];
	int i;

    /* If the key is longer than the hash algorithm block size,
	   let key = sha512(key), as per HMAC specifications. */
	if (lk > SHA512_BLOCKSIZE)
	{
		sha512_ctx tctx;

		sha512_begin (&tctx);
		sha512_hash ((unsigned char *) k, lk, &tctx);
		sha512_end ((unsigned char *) key, &tctx);

		k = key;
		lk = SHA512_DIGESTSIZE;

		burn (&tctx, sizeof(tctx));		// Prevent leaks
	}

	/**** Inner Digest ****/

	sha512_begin (&ictx);

	/* Pad the key for inner digest */
	for (i = 0; i < lk; ++i)
		buf[i] = (unsigned char) (k[i] ^ 0x36);
	for (i = lk; i < SHA512_BLOCKSIZE; ++i)
		buf[i] = (unsigned char) 0x36;

	sha512_hash ((unsigned char *) buf, SHA512_BLOCKSIZE, &ictx);
	sha512_hash ((unsigned char *) d, ld, &ictx);

	sha512_end ((unsigned char *) isha, &ictx);

	/**** Outer Digest ****/

	sha512_begin (&octx);

	for (i = 0; i < lk; ++i)
		buf[i] = (unsigned char) (k[i] ^ 0x5C);
	for (i = lk; i < SHA512_BLOCKSIZE; ++i)
		buf[i] = (unsigned char) 0x5C;

	sha512_hash ((unsigned char *) buf, SHA512_BLOCKSIZE, &octx);
	sha512_hash ((unsigned char *) isha, SHA512_DIGESTSIZE, &octx);

	sha512_end ((unsigned char *) osha, &octx);

	/* truncate and print the results */
	t = t > SHA512_DIGESTSIZE ? SHA512_DIGESTSIZE : t;
	cuda_hmac_truncate (osha, out, t);

	/* Prevent leaks */
	burn (&ictx, sizeof(ictx));
	burn (&octx, sizeof(octx));
	burn (isha, sizeof(isha));
	burn (osha, sizeof(osha));
	burn (buf, sizeof(buf));
	burn (key, sizeof(key));
}


__device__ void cuda_derive_u_sha512 (unsigned char *pwd, int pwd_len, unsigned char *salt, int salt_len, int iterations, unsigned char *u, int b)
{
	unsigned char j[SHA512_DIGESTSIZE], k[SHA512_DIGESTSIZE];
	unsigned char init[128];
	unsigned char counter[4];
	int c, i;

	/* iteration 1 */
	memset (counter, 0, 4);
	counter[3] = (char) b;
	memcpy (init, salt, salt_len);	/* salt */
	memcpy (&init[salt_len], counter, 4);	/* big-endian block number */
	cuda_hmac_sha512 (pwd, pwd_len, init, salt_len + 4, j, SHA512_DIGESTSIZE);
	memcpy (u, j, SHA512_DIGESTSIZE);

	/* remaining iterations */
	for (c = 1; c < iterations; c++)
	{
		cuda_hmac_sha512 (pwd, pwd_len, j, SHA512_DIGESTSIZE, k, SHA512_DIGESTSIZE);
		for (i = 0; i < SHA512_DIGESTSIZE; i++)
		{
			u[i] ^= k[i];
			j[i] = k[i];
		}
	}

	/* Prevent possible leaks. */
	burn (j, sizeof(j));
	burn (k, sizeof(k));
}


__device__ void cuda_derive_key_sha512 (unsigned char *pwd, int pwd_len, unsigned char *salt, int salt_len, int iterations, unsigned char *dk, int dklen)
{
	unsigned char u[SHA512_DIGESTSIZE];
	int b, l, r;

	if (dklen % SHA512_DIGESTSIZE)
	{
		l = 1 + dklen / SHA512_DIGESTSIZE;
	}
	else
	{
		l = dklen / SHA512_DIGESTSIZE;
	}

	r = dklen - (l - 1) * SHA512_DIGESTSIZE;

	/* first l - 1 blocks */
	for (b = 1; b < l; b++)
	{
		cuda_derive_u_sha512 (pwd, pwd_len, salt, salt_len, iterations, u, b);
		memcpy (dk, u, SHA512_DIGESTSIZE);
		dk += SHA512_DIGESTSIZE;
	}

	/* last block */
	cuda_derive_u_sha512 (pwd, pwd_len, salt, salt_len, iterations, u, b);
	memcpy (dk, u, r);


	/* Prevent possible leaks. */
	burn (u, sizeof(u));
}









__device__ void cuda_hmac_whirlpool
(
	  unsigned char *k,		/* secret key */
	  int lk,		/* length of the key in bytes */
	  unsigned char *d,		/* data */
	  int ld,		/* length of data in bytes */
	  unsigned char *out,	/* output buffer, at least "t" bytes */
	  int t
)
{
	WHIRLPOOL_CTX ictx, octx;
	unsigned char iwhi[WHIRLPOOL_DIGESTSIZE], owhi[WHIRLPOOL_DIGESTSIZE];
	unsigned char key[WHIRLPOOL_DIGESTSIZE];
	unsigned char buf[WHIRLPOOL_BLOCKSIZE];
	int i;

    /* If the key is longer than the hash algorithm block size,
	   let key = whirlpool(key), as per HMAC specifications. */
	if (lk > WHIRLPOOL_BLOCKSIZE)
	{
		WHIRLPOOL_CTX tctx;

		WHIRLPOOL_init (&tctx);
		WHIRLPOOL_add ((unsigned char *) k, lk * 8, &tctx);
		WHIRLPOOL_finalize (&tctx, (unsigned char *) key);

		k = key;
		lk = WHIRLPOOL_DIGESTSIZE;

		burn (&tctx, sizeof(tctx));		// Prevent leaks
	}

	/**** Inner Digest ****/

	WHIRLPOOL_init (&ictx);

	/* Pad the key for inner digest */
	for (i = 0; i < lk; ++i)
		buf[i] = (unsigned char) (k[i] ^ 0x36);
	for (i = lk; i < WHIRLPOOL_BLOCKSIZE; ++i)
		buf[i] = (unsigned char) 0x36;

	WHIRLPOOL_add ((unsigned char *) buf, WHIRLPOOL_BLOCKSIZE * 8, &ictx);
	WHIRLPOOL_add ((unsigned char *) d, ld * 8, &ictx);

	WHIRLPOOL_finalize (&ictx, (unsigned char *) iwhi);

	/**** Outer Digest ****/

	WHIRLPOOL_init (&octx);

	for (i = 0; i < lk; ++i)
		buf[i] = (unsigned char) (k[i] ^ 0x5C);
	for (i = lk; i < WHIRLPOOL_BLOCKSIZE; ++i)
		buf[i] = (unsigned char) 0x5C;

	WHIRLPOOL_add ((unsigned char *) buf, WHIRLPOOL_BLOCKSIZE * 8, &octx);
	WHIRLPOOL_add ((unsigned char *) iwhi, WHIRLPOOL_DIGESTSIZE * 8, &octx);

	WHIRLPOOL_finalize (&octx, (unsigned char *) owhi);

	/* truncate and print the results */
	t = t > WHIRLPOOL_DIGESTSIZE ? WHIRLPOOL_DIGESTSIZE : t;
	cuda_hmac_truncate (owhi, out, t);

	/* Prevent possible leaks. */
	burn (&ictx, sizeof(ictx));
	burn (&octx, sizeof(octx));
	burn (owhi, sizeof(owhi));
	burn (iwhi, sizeof(iwhi));
	burn (buf, sizeof(buf));
	burn (key, sizeof(key));
}

__device__ void cuda_derive_u_whirlpool (unsigned char *pwd, int pwd_len, unsigned char *salt, int salt_len, int iterations, unsigned char *u, int b)
{
	unsigned char j[WHIRLPOOL_DIGESTSIZE], k[WHIRLPOOL_DIGESTSIZE];
	unsigned char init[128];
	unsigned char counter[4];
	int c, i;

	/* iteration 1 */
	memset (counter, 0, 4);
	counter[3] = (char) b;
	memcpy (init, salt, salt_len);	/* salt */
	memcpy (&init[salt_len], counter, 4);	/* big-endian block number */
	cuda_hmac_whirlpool (pwd, pwd_len, init, salt_len + 4, j, WHIRLPOOL_DIGESTSIZE);
	memcpy (u, j, WHIRLPOOL_DIGESTSIZE);

	/* remaining iterations */
	for (c = 1; c < iterations; c++)
	{
		cuda_hmac_whirlpool (pwd, pwd_len, j, WHIRLPOOL_DIGESTSIZE, k, WHIRLPOOL_DIGESTSIZE);
		for (i = 0; i < WHIRLPOOL_DIGESTSIZE; i++)
		{
			u[i] ^= k[i];
			j[i] = k[i];
		}
	}

	/* Prevent possible leaks. */
	burn (j, sizeof(j));
	burn (k, sizeof(k));
}

__device__ void cuda_derive_key_whirlpool (unsigned char *pwd, int pwd_len, unsigned char *salt, int salt_len, int iterations, unsigned char *dk, int dklen)
{
	unsigned char u[WHIRLPOOL_DIGESTSIZE];
	int b, l, r;

	if (dklen % WHIRLPOOL_DIGESTSIZE)
	{
		l = 1 + dklen / WHIRLPOOL_DIGESTSIZE;
	}
	else
	{
		l = dklen / WHIRLPOOL_DIGESTSIZE;
	}

	r = dklen - (l - 1) * WHIRLPOOL_DIGESTSIZE;

	/* first l - 1 blocks */
	for (b = 1; b < l; b++)
	{
		cuda_derive_u_whirlpool (pwd, pwd_len, salt, salt_len, iterations, u, b);
		memcpy (dk, u, WHIRLPOOL_DIGESTSIZE);
		dk += WHIRLPOOL_DIGESTSIZE;
	}

	/* last block */
	cuda_derive_u_whirlpool (pwd, pwd_len, salt, salt_len, iterations, u, b);
	memcpy (dk, u, r);


	/* Prevent possible leaks. */
	burn (u, sizeof(u));
}/*
Collection of source files for AES encryption algorithm
- Aescrypt.c
- Aeskey.c
- Aestab.c
*/

//#include "Aes.h"


#include "Aestab.cu"
#include "Aeskey.cu"
#include "Aescrypt.cu"
// serpent.cpp - written and placed in the public domain by Wei Dai

/* Adapted for TrueCrypt */

#ifdef TC_WINDOWS_BOOT
#pragma optimize ("t", on)
#endif

#include "Serpent.cuh"
#include "Common/Endian.h"

#include <memory.h>

#if defined(_WIN32) && !defined(_DEBUG)
#include <stdlib.h>
#define rotlFixed _rotl
#define rotrFixed _rotr
#else
#define rotlFixed(x,n)   (((x) << (n)) | ((x) >> (32 - (n))))
#define rotrFixed(x,n)   (((x) >> (n)) | ((x) << (32 - (n))))
#endif



#define TC_MINIMIZE_CODE_SIZE


// linear transformation
#define LT(i,a,b,c,d,e)	{\
	a = rotlFixed(a, 13);	\
	c = rotlFixed(c, 3); 	\
	d = rotlFixed(d ^ c ^ (a << 3), 7); 	\
	b = rotlFixed(b ^ a ^ c, 1); 	\
	a = rotlFixed(a ^ b ^ d, 5); 		\
	c = rotlFixed(c ^ d ^ (b << 7), 22);}

// inverse linear transformation
#define ILT(i,a,b,c,d,e)	{\
	c = rotrFixed(c, 22);	\
	a = rotrFixed(a, 5); 	\
	c ^= d ^ (b << 7);	\
	a ^= b ^ d; 		\
	b = rotrFixed(b, 1); 	\
	d = rotrFixed(d, 7) ^ c ^ (a << 3);	\
	b ^= a ^ c; 		\
	c = rotrFixed(c, 3); 	\
	a = rotrFixed(a, 13);}

// order of output from S-box functions
#define beforeS0(f) f(0,a,b,c,d,e)
#define afterS0(f) f(1,b,e,c,a,d)
#define afterS1(f) f(2,c,b,a,e,d)
#define afterS2(f) f(3,a,e,b,d,c)
#define afterS3(f) f(4,e,b,d,c,a)
#define afterS4(f) f(5,b,a,e,c,d)
#define afterS5(f) f(6,a,c,b,e,d)
#define afterS6(f) f(7,a,c,d,b,e)
#define afterS7(f) f(8,d,e,b,a,c)

// order of output from inverse S-box functions
#define beforeI7(f) f(8,a,b,c,d,e)
#define afterI7(f) f(7,d,a,b,e,c)
#define afterI6(f) f(6,a,b,c,e,d)
#define afterI5(f) f(5,b,d,e,c,a)
#define afterI4(f) f(4,b,c,e,a,d)
#define afterI3(f) f(3,a,b,e,c,d)
#define afterI2(f) f(2,b,d,e,c,a)
#define afterI1(f) f(1,a,b,c,e,d)
#define afterI0(f) f(0,a,d,b,e,c)

// The instruction sequences for the S-box functions 
// come from Dag Arne Osvik's paper "Speeding up Serpent".

#define S0(i, r0, r1, r2, r3, r4) \
       {           \
    r3 ^= r0;   \
    r4 = r1;   \
    r1 &= r3;   \
    r4 ^= r2;   \
    r1 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r4;   \
    r4 ^= r3;   \
    r3 ^= r2;   \
    r2 |= r1;   \
    r2 ^= r4;   \
    r4 = ~r4;      \
    r4 |= r1;   \
    r1 ^= r3;   \
    r1 ^= r4;   \
    r3 |= r0;   \
    r1 ^= r3;   \
    r4 ^= r3;   \
            }

#define I0(i, r0, r1, r2, r3, r4) \
       {           \
    r2 = ~r2;      \
    r4 = r1;   \
    r1 |= r0;   \
    r4 = ~r4;      \
    r1 ^= r2;   \
    r2 |= r4;   \
    r1 ^= r3;   \
    r0 ^= r4;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r4 ^= r0;   \
    r0 |= r1;   \
    r0 ^= r2;   \
    r3 ^= r4;   \
    r2 ^= r1;   \
    r3 ^= r0;   \
    r3 ^= r1;   \
    r2 &= r3;   \
    r4 ^= r2;   \
            }

#define S1(i, r0, r1, r2, r3, r4) \
       {           \
    r0 = ~r0;      \
    r2 = ~r2;      \
    r4 = r0;   \
    r0 &= r1;   \
    r2 ^= r0;   \
    r0 |= r3;   \
    r3 ^= r2;   \
    r1 ^= r0;   \
    r0 ^= r4;   \
    r4 |= r1;   \
    r1 ^= r3;   \
    r2 |= r0;   \
    r2 &= r4;   \
    r0 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r0;   \
    r0 &= r2;   \
    r0 ^= r4;   \
            }

#define I1(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r1;   \
    r1 ^= r3;   \
    r3 &= r1;   \
    r4 ^= r2;   \
    r3 ^= r0;   \
    r0 |= r1;   \
    r2 ^= r3;   \
    r0 ^= r4;   \
    r0 |= r2;   \
    r1 ^= r3;   \
    r0 ^= r1;   \
    r1 |= r3;   \
    r1 ^= r0;   \
    r4 = ~r4;      \
    r4 ^= r1;   \
    r1 |= r0;   \
    r1 ^= r0;   \
    r1 |= r4;   \
    r3 ^= r1;   \
            }

#define S2(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r0;   \
    r0 &= r2;   \
    r0 ^= r3;   \
    r2 ^= r1;   \
    r2 ^= r0;   \
    r3 |= r4;   \
    r3 ^= r1;   \
    r4 ^= r2;   \
    r1 = r3;   \
    r3 |= r4;   \
    r3 ^= r0;   \
    r0 &= r1;   \
    r4 ^= r0;   \
    r1 ^= r3;   \
    r1 ^= r4;   \
    r4 = ~r4;      \
            }

#define I2(i, r0, r1, r2, r3, r4) \
       {           \
    r2 ^= r3;   \
    r3 ^= r0;   \
    r4 = r3;   \
    r3 &= r2;   \
    r3 ^= r1;   \
    r1 |= r2;   \
    r1 ^= r4;   \
    r4 &= r3;   \
    r2 ^= r3;   \
    r4 &= r0;   \
    r4 ^= r2;   \
    r2 &= r1;   \
    r2 |= r0;   \
    r3 = ~r3;      \
    r2 ^= r3;   \
    r0 ^= r3;   \
    r0 &= r1;   \
    r3 ^= r4;   \
    r3 ^= r0;   \
            }

#define S3(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r0;   \
    r0 |= r3;   \
    r3 ^= r1;   \
    r1 &= r4;   \
    r4 ^= r2;   \
    r2 ^= r3;   \
    r3 &= r0;   \
    r4 |= r1;   \
    r3 ^= r4;   \
    r0 ^= r1;   \
    r4 &= r0;   \
    r1 ^= r3;   \
    r4 ^= r2;   \
    r1 |= r0;   \
    r1 ^= r2;   \
    r0 ^= r3;   \
    r2 = r1;   \
    r1 |= r3;   \
    r1 ^= r0;   \
            }

#define I3(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r0;   \
    r0 &= r4;   \
    r4 ^= r3;   \
    r3 |= r1;   \
    r3 ^= r2;   \
    r0 ^= r4;   \
    r2 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r1;   \
    r4 ^= r2;   \
    r2 &= r3;   \
    r1 |= r3;   \
    r1 ^= r2;   \
    r4 ^= r0;   \
    r2 ^= r4;   \
            }

#define S4(i, r0, r1, r2, r3, r4) \
       {           \
    r1 ^= r3;   \
    r3 = ~r3;      \
    r2 ^= r3;   \
    r3 ^= r0;   \
    r4 = r1;   \
    r1 &= r3;   \
    r1 ^= r2;   \
    r4 ^= r3;   \
    r0 ^= r4;   \
    r2 &= r4;   \
    r2 ^= r0;   \
    r0 &= r1;   \
    r3 ^= r0;   \
    r4 |= r1;   \
    r4 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r2;   \
    r2 &= r3;   \
    r0 = ~r0;      \
    r4 ^= r2;   \
            }

#define I4(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 &= r3;   \
    r2 ^= r1;   \
    r1 |= r3;   \
    r1 &= r0;   \
    r4 ^= r2;   \
    r4 ^= r1;   \
    r1 &= r2;   \
    r0 = ~r0;      \
    r3 ^= r4;   \
    r1 ^= r3;   \
    r3 &= r0;   \
    r3 ^= r2;   \
    r0 ^= r1;   \
    r2 &= r0;   \
    r3 ^= r0;   \
    r2 ^= r4;   \
    r2 |= r3;   \
    r3 ^= r0;   \
    r2 ^= r1;   \
            }

#define S5(i, r0, r1, r2, r3, r4) \
       {           \
    r0 ^= r1;   \
    r1 ^= r3;   \
    r3 = ~r3;      \
    r4 = r1;   \
    r1 &= r0;   \
    r2 ^= r3;   \
    r1 ^= r2;   \
    r2 |= r4;   \
    r4 ^= r3;   \
    r3 &= r1;   \
    r3 ^= r0;   \
    r4 ^= r1;   \
    r4 ^= r2;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r2 = ~r2;      \
    r0 ^= r4;   \
    r4 |= r3;   \
    r2 ^= r4;   \
            }

#define I5(i, r0, r1, r2, r3, r4) \
       {           \
    r1 = ~r1;      \
    r4 = r3;   \
    r2 ^= r1;   \
    r3 |= r0;   \
    r3 ^= r2;   \
    r2 |= r1;   \
    r2 &= r0;   \
    r4 ^= r3;   \
    r2 ^= r4;   \
    r4 |= r0;   \
    r4 ^= r1;   \
    r1 &= r2;   \
    r1 ^= r3;   \
    r4 ^= r2;   \
    r3 &= r4;   \
    r4 ^= r1;   \
    r3 ^= r0;   \
    r3 ^= r4;   \
    r4 = ~r4;      \
            }

#define S6(i, r0, r1, r2, r3, r4) \
       {           \
    r2 = ~r2;      \
    r4 = r3;   \
    r3 &= r0;   \
    r0 ^= r4;   \
    r3 ^= r2;   \
    r2 |= r4;   \
    r1 ^= r3;   \
    r2 ^= r0;   \
    r0 |= r1;   \
    r2 ^= r1;   \
    r4 ^= r0;   \
    r0 |= r3;   \
    r0 ^= r2;   \
    r4 ^= r3;   \
    r4 ^= r0;   \
    r3 = ~r3;      \
    r2 &= r4;   \
    r2 ^= r3;   \
            }

#define I6(i, r0, r1, r2, r3, r4) \
       {           \
    r0 ^= r2;   \
    r4 = r2;   \
    r2 &= r0;   \
    r4 ^= r3;   \
    r2 = ~r2;      \
    r3 ^= r1;   \
    r2 ^= r3;   \
    r4 |= r0;   \
    r0 ^= r2;   \
    r3 ^= r4;   \
    r4 ^= r1;   \
    r1 &= r3;   \
    r1 ^= r0;   \
    r0 ^= r3;   \
    r0 |= r2;   \
    r3 ^= r1;   \
    r4 ^= r0;   \
            }

#define S7(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 &= r1;   \
    r2 ^= r3;   \
    r3 &= r1;   \
    r4 ^= r2;   \
    r2 ^= r1;   \
    r1 ^= r0;   \
    r0 |= r4;   \
    r0 ^= r2;   \
    r3 ^= r1;   \
    r2 ^= r3;   \
    r3 &= r0;   \
    r3 ^= r4;   \
    r4 ^= r2;   \
    r2 &= r0;   \
    r4 = ~r4;      \
    r2 ^= r4;   \
    r4 &= r0;   \
    r1 ^= r3;   \
    r4 ^= r1;   \
            }

#define I7(i, r0, r1, r2, r3, r4) \
       {           \
    r4 = r2;   \
    r2 ^= r0;   \
    r0 &= r3;   \
    r2 = ~r2;      \
    r4 |= r3;   \
    r3 ^= r1;   \
    r1 |= r0;   \
    r0 ^= r2;   \
    r2 &= r4;   \
    r1 ^= r2;   \
    r2 ^= r0;   \
    r0 |= r2;   \
    r3 &= r4;   \
    r0 ^= r3;   \
    r4 ^= r1;   \
    r3 ^= r4;   \
    r4 |= r0;   \
    r3 ^= r2;   \
    r4 ^= r2;   \
            }

// key xor
#define KX(r, a, b, c, d, e)	{\
	a ^= k[4 * r + 0]; \
	b ^= k[4 * r + 1]; \
	c ^= k[4 * r + 2]; \
	d ^= k[4 * r + 3];}


#ifdef TC_MINIMIZE_CODE_SIZE

__device__  void S0f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{
	*r3 ^= *r0;
	*r4 = *r1;
	*r1 &= *r3;
	*r4 ^= *r2;
	*r1 ^= *r0;
	*r0 |= *r3;
	*r0 ^= *r4;
	*r4 ^= *r3;
	*r3 ^= *r2;
	*r2 |= *r1;
	*r2 ^= *r4;
	*r4 = ~*r4;
	*r4 |= *r1;
	*r1 ^= *r3;
	*r1 ^= *r4;
	*r3 |= *r0;
	*r1 ^= *r3;
	*r4 ^= *r3;
}

__device__  void S1f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
    *r0 = ~*r0;   
    *r2 = ~*r2;   
    *r4 = *r0;
    *r0 &= *r1;
    *r2 ^= *r0;
    *r0 |= *r3;
    *r3 ^= *r2;
    *r1 ^= *r0;
    *r0 ^= *r4;
    *r4 |= *r1;
    *r1 ^= *r3;
    *r2 |= *r0;
    *r2 &= *r4;
    *r0 ^= *r1;
    *r1 &= *r2;
    *r1 ^= *r0;
    *r0 &= *r2;
    *r0 ^= *r4;
}

__device__  void S2f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
	*r4 = *r0;
	*r0 &= *r2;
	*r0 ^= *r3;
	*r2 ^= *r1;
	*r2 ^= *r0;
	*r3 |= *r4;
	*r3 ^= *r1;
	*r4 ^= *r2;
	*r1 = *r3;
	*r3 |= *r4;
	*r3 ^= *r0;
	*r0 &= *r1;
	*r4 ^= *r0;
	*r1 ^= *r3;
	*r1 ^= *r4;
	*r4 = ~*r4;   
}

__device__  void S3f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
	*r4 = *r0;
	*r0 |= *r3;
	*r3 ^= *r1;
	*r1 &= *r4;
	*r4 ^= *r2;
	*r2 ^= *r3;
	*r3 &= *r0;
	*r4 |= *r1;
	*r3 ^= *r4;
	*r0 ^= *r1;
	*r4 &= *r0;
	*r1 ^= *r3;
	*r4 ^= *r2;
	*r1 |= *r0;
	*r1 ^= *r2;
	*r0 ^= *r3;
	*r2 = *r1;
	*r1 |= *r3;
	*r1 ^= *r0;
}

__device__  void S4f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
	*r1 ^= *r3;
	*r3 = ~*r3;   
	*r2 ^= *r3;
	*r3 ^= *r0;
	*r4 = *r1;
	*r1 &= *r3;
	*r1 ^= *r2;
	*r4 ^= *r3;
	*r0 ^= *r4;
	*r2 &= *r4;
	*r2 ^= *r0;
	*r0 &= *r1;
	*r3 ^= *r0;
	*r4 |= *r1;
	*r4 ^= *r0;
	*r0 |= *r3;
	*r0 ^= *r2;
	*r2 &= *r3;
	*r0 = ~*r0;   
	*r4 ^= *r2;
}

__device__  void S5f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
	*r0 ^= *r1;
	*r1 ^= *r3;
	*r3 = ~*r3;   
	*r4 = *r1;
	*r1 &= *r0;
	*r2 ^= *r3;
	*r1 ^= *r2;
	*r2 |= *r4;
	*r4 ^= *r3;
	*r3 &= *r1;
	*r3 ^= *r0;
	*r4 ^= *r1;
	*r4 ^= *r2;
	*r2 ^= *r0;
	*r0 &= *r3;
	*r2 = ~*r2;   
	*r0 ^= *r4;
	*r4 |= *r3;
	*r2 ^= *r4;
}

__device__  void S6f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
	*r2 = ~*r2;   
	*r4 = *r3;
	*r3 &= *r0;
	*r0 ^= *r4;
	*r3 ^= *r2;
	*r2 |= *r4;
	*r1 ^= *r3;
	*r2 ^= *r0;
	*r0 |= *r1;
	*r2 ^= *r1;
	*r4 ^= *r0;
	*r0 |= *r3;
	*r0 ^= *r2;
	*r4 ^= *r3;
	*r4 ^= *r0;
	*r3 = ~*r3;   
	*r2 &= *r4;
	*r2 ^= *r3;
}

__device__  void S7f (unsigned __int32 *r0, unsigned __int32 *r1, unsigned __int32 *r2, unsigned __int32 *r3, unsigned __int32 *r4)
{        
	*r4 = *r2;
	*r2 &= *r1;
	*r2 ^= *r3;
	*r3 &= *r1;
	*r4 ^= *r2;
	*r2 ^= *r1;
	*r1 ^= *r0;
	*r0 |= *r4;
	*r0 ^= *r2;
	*r3 ^= *r1;
	*r2 ^= *r3;
	*r3 &= *r0;
	*r3 ^= *r4;
	*r4 ^= *r2;
	*r2 &= *r0;
	*r4 = ~*r4;   
	*r2 ^= *r4;
	*r4 &= *r0;
	*r1 ^= *r3;
	*r4 ^= *r1;
}

__device__  void KXf (const unsigned __int32 *k, unsigned int r, unsigned __int32 *a, unsigned __int32 *b, unsigned __int32 *c, unsigned __int32 *d)
{
	*a ^= k[r];
	*b ^= k[r + 1];
	*c ^= k[r + 2];
	*d ^= k[r + 3];
}

#endif // TC_MINIMIZE_CODE_SIZE

#ifndef TC_MINIMIZE_CODE_SIZE

__device__ void serpent_set_key(const unsigned __int8 userKey[], int keylen, unsigned __int8 *ks)
{
	unsigned __int32 a,b,c,d,e;
	unsigned __int32 *k = (unsigned __int32 *)ks;
	unsigned __int32 t;
	int i;

	for (i = 0; i < keylen / (int)sizeof(__int32); i++)
		k[i] = LE32(((unsigned __int32*)userKey)[i]);

	if (keylen < 32)
		k[keylen/4] |= (unsigned __int32)1 << ((keylen%4)*8);

	k += 8;
	t = k[-1];
	for (i = 0; i < 132; ++i)
		k[i] = t = rotlFixed(k[i-8] ^ k[i-5] ^ k[i-3] ^ t ^ 0x9e3779b9 ^ i, 11);
	k -= 20;

#define LK(r, a, b, c, d, e)	{\
	a = k[(8-r)*4 + 0];		\
	b = k[(8-r)*4 + 1];		\
	c = k[(8-r)*4 + 2];		\
	d = k[(8-r)*4 + 3];}

#define SK(r, a, b, c, d, e)	{\
	k[(8-r)*4 + 4] = a;		\
	k[(8-r)*4 + 5] = b;		\
	k[(8-r)*4 + 6] = c;		\
	k[(8-r)*4 + 7] = d;}	\

	for (i=0; i<4; i++)
	{
		afterS2(LK); afterS2(S3); afterS3(SK);
		afterS1(LK); afterS1(S2); afterS2(SK);
		afterS0(LK); afterS0(S1); afterS1(SK);
		beforeS0(LK); beforeS0(S0); afterS0(SK);
		k += 8*4;
		afterS6(LK); afterS6(S7); afterS7(SK);
		afterS5(LK); afterS5(S6); afterS6(SK);
		afterS4(LK); afterS4(S5); afterS5(SK);
		afterS3(LK); afterS3(S4); afterS4(SK);
	}
	afterS2(LK); afterS2(S3); afterS3(SK);
}

#else // TC_MINIMIZE_CODE_SIZE

__device__  void LKf (unsigned __int32 *k, unsigned int r, unsigned __int32 *a, unsigned __int32 *b, unsigned __int32 *c, unsigned __int32 *d)
{
	*a = k[r];
	*b = k[r + 1];
	*c = k[r + 2];
	*d = k[r + 3];
}

__device__  void SKf (unsigned __int32 *k, unsigned int r, unsigned __int32 *a, unsigned __int32 *b, unsigned __int32 *c, unsigned __int32 *d)
{
	k[r + 4] = *a;
	k[r + 5] = *b;
	k[r + 6] = *c;
	k[r + 7] = *d;
}

__device__ void serpent_set_key(const unsigned __int8 userKey[], int keylen, unsigned __int8 *ks)
{
	unsigned __int32 a,b,c,d,e;
	unsigned __int32 *k = (unsigned __int32 *)ks;
	unsigned __int32 t;	
	int i;

	for (i = 0; i < keylen / (int)sizeof(__int32); i++)
		k[i] = LE32(((unsigned __int32*)userKey)[i]);

	if (keylen < 32)
		k[keylen/4] |= (unsigned __int32)1 << ((keylen%4)*8);

	k += 8;
	t = k[-1];
	for (i = 0; i < 132; ++i)
		k[i] = t = rotlFixed(k[i-8] ^ k[i-5] ^ k[i-3] ^ t ^ 0x9e3779b9 ^ i, 11);
	k -= 20;

	for (i=0; i<4; i++)
	{
		LKf (k, 20, &a, &e, &b, &d); S3f (&a, &e, &b, &d, &c); SKf (k, 16, &e, &b, &d, &c);
		LKf (k, 24, &c, &b, &a, &e); S2f (&c, &b, &a, &e, &d); SKf (k, 20, &a, &e, &b, &d);
		LKf (k, 28, &b, &e, &c, &a); S1f (&b, &e, &c, &a, &d); SKf (k, 24, &c, &b, &a, &e);
		LKf (k, 32, &a, &b, &c, &d); S0f (&a, &b, &c, &d, &e); SKf (k, 28, &b, &e, &c, &a);
		k += 8*4;
		LKf (k,  4, &a, &c, &d, &b); S7f (&a, &c, &d, &b, &e); SKf (k,  0, &d, &e, &b, &a);
		LKf (k,  8, &a, &c, &b, &e); S6f (&a, &c, &b, &e, &d); SKf (k,  4, &a, &c, &d, &b);
		LKf (k, 12, &b, &a, &e, &c); S5f (&b, &a, &e, &c, &d); SKf (k,  8, &a, &c, &b, &e);
		LKf (k, 16, &e, &b, &d, &c); S4f (&e, &b, &d, &c, &a); SKf (k, 12, &b, &a, &e, &c);
	}
	LKf (k, 20, &a, &e, &b, &d); S3f (&a, &e, &b, &d, &c); SKf (k, 16, &e, &b, &d, &c);
	
}

#endif // TC_MINIMIZE_CODE_SIZE


#ifndef TC_MINIMIZE_CODE_SIZE

__device__ void serpent_encrypt(const unsigned __int8 *inBlock, unsigned __int8 *outBlock, unsigned __int8 *ks)
{
	unsigned __int32 a, b, c, d, e;
	unsigned int i=1;
	const unsigned __int32 *k = (unsigned __int32 *)ks + 8;
	unsigned __int32 *in = (unsigned __int32 *) inBlock;
	unsigned __int32 *out = (unsigned __int32 *) outBlock;

    a = LE32(in[0]);
	b = LE32(in[1]);
	c = LE32(in[2]);
	d = LE32(in[3]);

	do
	{
		beforeS0(KX); beforeS0(S0); afterS0(LT);
		afterS0(KX); afterS0(S1); afterS1(LT);
		afterS1(KX); afterS1(S2); afterS2(LT);
		afterS2(KX); afterS2(S3); afterS3(LT);
		afterS3(KX); afterS3(S4); afterS4(LT);
		afterS4(KX); afterS4(S5); afterS5(LT);
		afterS5(KX); afterS5(S6); afterS6(LT);
		afterS6(KX); afterS6(S7);

		if (i == 4)
			break;

		++i;
		c = b;
		b = e;
		e = d;
		d = a;
		a = e;
		k += 32;
		beforeS0(LT);
	}
	while (1);

	afterS7(KX);
	
    out[0] = LE32(d);
	out[1] = LE32(e);
	out[2] = LE32(b);
	out[3] = LE32(a);
}

#else // TC_MINIMIZE_CODE_SIZE

typedef unsigned __int32 uint32;

__device__  void LTf (uint32 *a, uint32 *b, uint32 *c, uint32 *d)
{
	*a = rotlFixed(*a, 13);
	*c = rotlFixed(*c, 3);
	*d = rotlFixed(*d ^ *c ^ (*a << 3), 7);
	*b = rotlFixed(*b ^ *a ^ *c, 1);
	*a = rotlFixed(*a ^ *b ^ *d, 5);
	*c = rotlFixed(*c ^ *d ^ (*b << 7), 22);
}

__device__ void serpent_encrypt(const unsigned __int8 *inBlock, unsigned __int8 *outBlock, unsigned __int8 *ks)
{
	unsigned __int32 a, b, c, d, e;
	unsigned int i=1;
	const unsigned __int32 *k = (unsigned __int32 *)ks + 8;
	unsigned __int32 *in = (unsigned __int32 *) inBlock;
	unsigned __int32 *out = (unsigned __int32 *) outBlock;

    a = LE32(in[0]);
	b = LE32(in[1]);
	c = LE32(in[2]);
	d = LE32(in[3]);

	do
	{
		KXf (k,  0, &a, &b, &c, &d); S0f (&a, &b, &c, &d, &e); LTf (&b, &e, &c, &a);
		KXf (k,  4, &b, &e, &c, &a); S1f (&b, &e, &c, &a, &d); LTf (&c, &b, &a, &e);
		KXf (k,  8, &c, &b, &a, &e); S2f (&c, &b, &a, &e, &d); LTf (&a, &e, &b, &d);
		KXf (k, 12, &a, &e, &b, &d); S3f (&a, &e, &b, &d, &c); LTf (&e, &b, &d, &c);
		KXf (k, 16, &e, &b, &d, &c); S4f (&e, &b, &d, &c, &a); LTf (&b, &a, &e, &c);
		KXf (k, 20, &b, &a, &e, &c); S5f (&b, &a, &e, &c, &d); LTf (&a, &c, &b, &e);
		KXf (k, 24, &a, &c, &b, &e); S6f (&a, &c, &b, &e, &d); LTf (&a, &c, &d, &b);
		KXf (k, 28, &a, &c, &d, &b); S7f (&a, &c, &d, &b, &e);

		if (i == 4)
			break;

		++i;
		c = b;
		b = e;
		e = d;
		d = a;
		a = e;
		k += 32;
		LTf (&a,&b,&c,&d);
	}
	while (1);

	KXf (k, 32, &d, &e, &b, &a);
	
    out[0] = LE32(d);
	out[1] = LE32(e);
	out[2] = LE32(b);
	out[3] = LE32(a);
}

#endif // TC_MINIMIZE_CODE_SIZE

#if !defined (TC_MINIMIZE_CODE_SIZE) || defined (TC_WINDOWS_BOOT_SERPENT)

__device__ void serpent_decrypt(const unsigned __int8 *inBlock, unsigned __int8 *outBlock, unsigned __int8 *ks)
{
	unsigned __int32 a, b, c, d, e;
	const unsigned __int32 *k = (unsigned __int32 *)ks + 104;
	unsigned int i=4;
	unsigned __int32 *in = (unsigned __int32 *) inBlock;
	unsigned __int32 *out = (unsigned __int32 *) outBlock;

    a = LE32(in[0]);
	b = LE32(in[1]);
	c = LE32(in[2]);
	d = LE32(in[3]);

	beforeI7(KX);
	goto start;

	do
	{
		c = b;
		b = d;
		d = e;
		k -= 32;
		beforeI7(ILT);
start:
		beforeI7(I7); afterI7(KX); 
		afterI7(ILT); afterI7(I6); afterI6(KX); 
		afterI6(ILT); afterI6(I5); afterI5(KX); 
		afterI5(ILT); afterI5(I4); afterI4(KX); 
		afterI4(ILT); afterI4(I3); afterI3(KX); 
		afterI3(ILT); afterI3(I2); afterI2(KX); 
		afterI2(ILT); afterI2(I1); afterI1(KX); 
		afterI1(ILT); afterI1(I0); afterI0(KX);
	}
	while (--i != 0);
	
    out[0] = LE32(a);
	out[1] = LE32(d);
	out[2] = LE32(b);
	out[3] = LE32(e);
}

#else // TC_MINIMIZE_CODE_SIZE && !TC_WINDOWS_BOOT_SERPENT

__device__  void ILTf (uint32 *a, uint32 *b, uint32 *c, uint32 *d)
{ 
	*c = rotrFixed(*c, 22);
	*a = rotrFixed(*a, 5);
	*c ^= *d ^ (*b << 7);
	*a ^= *b ^ *d;
	*b = rotrFixed(*b, 1);
	*d = rotrFixed(*d, 7) ^ *c ^ (*a << 3);
	*b ^= *a ^ *c;
	*c = rotrFixed(*c, 3);
	*a = rotrFixed(*a, 13);
}

__device__ void serpent_decrypt(const unsigned __int8 *inBlock, unsigned __int8 *outBlock, unsigned __int8 *ks)
{
	unsigned __int32 a, b, c, d, e;
	const unsigned __int32 *k = (unsigned __int32 *)ks + 104;
	unsigned int i=4;
	unsigned __int32 *in = (unsigned __int32 *) inBlock;
	unsigned __int32 *out = (unsigned __int32 *) outBlock;

    a = LE32(in[0]);
	b = LE32(in[1]);
	c = LE32(in[2]);
	d = LE32(in[3]);

	KXf (k, 32, &a, &b, &c, &d);
	goto start;

	do
	{
		c = b;
		b = d;
		d = e;
		k -= 32;
		beforeI7(ILT);
start:
		beforeI7(I7); KXf (k, 28, &d, &a, &b, &e);
		ILTf (&d, &a, &b, &e); afterI7(I6); KXf (k, 24, &a, &b, &c, &e); 
		ILTf (&a, &b, &c, &e); afterI6(I5); KXf (k, 20, &b, &d, &e, &c); 
		ILTf (&b, &d, &e, &c); afterI5(I4); KXf (k, 16, &b, &c, &e, &a); 
		ILTf (&b, &c, &e, &a); afterI4(I3); KXf (k, 12, &a, &b, &e, &c);
		ILTf (&a, &b, &e, &c); afterI3(I2); KXf (k, 8,  &b, &d, &e, &c);
		ILTf (&b, &d, &e, &c); afterI2(I1); KXf (k, 4,  &a, &b, &c, &e);
		ILTf (&a, &b, &c, &e); afterI1(I0); KXf (k, 0,  &a, &d, &b, &e);
	}
	while (--i != 0);
	
    out[0] = LE32(a);
	out[1] = LE32(d);
	out[2] = LE32(b);
	out[3] = LE32(e);
}

#endif // TC_MINIMIZE_CODE_SIZE && !TC_WINDOWS_BOOT_SERPENT
/*
 ---------------------------------------------------------------------------
 Copyright (c) 1999, Dr Brian Gladman, Worcester, UK.   All rights reserved.

 LICENSE TERMS

 The free distribution and use of this software is allowed (with or without
 changes) provided that:

  1. source code distributions include the above copyright notice, this
     list of conditions and the following disclaimer;

  2. binary distributions include the above copyright notice, this list
     of conditions and the following disclaimer in their documentation;

  3. the name of the copyright holder is not used to endorse products
     built using this software without specific written permission.

 DISCLAIMER

 This software is provided 'as is' with no explicit or implied warranties
 in respect of its properties, including, but not limited to, correctness
 and/or fitness for purpose.
 ---------------------------------------------------------------------------

 My thanks to Doug Whiting and Niels Ferguson for comments that led
 to improvements in this implementation.

 Issue Date: 14th January 1999
*/

/* Adapted for TrueCrypt */


#ifdef TC_WINDOWS_BOOT
#pragma optimize ("tl", on)
#endif

#include "Twofish.cuh"
#include "Common/Endian.h"

#define Q_TABLES
#define M_TABLE

#if !defined (TC_MINIMIZE_CODE_SIZE) || defined (TC_WINDOWS_BOOT_TWOFISH)
#	define MK_TABLE
#	define ONE_STEP
#endif

/* finite field arithmetic for GF(2**8) with the modular    */
/* polynomial x^8 + x^6 + x^5 + x^3 + 1 (0x169)             */

#define G_M 0x0169

__device__ static u1byte  tab_5b[4] = { 0, G_M >> 2, G_M >> 1, (G_M >> 1) ^ (G_M >> 2) };
__device__ static u1byte  tab_ef[4] = { 0, (G_M >> 1) ^ (G_M >> 2), G_M >> 1, G_M >> 2 };

#define ffm_01(x)    (x)
#define ffm_5b(x)   ((x) ^ ((x) >> 2) ^ tab_5b[(x) & 3])
#define ffm_ef(x)   ((x) ^ ((x) >> 1) ^ ((x) >> 2) ^ tab_ef[(x) & 3])

__device__ static u1byte ror4[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
__device__ static u1byte ashx[16] = { 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 5, 14, 7 };

__device__ static u1byte qt0[2][16] = 
{   { 8, 1, 7, 13, 6, 15, 3, 2, 0, 11, 5, 9, 14, 12, 10, 4 },
    { 2, 8, 11, 13, 15, 7, 6, 14, 3, 1, 9, 4, 0, 10, 12, 5 }
};

__device__ static u1byte qt1[2][16] =
{   { 14, 12, 11, 8, 1, 2, 3, 5, 15, 4, 10, 6, 7, 0, 9, 13 }, 
    { 1, 14, 2, 11, 4, 12, 3, 7, 6, 13, 10, 5, 15, 9, 0, 8 }
};

__device__ static u1byte qt2[2][16] = 
{   { 11, 10, 5, 14, 6, 13, 9, 0, 12, 8, 15, 3, 2, 4, 7, 1 },
    { 4, 12, 7, 5, 1, 6, 9, 10, 0, 14, 13, 8, 2, 11, 3, 15 }
};

__device__ static u1byte qt3[2][16] = 
{   { 13, 7, 15, 4, 1, 2, 6, 14, 9, 11, 3, 0, 8, 5, 12, 10 },
    { 11, 9, 5, 1, 12, 3, 13, 14, 6, 4, 7, 15, 2, 0, 8, 10 }
};
 
__device__ static u1byte qp(const u4byte n, const u1byte x)
{   u1byte  a0, a1, a2, a3, a4, b0, b1, b2, b3, b4;

    a0 = x >> 4; b0 = x & 15;
    a1 = a0 ^ b0; b1 = ror4[b0] ^ ashx[a0];
    a2 = qt0[n][a1]; b2 = qt1[n][b1];
    a3 = a2 ^ b2; b3 = ror4[b2] ^ ashx[a2];
    a4 = qt2[n][a3]; b4 = qt3[n][b3];
    return (b4 << 4) | a4;
};

#ifdef  Q_TABLES

__device__ static u4byte  qt_gen = 0;
__device__ static u1byte  q_tab[2][256];

#define q(n,x)  q_tab[n][x]

__device__ static void gen_qtab(void)
{   u4byte  i;

    for(i = 0; i < 256; ++i)
    {       
        q(0,i) = qp(0, (u1byte)i);
        q(1,i) = qp(1, (u1byte)i);
    }
};

#else

#define q(n,x)  qp(n, x)

#endif

#ifdef  M_TABLE

__device__ static u4byte  mt_gen = 0;
__device__ static u4byte  m_tab[4][256];

__device__ static void gen_mtab(void)
{   u4byte  i, f01, f5b, fef;
    
    for(i = 0; i < 256; ++i)
    {
        f01 = q(1,i); f5b = ffm_5b(f01); fef = ffm_ef(f01);
        m_tab[0][i] = f01 + (f5b << 8) + (fef << 16) + (fef << 24);
        m_tab[2][i] = f5b + (fef << 8) + (f01 << 16) + (fef << 24);

        f01 = q(0,i); f5b = ffm_5b(f01); fef = ffm_ef(f01);
        m_tab[1][i] = fef + (fef << 8) + (f5b << 16) + (f01 << 24);
        m_tab[3][i] = f5b + (f01 << 8) + (fef << 16) + (f5b << 24);
    }
};

#define mds(n,x)    m_tab[n][x]

#else

#define fm_00   ffm_01
#define fm_10   ffm_5b
#define fm_20   ffm_ef
#define fm_30   ffm_ef
#define q_0(x)  q(1,x)

#define fm_01   ffm_ef
#define fm_11   ffm_ef
#define fm_21   ffm_5b
#define fm_31   ffm_01
#define q_1(x)  q(0,x)

#define fm_02   ffm_5b
#define fm_12   ffm_ef
#define fm_22   ffm_01
#define fm_32   ffm_ef
#define q_2(x)  q(1,x)

#define fm_03   ffm_5b
#define fm_13   ffm_01
#define fm_23   ffm_ef
#define fm_33   ffm_5b
#define q_3(x)  q(0,x)

#define f_0(n,x)    ((u4byte)fm_0##n(x))
#define f_1(n,x)    ((u4byte)fm_1##n(x) << 8)
#define f_2(n,x)    ((u4byte)fm_2##n(x) << 16)
#define f_3(n,x)    ((u4byte)fm_3##n(x) << 24)

#define mds(n,x)    f_0(n,q_##n(x)) ^ f_1(n,q_##n(x)) ^ f_2(n,q_##n(x)) ^ f_3(n,q_##n(x))

#endif

__device__ static u4byte h_fun(TwofishInstance *instance, const u4byte x, const u4byte key[])
{   u4byte  b0, b1, b2, b3;

#ifndef M_TABLE
    u4byte  m5b_b0, m5b_b1, m5b_b2, m5b_b3;
    u4byte  mef_b0, mef_b1, mef_b2, mef_b3;
#endif

    b0 = extract_byte(x, 0); b1 = extract_byte(x, 1); b2 = extract_byte(x, 2); b3 = extract_byte(x, 3);

    switch(instance->k_len)
    {
    case 4: b0 = q(1, (u1byte) b0) ^ extract_byte(key[3],0);
            b1 = q(0, (u1byte) b1) ^ extract_byte(key[3],1);
            b2 = q(0, (u1byte) b2) ^ extract_byte(key[3],2);
            b3 = q(1, (u1byte) b3) ^ extract_byte(key[3],3);
    case 3: b0 = q(1, (u1byte) b0) ^ extract_byte(key[2],0);
            b1 = q(1, (u1byte) b1) ^ extract_byte(key[2],1);
            b2 = q(0, (u1byte) b2) ^ extract_byte(key[2],2);
            b3 = q(0, (u1byte) b3) ^ extract_byte(key[2],3);
    case 2: b0 = q(0, (u1byte) (q(0, (u1byte) b0) ^ extract_byte(key[1],0))) ^ extract_byte(key[0],0);
            b1 = q(0, (u1byte) (q(1, (u1byte) b1) ^ extract_byte(key[1],1))) ^ extract_byte(key[0],1);
            b2 = q(1, (u1byte) (q(0, (u1byte) b2) ^ extract_byte(key[1],2))) ^ extract_byte(key[0],2);
            b3 = q(1, (u1byte) (q(1, (u1byte) b3) ^ extract_byte(key[1],3))) ^ extract_byte(key[0],3);
    }
#ifdef  M_TABLE

    return  mds(0, b0) ^ mds(1, b1) ^ mds(2, b2) ^ mds(3, b3);

#else

    b0 = q(1, (u1byte) b0); b1 = q(0, (u1byte) b1); b2 = q(1, (u1byte) b2); b3 = q(0, (u1byte) b3);
    m5b_b0 = ffm_5b(b0); m5b_b1 = ffm_5b(b1); m5b_b2 = ffm_5b(b2); m5b_b3 = ffm_5b(b3);
    mef_b0 = ffm_ef(b0); mef_b1 = ffm_ef(b1); mef_b2 = ffm_ef(b2); mef_b3 = ffm_ef(b3);
    b0 ^= mef_b1 ^ m5b_b2 ^ m5b_b3; b3 ^= m5b_b0 ^ mef_b1 ^ mef_b2;
    b2 ^= mef_b0 ^ m5b_b1 ^ mef_b3; b1 ^= mef_b0 ^ mef_b2 ^ m5b_b3;

    return b0 | (b3 << 8) | (b2 << 16) | (b1 << 24);

#endif
};

#ifdef  MK_TABLE

#ifdef  ONE_STEP
//u4byte  mk_tab[4][256];
#else
__device__ static u1byte  sb[4][256];
#endif

#define q20(x)  q(0,q(0,x) ^ extract_byte(key[1],0)) ^ extract_byte(key[0],0)
#define q21(x)  q(0,q(1,x) ^ extract_byte(key[1],1)) ^ extract_byte(key[0],1)
#define q22(x)  q(1,q(0,x) ^ extract_byte(key[1],2)) ^ extract_byte(key[0],2)
#define q23(x)  q(1,q(1,x) ^ extract_byte(key[1],3)) ^ extract_byte(key[0],3)

#define q30(x)  q(0,q(0,q(1, x) ^ extract_byte(key[2],0)) ^ extract_byte(key[1],0)) ^ extract_byte(key[0],0)
#define q31(x)  q(0,q(1,q(1, x) ^ extract_byte(key[2],1)) ^ extract_byte(key[1],1)) ^ extract_byte(key[0],1)
#define q32(x)  q(1,q(0,q(0, x) ^ extract_byte(key[2],2)) ^ extract_byte(key[1],2)) ^ extract_byte(key[0],2)
#define q33(x)  q(1,q(1,q(0, x) ^ extract_byte(key[2],3)) ^ extract_byte(key[1],3)) ^ extract_byte(key[0],3)

#define q40(x)  q(0,q(0,q(1, q(1, x) ^ extract_byte(key[3],0)) ^ extract_byte(key[2],0)) ^ extract_byte(key[1],0)) ^ extract_byte(key[0],0)
#define q41(x)  q(0,q(1,q(1, q(0, x) ^ extract_byte(key[3],1)) ^ extract_byte(key[2],1)) ^ extract_byte(key[1],1)) ^ extract_byte(key[0],1)
#define q42(x)  q(1,q(0,q(0, q(0, x) ^ extract_byte(key[3],2)) ^ extract_byte(key[2],2)) ^ extract_byte(key[1],2)) ^ extract_byte(key[0],2)
#define q43(x)  q(1,q(1,q(0, q(1, x) ^ extract_byte(key[3],3)) ^ extract_byte(key[2],3)) ^ extract_byte(key[1],3)) ^ extract_byte(key[0],3)

__device__ static void gen_mk_tab(TwofishInstance *instance, u4byte key[])
{   u4byte  i;
    u1byte  by;

	u4byte *mk_tab = instance->mk_tab;

    switch(instance->k_len)
    {
    case 2: for(i = 0; i < 256; ++i)
            {
                by = (u1byte)i;
#ifdef ONE_STEP
                mk_tab[0 + 4*i] = mds(0, q20(by)); mk_tab[1 + 4*i] = mds(1, q21(by));
                mk_tab[2 + 4*i] = mds(2, q22(by)); mk_tab[3 + 4*i] = mds(3, q23(by));
#else
                sb[0][i] = q20(by); sb[1][i] = q21(by); 
                sb[2][i] = q22(by); sb[3][i] = q23(by);
#endif
            }
            break;
    
    case 3: for(i = 0; i < 256; ++i)
            {
                by = (u1byte)i;
#ifdef ONE_STEP
                mk_tab[0 + 4*i] = mds(0, q30(by)); mk_tab[1 + 4*i] = mds(1, q31(by));
                mk_tab[2 + 4*i] = mds(2, q32(by)); mk_tab[3 + 4*i] = mds(3, q33(by));
#else
                sb[0][i] = q30(by); sb[1][i] = q31(by); 
                sb[2][i] = q32(by); sb[3][i] = q33(by);
#endif
            }
            break;
    
    case 4: for(i = 0; i < 256; ++i)
            {
                by = (u1byte)i;
#ifdef ONE_STEP
                mk_tab[0 + 4*i] = mds(0, q40(by)); mk_tab[1 + 4*i] = mds(1, q41(by));
                mk_tab[2 + 4*i] = mds(2, q42(by)); mk_tab[3 + 4*i] = mds(3, q43(by));
#else
                sb[0][i] = q40(by); sb[1][i] = q41(by); 
                sb[2][i] = q42(by); sb[3][i] = q43(by);
#endif
            }
    }
};

#  ifdef ONE_STEP
#    define g0_fun(x) ( mk_tab[0 + 4*extract_byte(x,0)] ^ mk_tab[1 + 4*extract_byte(x,1)] \
                      ^ mk_tab[2 + 4*extract_byte(x,2)] ^ mk_tab[3 + 4*extract_byte(x,3)] )
#    define g1_fun(x) ( mk_tab[0 + 4*extract_byte(x,3)] ^ mk_tab[1 + 4*extract_byte(x,0)] \
                      ^ mk_tab[2 + 4*extract_byte(x,1)] ^ mk_tab[3 + 4*extract_byte(x,2)] )


#  else
#    define g0_fun(x) ( mds(0, sb[0][extract_byte(x,0)]) ^ mds(1, sb[1][extract_byte(x,1)]) \
                      ^ mds(2, sb[2][extract_byte(x,2)]) ^ mds(3, sb[3][extract_byte(x,3)]) )
#    define g1_fun(x) ( mds(0, sb[0][extract_byte(x,3)]) ^ mds(1, sb[1][extract_byte(x,0)]) \
                      ^ mds(2, sb[2][extract_byte(x,1)]) ^ mds(3, sb[3][extract_byte(x,2)]) )
#  endif

#else

#define g0_fun(x)   h_fun(instance, x, instance->s_key)
#define g1_fun(x)   h_fun(instance, rotl(x,8), instance->s_key)

#endif

/* The (12,8) Reed Soloman code has the generator polynomial

  g(x) = x^4 + (a + 1/a) * x^3 + a * x^2 + (a + 1/a) * x + 1

where the coefficients are in the finite field GF(2^8) with a
modular polynomial a^8 + a^6 + a^3 + a^2 + 1. To generate the
remainder we have to start with a 12th order polynomial with our
eight input bytes as the coefficients of the 4th to 11th terms. 
That is:

  m[7] * x^11 + m[6] * x^10 ... + m[0] * x^4 + 0 * x^3 +... + 0
  
We then multiply the generator polynomial by m[7] * x^7 and subtract
it - xor in GF(2^8) - from the above to eliminate the x^7 term (the 
artihmetic on the coefficients is done in GF(2^8). We then multiply 
the generator polynomial by x^6 * coeff(x^10) and use this to remove
the x^10 term. We carry on in this way until the x^4 term is removed
so that we are left with:

  r[3] * x^3 + r[2] * x^2 + r[1] 8 x^1 + r[0]

which give the resulting 4 bytes of the remainder. This is equivalent 
to the matrix multiplication in the Twofish description but much faster 
to implement.

*/

#define G_MOD   0x0000014d

__device__ static u4byte mds_rem(u4byte p0, u4byte p1)
{   u4byte  i, t, u;

    for(i = 0; i < 8; ++i)
    {
        t = p1 >> 24;   // get most significant coefficient
        
        p1 = (p1 << 8) | (p0 >> 24); p0 <<= 8;  // shift others up
            
        // multiply t by a (the primitive element - i.e. left shift)

        u = (t << 1); 
        
        if(t & 0x80)            // subtract modular polynomial on overflow
        
            u ^= G_MOD; 

        p1 ^= t ^ (u << 16);    // remove t * (a * x^2 + 1)  

        u ^= (t >> 1);          // form u = a * t + t / a = t * (a + 1 / a); 
        
        if(t & 0x01)            // add the modular polynomial on underflow
        
            u ^= G_MOD >> 1;

        p1 ^= (u << 24) | (u << 8); // remove t * (a + 1/a) * (x^3 + x)
    }

    return p1;
};

/* initialise the key schedule from the user supplied key   */

__device__ u4byte *twofish_set_key(TwofishInstance *instance, const u4byte in_key[], const u4byte key_len)
{   u4byte  i, a, b, me_key[4], mo_key[4];
	u4byte *l_key, *s_key;

	l_key = instance->l_key;
	s_key = instance->s_key;

#ifdef Q_TABLES
    if(!qt_gen)
    {
        gen_qtab(); qt_gen = 1;
    }
#endif

#ifdef M_TABLE
    if(!mt_gen)
    {
        gen_mtab(); mt_gen = 1;
    }
#endif

    instance->k_len = key_len / 64;   /* 2, 3 or 4 */

    for(i = 0; i < instance->k_len; ++i)
    {
        a = LE32(in_key[i + i]);     me_key[i] = a;
        b = LE32(in_key[i + i + 1]); mo_key[i] = b;
        s_key[instance->k_len - i - 1] = mds_rem(a, b);
    }

    for(i = 0; i < 40; i += 2)
    {
        a = 0x01010101 * i; b = a + 0x01010101;
        a = h_fun(instance, a, me_key);
        b = rotl(h_fun(instance, b, mo_key), 8);
        l_key[i] = a + b;
        l_key[i + 1] = rotl(a + 2 * b, 9);
    }

#ifdef MK_TABLE
    gen_mk_tab(instance, s_key);
#endif

    return l_key;
};

/* encrypt a block of text  */

#ifndef TC_MINIMIZE_CODE_SIZE

#define f_rnd(i)                                                    \
    t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);                       \
    blk[2] = rotr(blk[2] ^ (t0 + t1 + l_key[4 * (i) + 8]), 1);      \
    blk[3] = rotl(blk[3], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 9]);  \
    t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);                       \
    blk[0] = rotr(blk[0] ^ (t0 + t1 + l_key[4 * (i) + 10]), 1);     \
    blk[1] = rotl(blk[1], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 11])

__device__ void twofish_encrypt(TwofishInstance *instance, const u4byte in_blk[4], u4byte out_blk[])
{   u4byte  t0, t1, blk[4];

	u4byte *l_key = instance->l_key;
	u4byte *mk_tab = instance->mk_tab;

	blk[0] = LE32(in_blk[0]) ^ l_key[0];
    blk[1] = LE32(in_blk[1]) ^ l_key[1];
    blk[2] = LE32(in_blk[2]) ^ l_key[2];
    blk[3] = LE32(in_blk[3]) ^ l_key[3];

    f_rnd(0); f_rnd(1); f_rnd(2); f_rnd(3);
    f_rnd(4); f_rnd(5); f_rnd(6); f_rnd(7);

    out_blk[0] = LE32(blk[2] ^ l_key[4]);
    out_blk[1] = LE32(blk[3] ^ l_key[5]);
    out_blk[2] = LE32(blk[0] ^ l_key[6]);
    out_blk[3] = LE32(blk[1] ^ l_key[7]); 
};

#else // TC_MINIMIZE_CODE_SIZE

__device__ void twofish_encrypt(TwofishInstance *instance, const u4byte in_blk[4], u4byte out_blk[])
{   u4byte  t0, t1, blk[4];

	u4byte *l_key = instance->l_key;
#ifdef TC_WINDOWS_BOOT_TWOFISH
	u4byte *mk_tab = instance->mk_tab;
#endif
	int i;

	blk[0] = LE32(in_blk[0]) ^ l_key[0];
    blk[1] = LE32(in_blk[1]) ^ l_key[1];
    blk[2] = LE32(in_blk[2]) ^ l_key[2];
    blk[3] = LE32(in_blk[3]) ^ l_key[3];

	for (i = 0; i <= 7; ++i)
	{
		t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);
		blk[2] = rotr(blk[2] ^ (t0 + t1 + l_key[4 * (i) + 8]), 1);
		blk[3] = rotl(blk[3], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 9]);
		t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);
		blk[0] = rotr(blk[0] ^ (t0 + t1 + l_key[4 * (i) + 10]), 1);
		blk[1] = rotl(blk[1], 1) ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]);
	}

    out_blk[0] = LE32(blk[2] ^ l_key[4]);
    out_blk[1] = LE32(blk[3] ^ l_key[5]);
    out_blk[2] = LE32(blk[0] ^ l_key[6]);
    out_blk[3] = LE32(blk[1] ^ l_key[7]); 
};

#endif // TC_MINIMIZE_CODE_SIZE

/* decrypt a block of text  */

#ifndef TC_MINIMIZE_CODE_SIZE

#define i_rnd(i)                                                        \
        t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);                       \
        blk[2] = rotl(blk[2], 1) ^ (t0 + t1 + l_key[4 * (i) + 10]);     \
        blk[3] = rotr(blk[3] ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]), 1); \
        t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);                       \
        blk[0] = rotl(blk[0], 1) ^ (t0 + t1 + l_key[4 * (i) +  8]);     \
        blk[1] = rotr(blk[1] ^ (t0 + 2 * t1 + l_key[4 * (i) +  9]), 1)

__device__ void twofish_decrypt(TwofishInstance *instance, const u4byte in_blk[4], u4byte out_blk[4])
{   u4byte  t0, t1, blk[4];

	u4byte *l_key = instance->l_key;
	u4byte *mk_tab = instance->mk_tab;

    blk[0] = LE32(in_blk[0]) ^ l_key[4];
    blk[1] = LE32(in_blk[1]) ^ l_key[5];
    blk[2] = LE32(in_blk[2]) ^ l_key[6];
    blk[3] = LE32(in_blk[3]) ^ l_key[7];

    i_rnd(7); i_rnd(6); i_rnd(5); i_rnd(4);
    i_rnd(3); i_rnd(2); i_rnd(1); i_rnd(0);

    out_blk[0] = LE32(blk[2] ^ l_key[0]);
    out_blk[1] = LE32(blk[3] ^ l_key[1]);
    out_blk[2] = LE32(blk[0] ^ l_key[2]);
    out_blk[3] = LE32(blk[1] ^ l_key[3]); 
};

#else // TC_MINIMIZE_CODE_SIZE

__device__ void twofish_decrypt(TwofishInstance *instance, const u4byte in_blk[4], u4byte out_blk[4])
{   u4byte  t0, t1, blk[4];

	u4byte *l_key = instance->l_key;
#ifdef TC_WINDOWS_BOOT_TWOFISH
	u4byte *mk_tab = instance->mk_tab;
#endif
	int i;

    blk[0] = LE32(in_blk[0]) ^ l_key[4];
    blk[1] = LE32(in_blk[1]) ^ l_key[5];
    blk[2] = LE32(in_blk[2]) ^ l_key[6];
    blk[3] = LE32(in_blk[3]) ^ l_key[7];

	for (i = 7; i >= 0; --i)
	{
		t1 = g1_fun(blk[1]); t0 = g0_fun(blk[0]);
		blk[2] = rotl(blk[2], 1) ^ (t0 + t1 + l_key[4 * (i) + 10]);
		blk[3] = rotr(blk[3] ^ (t0 + 2 * t1 + l_key[4 * (i) + 11]), 1);
		t1 = g1_fun(blk[3]); t0 = g0_fun(blk[2]);
		blk[0] = rotl(blk[0], 1) ^ (t0 + t1 + l_key[4 * (i) +  8]);
		blk[1] = rotr(blk[1] ^ (t0 + 2 * t1 + l_key[4 * (i) +  9]), 1);
	}

    out_blk[0] = LE32(blk[2] ^ l_key[0]);
    out_blk[1] = LE32(blk[3] ^ l_key[1]);
    out_blk[2] = LE32(blk[0] ^ l_key[2]);
    out_blk[3] = LE32(blk[1] ^ l_key[3]); 
};

#endif // TC_MINIMIZE_CODE_SIZE
/*
 * Copyright (C)  2011  Luca Vaccaro
 * Based on TrueCrypt, freely available at http://www.truecrypt.org/
 *
 * TrueCrack is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */
/*
 Legal Notice: Some portions of the source code contained in this file were
 derived from the source code of Encryption for the Masses 2.02a, which is
 Copyright (c) 1998-2000 Paul Le Roux and which is governed by the 'License
 Agreement for Encryption for the Masses'. Modifications and additions to
 the original source code (contained in this file) and all other portions
 of this file are Copyright (c) 2003-2010 TrueCrypt Developers Association
 and are governed by the TrueCrypt License 3.0 the full text of which is
 contained in the file License.txt included in TrueCrypt binary and source
 code distribution packages. */

#include "Tcdefs.h"
#include "Crypto.cuh"
#include "Xts.cuh"
#include "Crc.h"
#include "Common/Endian.h"
#include <string.h>
//#ifndef TC_WINDOWS_BOOT
//#include "EncryptionThreadPool.h"
//#endif
#include "Volumes.cuh"
#include "Twofish.cuh"


/*


// Cipher configuration
static Cipher Ciphers[] =
{
//								Block Size	Key Size	Key Schedule Size
//	  ID		Name			(Bytes)		(Bytes)		(Bytes)
	{ AES,		"AES",			16,			32,			AES_KS				},
	{ SERPENT,	"Serpent",		16,			32,			140*4				},
	{ TWOFISH,	"Twofish",		16,			32,			TWOFISH_KS			},
#ifndef TC_WINDOWS_BOOT
	{ BLOWFISH,	"Blowfish",		8,			56,			sizeof (BF_KEY)		},	// Deprecated/legacy
	{ CAST,		"CAST5",		8,			16,			sizeof (CAST_KEY)	},	// Deprecated/legacy
	{ TRIPLEDES,"Triple DES",	8,			8*3,		sizeof (TDES_KEY)	},	// Deprecated/legacy
#endif
	{ 0,		0,				0,			0,			0					}
};


// Encryption algorithm configuration
// The following modes have been deprecated (legacy): LRW, CBC, INNER_CBC, OUTER_CBC
static EncryptionAlgorithm EncryptionAlgorithms[] =
{
	//  Cipher(s)                     Modes						FormatEnabled

#ifndef TC_WINDOWS_BOOT

	{ { 0,						0 }, { 0, 0, 0, 0 },				0 },	// Must be all-zero
	{ { AES,					0 }, { XTS, LRW, CBC, 0 },			1 },
	{ { SERPENT,				0 }, { XTS, LRW, CBC, 0 },			1 },
	{ { TWOFISH,				0 }, { XTS, LRW, CBC, 0 },			1 },
	{ { TWOFISH, AES,			0 }, { XTS, LRW, OUTER_CBC, 0 },	1 },
	{ { SERPENT, TWOFISH, AES,	0 }, { XTS, LRW, OUTER_CBC, 0 },	1 },
	{ { AES, SERPENT,			0 }, { XTS, LRW, OUTER_CBC, 0 },	1 },
	{ { AES, TWOFISH, SERPENT,	0 }, { XTS, LRW, OUTER_CBC, 0 },	1 },
	{ { SERPENT, TWOFISH,		0 }, { XTS, LRW, OUTER_CBC, 0 },	1 },
	{ { BLOWFISH,				0 }, { LRW, CBC, 0, 0 },			0 },	// Deprecated/legacy
	{ { CAST,					0 }, { LRW, CBC, 0, 0 },			0 },	// Deprecated/legacy
	{ { TRIPLEDES,				0 }, { LRW, CBC, 0, 0 },			0 },	// Deprecated/legacy
	{ { BLOWFISH, AES,			0 }, { INNER_CBC, 0, 0, 0 },		0 },	// Deprecated/legacy
	{ { SERPENT, BLOWFISH, AES,	0 }, { INNER_CBC, 0, 0, 0 },		0 },	// Deprecated/legacy
	{ { 0,						0 }, { 0, 0, 0, 0 },				0 }		// Must be all-zero

#else // TC_WINDOWS_BOOT

	// Encryption algorithms available for boot drive encryption
	{ { 0,						0 }, { 0, 0 },		0 },	// Must be all-zero
	{ { AES,					0 }, { XTS, 0 },	1 },
	{ { SERPENT,				0 }, { XTS, 0 },	1 },
	{ { TWOFISH,				0 }, { XTS, 0 },	1 },
	{ { TWOFISH, AES,			0 }, { XTS, 0 },	1 },
	{ { SERPENT, TWOFISH, AES,	0 }, { XTS, 0 },	1 },
	{ { AES, SERPENT,			0 }, { XTS, 0 },	1 },
	{ { AES, TWOFISH, SERPENT,	0 }, { XTS, 0 },	1 },
	{ { SERPENT, TWOFISH,		0 }, { XTS, 0 },	1 },
	{ { 0,						0 }, { 0, 0 },		0 },	// Must be all-zero

#endif

};



// Hash algorithms
static Hash Hashes[] =
{	// ID			Name			Deprecated		System Encryption
	{ RIPEMD160,	"RIPEMD-160",	FALSE,			TRUE },
#ifndef TC_WINDOWS_BOOT
	{ SHA512,		"SHA-512",		FALSE,			FALSE },
	{ WHIRLPOOL,	"Whirlpool",	FALSE,			FALSE },
	{ SHA1,			"SHA-1",		TRUE,			FALSE },	// Deprecated/legacy
#endif
	{ 0, 0, 0 }
};
 */




/* Return values: 0 = success, ERR_CIPHER_INIT_FAILURE (fatal), ERR_CIPHER_INIT_WEAK_KEY (non-fatal) */
__device__ int cuCipherInit (int cipher, unsigned char *key, unsigned __int8 *ks)
{
    int retVal = ERR_SUCCESS;
	
    switch (cipher)
    {
		case AES:
#ifndef TC_WINDOWS_BOOT
			if (aes_encrypt_key256 (key, (aes_encrypt_ctx *) ks) != EXIT_SUCCESS)
				return ERR_CIPHER_INIT_FAILURE;
			
			if (aes_decrypt_key256 (key, (aes_decrypt_ctx *) (ks + sizeof(aes_encrypt_ctx))) != EXIT_SUCCESS)
				return ERR_CIPHER_INIT_FAILURE;
#else
			if (aes_set_key (key, (length_type) 32, (aes_context *) ks) != 0)
				return ERR_CIPHER_INIT_FAILURE;
#endif
			break;
			
		case SERPENT:
			serpent_set_key (key, 32 *8, ks);
			break;
			
		case TWOFISH:
			twofish_set_key ((TwofishInstance *)ks, (const u4byte *)key, 32 * 8);
			break;
		default:
			// Unknown/wrong cipher ID
			return ERR_CIPHER_INIT_FAILURE;
    }
	
    return retVal;
}

// Converts a 64-bit unsigned integer (passed as two 32-bit integers for compatibility with non-64-bit
// environments/platforms) into a little-endian 16-byte array.
__device__ void cuUint64ToLE16ByteArray (unsigned __int8 *byteBuf, unsigned __int32 highInt32, unsigned __int32 lowInt32)
{
    unsigned __int32 *bufPtr32 = (unsigned __int32 *) byteBuf;
	
    *bufPtr32++ = lowInt32;
    *bufPtr32++ = highInt32;
	
    // We're converting a 64-bit number into a little-endian 16-byte array so we can zero the last 8 bytes
    *bufPtr32++ = 0;
    *bufPtr32 = 0;
}

__device__ void cuEncipherBlock(int cipher, void *data, void *ks)
{
    switch (cipher)
    {
		case AES:
			// In 32-bit kernel mode, due to KeSaveFloatingPointState() overhead, AES instructions can be used only when processing the whole data unit.
			aes_encrypt ((const unsigned char*)data, (unsigned char*)data, (const aes_encrypt_ctx *)ks);
			break;
		case TWOFISH:
			twofish_encrypt ((TwofishInstance *)ks, (const unsigned int *)data, (unsigned int *)data);
			break;
		case SERPENT:
			serpent_encrypt ((const unsigned char *)data, (unsigned char *)data, (unsigned char *)ks);
			break;
		default:
			;//TC_THROW_FATAL_EXCEPTION;	// Unknown/wrong ID
    }
}

__device__ void cuDecipherBlock(int cipher, void *data, void *ks)
{
    switch (cipher)
    {
#ifndef TC_WINDOWS_BOOT
			
		case AES:
			aes_decrypt ((const unsigned char*)data, (unsigned char*)data, (const aes_decrypt_ctx *) ((char *) ks + sizeof(aes_decrypt_ctx)));
			break;
#else
		case AES:
			aes_decrypt ((unsigned char*)data, (unsigned char*)data, ((const aes_decrypt_ctx *))ks);
			break;
#endif
		case SERPENT:
			serpent_decrypt ((const unsigned char *)data, (unsigned char *)data, (unsigned char *)ks);
			break;
		case TWOFISH:
			twofish_decrypt ((TwofishInstance *)ks, (const unsigned int *)data, (unsigned int *)data);
			break;
		default:
			;//TC_THROW_FATAL_EXCEPTION;	// Unknown/wrong ID
    }
}


/*
 * Copyright (C)  2011  Luca Vaccaro
 * Based on TrueCrypt, freely available at http://www.truecrypt.org/
 *
 * TrueCrack is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */
#if BYTE_ORDER == BIG_ENDIAN
#error The TC_NO_COMPILER_INT64 version of the XTS code is not compatible with big-endian platforms
#endif
#include "Endian.h"

#if BYTE_ORDER == LITTLE_ENDIAN
#	define CUDA_BE16(x) cuda_MirrorBytes16(x)
#	define CUDA_BE32(x) cuda_MirrorBytes32(x)
#	define CUDA_BE64(x) cuda_MirrorBytes64(x)
#else
#	define CUDA_BE16(x) (x)
#	define CUDA_BE32(x) (x)
#	define CUDA_BE64(x) (x)
#endif

__device__ unsigned __int16 cuda_MirrorBytes16 (unsigned __int16 x)
{
	return (x << 8) | (x >> 8);
}


__device__ unsigned __int32 cuda_MirrorBytes32 (unsigned __int32 x)
{
	unsigned __int32 n = (unsigned __int8) x;
	n <<= 8; n |= (unsigned __int8) (x >> 8);
	n <<= 8; n |= (unsigned __int8) (x >> 16);
	return (n << 8) | (unsigned __int8) (x >> 24);
}

#define GetHeaderField16(header,offset) (CUDA_BE16(*(uint16*)(header+offset)))
#define GetHeaderField32(header,offset) (CUDA_BE32(*(uint32*)(header+offset)))

/* CRC polynomial 0x04c11db7 */
__constant__ unsigned __int32 cuda_crc_32_tab[]=
{
	0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
	0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
	0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
	0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
	0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
	0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
	0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
	0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
	0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
	0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
	0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
	0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
	0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
	0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
	0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
	0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
	0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
	0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
	0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
	0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
	0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
	0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
	0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
	0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
	0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
	0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
	0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
	0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
	0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
	0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
	0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
	0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

__device__ unsigned __int32 cuGetCrc32 (unsigned char *data, int length)
{
	unsigned __int32 CRC = 0xffffffff;
	
	while (length--)
	{
		CRC = (CRC >> 8) ^ cuda_crc_32_tab[ (CRC ^ *data++) & 0xFF ];
	}
	
	return CRC ^ 0xffffffff;
}

__device__ void cuda_memcpy (unsigned char* to , unsigned char* from, int length){
	int i;
	for (i=0;i<length;i++)
		to[i]=from[i];
}




// Encrypts or decrypts all blocks in the buffer in XTS mode. For descriptions of the input parameters,
// see the 64-bit version of EncryptBufferXTS().
__device__ static void cuEncryptDecryptBufferXTS32 (const unsigned __int8 *buffer,
													   TC_LARGEST_COMPILER_UINT length,
													   const UINT64_STRUCT *startDataUnitNo,
													   unsigned int startBlock,
													   unsigned __int8 *ks,
													   unsigned __int8 *ks2,
													   int cipher,
													   BOOL decryption)
{
	
	__align__(8) unsigned __int8 byteBufUnitNo [BYTES_PER_XTS_BLOCK];
	__align__(8) unsigned __int8 whiteningValue [BYTES_PER_XTS_BLOCK];
	__align__(8) unsigned __int8 finalCarry;
	unsigned __int32 *whiteningValuePtr32;
	unsigned __int32 *finalDwordWhiteningValuePtr;
	unsigned __int32 *bufPtr32;
	
	TC_LARGEST_COMPILER_UINT blockCount;
	UINT64_STRUCT dataUnitNo;
	unsigned int block;
	unsigned int endBlock;
	
	
	bufPtr32 = (unsigned __int32 *) buffer;
	whiteningValuePtr32 = (unsigned __int32 *) whiteningValue;
	finalDwordWhiteningValuePtr = whiteningValuePtr32 + sizeof (whiteningValue) / sizeof (*whiteningValuePtr32) - 1;
	
	
	// Store the 64-bit data unit number in a way compatible with non-64-bit environments/platforms
	dataUnitNo.HighPart = startDataUnitNo->HighPart;
	dataUnitNo.LowPart = startDataUnitNo->LowPart;
	
	blockCount = length / BYTES_PER_XTS_BLOCK;
	
	// Convert the 64-bit data unit number into a little-endian 16-byte array.
	// (Passed as two 32-bit integers for compatibility with non-64-bit environments/platforms.)
	cuUint64ToLE16ByteArray (byteBufUnitNo, dataUnitNo.HighPart, dataUnitNo.LowPart);
	
	// Generate whitening values for all blocks in the buffer
	while (blockCount > 0)
	{
		
		
		if (blockCount < BLOCKS_PER_XTS_DATA_UNIT)
			endBlock = startBlock + (unsigned int) blockCount;
		else
			endBlock = BLOCKS_PER_XTS_DATA_UNIT;
		
		
		// Encrypt the data unit number using the secondary key (in order to generate the first
		// whitening value for this data unit)
		cuUint64ToLE16ByteArray (byteBufUnitNo, dataUnitNo.HighPart, dataUnitNo.LowPart);
		cuda_memcpy (whiteningValue, byteBufUnitNo, BYTES_PER_XTS_BLOCK);
		cuEncipherBlock (cipher, whiteningValue, ks2);
		
		// Generate (and apply) subsequent whitening values for blocks in this data unit and
		// encrypt/decrypt all relevant blocks in this data unit
		for (block = 0; block < endBlock; block++)
		{
			if (block >= startBlock)
			{
				whiteningValuePtr32 = (unsigned __int32 *) whiteningValue;
				
				// Whitening
				*bufPtr32++ ^= *whiteningValuePtr32++;
				*bufPtr32++ ^= *whiteningValuePtr32++;
				*bufPtr32++ ^= *whiteningValuePtr32++;
				*bufPtr32 ^= *whiteningValuePtr32;
				
				bufPtr32 -= BYTES_PER_XTS_BLOCK / sizeof (*bufPtr32) - 1;
				
				// Actual encryption/decryption
				if (decryption)
					cuDecipherBlock (cipher, bufPtr32, ks);
				else
					cuEncipherBlock (cipher, bufPtr32, ks);
				
				whiteningValuePtr32 = (unsigned __int32 *) whiteningValue;
				
				// Whitening
				*bufPtr32++ ^= *whiteningValuePtr32++;
				*bufPtr32++ ^= *whiteningValuePtr32++;
				*bufPtr32++ ^= *whiteningValuePtr32++;
				*bufPtr32++ ^= *whiteningValuePtr32;
			}
			
			// Derive the next whitening value
			
			finalCarry = 0;
			
			for (whiteningValuePtr32 = finalDwordWhiteningValuePtr;
				 whiteningValuePtr32 >= (unsigned __int32 *) whiteningValue;
				 whiteningValuePtr32--)
			{
				if (*whiteningValuePtr32 & 0x80000000)	// If the following shift results in a carry
				{
					if (whiteningValuePtr32 != finalDwordWhiteningValuePtr)	// If not processing the highest double word
					{
						// A regular carry
						*(whiteningValuePtr32 + 1) |= 1;
					}
					else
					{
						// The highest byte shift will result in a carry
						finalCarry = 135;
					}
				}
				
				*whiteningValuePtr32 <<= 1;
			}
			
			whiteningValue[0] ^= finalCarry;
		}
		
		blockCount -= endBlock - startBlock;
		startBlock = 0;
		
		// Increase the data unit number by one
		if (!++dataUnitNo.LowPart)
		{
			dataUnitNo.HighPart++;
		}
		
		// Convert the 64-bit data unit number into a little-endian 16-byte array.
		cuUint64ToLE16ByteArray (byteBufUnitNo, dataUnitNo.HighPart, dataUnitNo.LowPart);
	}
	
	FAST_ERASE64 (whiteningValue, sizeof (whiteningValue));
}


// For descriptions of the input parameters, see the 64-bit version of EncryptBufferXTS().
__device__ void cuDecryptBufferXTS (unsigned __int8 *buffer,
									   TC_LARGEST_COMPILER_UINT length,
									   const UINT64_STRUCT *startDataUnitNo,
									   unsigned int startCipherBlockNo,
									   unsigned __int8 *ks,
									   unsigned __int8 *ks2,
									   int cipher)
{
	// Decrypt all ciphertext blocks in the buffer
	cuEncryptDecryptBufferXTS32 (buffer, length, startDataUnitNo, startCipherBlockNo, ks, ks2, cipher, TRUE);
}

__device__ void cuDecryptBuffer (unsigned __int8 *buf, TC_LARGEST_COMPILER_UINT len, PCRYPTO_INFO cryptoInfo)
{
	//unsigned __int8 *ks = cryptoInfo->ks;  //+ EAGetKeyScheduleSize (cryptoInfo->ea);
	//unsigned __int8 *ks2 = cryptoInfo->ks2;// + EAGetKeyScheduleSize (cryptoInfo->ea);
	UINT64_STRUCT dataUnitNo;
	//int cipher;
	
	// When encrypting/decrypting a buffer (typically a volume header) the sequential number
	// of the first XTS data unit in the buffer is always 0 and the start of the buffer is
	// always assumed to be aligned with the start of the data unit 0.
	dataUnitNo.LowPart = 0;
	dataUnitNo.HighPart = 0;
	
	//	for (cipher = EAGetLastCipher (cryptoInfo->ea);
	//		cipher != 0;
	//		cipher = EAGetPreviousCipher (cryptoInfo->ea, cipher))
	//	{
	//		ks -= CipherGetKeyScheduleSize (cipher);
	//		ks2 -= CipherGetKeyScheduleSize (cipher);
	cuDecryptBufferXTS (buf, len, &dataUnitNo, 0, cryptoInfo->ks, cryptoInfo->ks2, cryptoInfo->ea);
	//	}
}



__device__ int cuXts(int encryptionAlgorithm, unsigned char *encryptedHeader, unsigned char *headerKey, unsigned char *header) {
	
    PCRYPTO_INFO cryptoInfo;
    CRYPTO_INFO cryptoInfo_struct;
	
    uint16 headerVersion;
    int status = ERR_PARAMETER_INCORRECT;
    int primaryKeyOffset=0;
	int eaGetKeySize=32; 
	
    //int pkcs5PrfCount = LAST_PRF_ID - FIRST_PRF_ID + 1;
	
    cryptoInfo=&cryptoInfo_struct;    
    if (cryptoInfo == NULL)
        return ERR_OUT_OF_MEMORY;
    memset (cryptoInfo, 0, sizeof (CRYPTO_INFO));

    // Init objects related to the mode of operation
	// Support only XTS
    cryptoInfo->mode= XTS ;
	if (encryptionAlgorithm!=AES && encryptionAlgorithm!=SERPENT && encryptionAlgorithm!=TWOFISH)
		return UNDEFINED;
    cryptoInfo->ea=encryptionAlgorithm;
	
	// Primary key schedule
	cuda_memcpy (cryptoInfo->k2, headerKey + primaryKeyOffset, 64);
	status = cuCipherInit (cryptoInfo->ea, cryptoInfo->k2, cryptoInfo->ks);
    if (status != ERR_SUCCESS)
        return ERR_CIPHER_INIT;
        
    // Secondary key schedule
    cuda_memcpy (cryptoInfo->k2, headerKey + eaGetKeySize, eaGetKeySize);
	status = cuCipherInit (cryptoInfo->ea, cryptoInfo->k2, cryptoInfo->ks2);
    if (status != ERR_SUCCESS)
        return ERR_MODE_INIT;
    
 
    // Copy the header for decryption
    cuda_memcpy (header, encryptedHeader, 512*sizeof(unsigned char));
	
    // Try to decrypt header
    cuDecryptBuffer (header + HEADER_ENCRYPTED_DATA_OFFSET, HEADER_ENCRYPTED_DATA_SIZE, cryptoInfo);
	    
	// Magic 'TRUE'
	if (GetHeaderField32 (header, TC_HEADER_OFFSET_MAGIC) != 0x54525545)
		return ERR_MAGIC_TRUE;
	
	// Header version
	headerVersion = GetHeaderField16 (header, TC_HEADER_OFFSET_VERSION);
	if (headerVersion > VOLUME_HEADER_VERSION) {
		return ERR_VERSION_REQUIRED;
	}
	
	// Check CRC of the header fields
	if (headerVersion >= 4
		&& GetHeaderField32 (header, TC_HEADER_OFFSET_HEADER_CRC) != cuGetCrc32 (header + TC_HEADER_OFFSET_MAGIC, TC_HEADER_OFFSET_HEADER_CRC - TC_HEADER_OFFSET_MAGIC))
		//printf("Unsuccessful\n");
		return ERR_CRC_HEADER_FIELDS;
	// Required program version
	//cryptoInfo->RequiredProgramVersion = GetHeaderField16 (header, TC_HEADER_OFFSET_REQUIRED_VERSION);
	//cryptoInfo->LegacyVolume = cryptoInfo->RequiredProgramVersion < 0x600;
	
	// Check CRC of the key set
	if (GetHeaderField32 (header, TC_HEADER_OFFSET_KEY_AREA_CRC) != cuGetCrc32 (header + HEADER_MASTER_KEYDATA_OFFSET, MASTER_KEYDATA_SIZE))
		return ERR_CRC_KEY_SET;

    return SUCCESS;
}



/*
 * Copyright (C)  2011  Luca Vaccaro
 * Based on TrueCrypt, freely available at http://www.truecrypt.org/
 *
 * TrueCrack is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */
// RIPEMD-160 written and placed in the public domain by Wei Dai

/*
 * This code implements the MD4 message-digest algorithm.
 * The algorithm is due to Ron Rivest.  This code was
 * written by Colin Plumb in 1993, no copyright is claimed.
 * This code is in the public domain; do with it what you wish.
 */

/* Adapted for TrueCrypt */

#include <memory.h>
#include "Tcdefs.h"
#include "Endian.h"
#include "Rmd160.alternative.cuh"

#define F(x, y, z)    (x ^ y ^ z) 
#define G(x, y, z)    (z ^ (x & (y^z)))
#define H(x, y, z)    (z ^ (x | ~y))
#define I(x, y, z)    (y ^ (z & (x^y)))
#define J(x, y, z)    (x ^ (y | ~z))

#define PUT_64BIT_LE(cp, value) do {                                    \
	(cp)[7] = (byte) ((value) >> 56);                                        \
	(cp)[6] = (byte) ((value) >> 48);                                        \
	(cp)[5] = (byte) ((value) >> 40);                                        \
	(cp)[4] = (byte) ((value) >> 32);                                        \
	(cp)[3] = (byte) ((value) >> 24);                                        \
	(cp)[2] = (byte) ((value) >> 16);                                        \
	(cp)[1] = (byte) ((value) >> 8);                                         \
	(cp)[0] = (byte) (value); } while (0)

#define PUT_32BIT_LE(cp, value) do {                                    \
	(cp)[3] = (byte) ((value) >> 24);                                        \
	(cp)[2] = (byte) ((value) >> 16);                                        \
	(cp)[1] = (byte) ((value) >> 8);                                         \
	(cp)[0] = (byte) (value); } while (0)

#define word32 unsigned __int32

#define k0 0
#define k1 0x5a827999UL
#define k2 0x6ed9eba1UL
#define k3 0x8f1bbcdcUL
#define k4 0xa953fd4eUL
#define k5 0x50a28be6UL
#define k6 0x5c4dd124UL
#define k7 0x6d703ef3UL
#define k8 0x7a6d76e9UL
#define k9 0

#define  rrotlFixed( x, y) (word32)((x<<y) | (x>>(sizeof(word32)*8-y)))
  
//__device__ word32 rrotlFixed (word32 x, unsigned int y)
//{ 
//	return (word32)((x<<y) | (x>>(sizeof(word32)*8-y)));
//}

#define Subround(f, a, b, c, d, e, x, s, k)        \
	a += f(b, c, d) + x + k;\
	a = rrotlFixed((word32)a, s) + e;\
	c = rrotlFixed((word32)c, 10U)


/*
static byte PADDING[64]= {
	0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
*/

__device__ void cuda_RMD160 (RMD160_CTX *ctx, const unsigned char *input1, unsigned __int32 lenArg1, const unsigned char *input2, unsigned __int32 lenArg2, unsigned char *digest){
	//global variable of the subset of functions
	uint32 padlen;
	byte count[8];
	byte PADDING[64];
	unsigned int update2_flags;
	
	if (input2==NULL || lenArg2==0)
	  update2_flags=FALSE;
	else
	  update2_flags=TRUE;
	
	// INCLUDE: void RMD160Init (RMD160_CTX *ctx)
	{
	int i;
	ctx->count = 0;
	ctx->state[0] = 0x67452301;
	ctx->state[1] = 0xefcdab89;
	ctx->state[2] = 0x98badcfe;
	ctx->state[3] = 0x10325476;
	ctx->state[4] = 0xc3d2e1f0;

	for (i=0;i<64;i++)
		PADDING[i]=0;
	PADDING[0] = 0x80;
	}
	// ENDINCLUDE: void RMD160Init (RMD160_CTX *ctx)
	
	
	// INCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input1, unsigned __int32 lenArg1)
	{
#ifndef TC_WINDOWS_BOOT
	uint32 len = lenArg1, have, need;
#else
	uint16 len = (uint16) lenArg1, have, need;
#endif
	/* Check how many bytes we already have and how many more we need. */
	have = ((ctx->count >> 3) & (RIPEMD160_BLOCK_LENGTH - 1));
	need = RIPEMD160_BLOCK_LENGTH - have;
	/* Update bitcount */
	ctx->count += len << 3;
	if (len >= need) {
		if (have != 0) {
			memcpy (ctx->buffer + have, input1, (size_t) need);
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) ctx->buffer);
			input1 += need;
			len -= need;
			have = 0;
		}
		// Process data in RIPEMD160_BLOCK_LENGTH-byte chunks. 
		while (len >= RIPEMD160_BLOCK_LENGTH) {
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) input1);
			input1 += RIPEMD160_BLOCK_LENGTH;
			len -= RIPEMD160_BLOCK_LENGTH;
		}	  
	}
	/* Handle any remaining bytes of data. */
	if (len != 0)
		memcpy (ctx->buffer + have, input1, (size_t) len);
	}
	// ENDINCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input1, unsigned __int32 lenArg1)
	
	if (update2_flags==TRUE)
	// INCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input2, unsigned __int32 lenArg2)
	{
#ifndef TC_WINDOWS_BOOT
	uint32 len = lenArg2, have, need;
#else
	uint16 len = (uint16) lenArg2, have, need;
#endif
	/* Check how many bytes we already have and how many more we need. */
	have = ((ctx->count >> 3) & (RIPEMD160_BLOCK_LENGTH - 1));
	need = RIPEMD160_BLOCK_LENGTH - have;
	/* Update bitcount */
	ctx->count += len << 3;
	if (len >= need) {
		if (have != 0) {
			memcpy (ctx->buffer + have, input2, (size_t) need);
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) ctx->buffer);
			input2 += need;
			len -= need;
			have = 0;
		}
		// Process data in RIPEMD160_BLOCK_LENGTH-byte chunks. 
		while (len >= RIPEMD160_BLOCK_LENGTH) {
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) input2);
			input2 += RIPEMD160_BLOCK_LENGTH;
			len -= RIPEMD160_BLOCK_LENGTH;
		}	  
	}
	/* Handle any remaining bytes of data. */
	if (len != 0)
		memcpy (ctx->buffer + have, input2, (size_t) len);
	}// ENDINCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input2, unsigned __int32 lenArg2)
	
	
	// INCLUDE: void RMD160Pad(RMD160_CTX *ctx)
	{
	/* Convert count to 8 bytes in little endian order. */
#ifndef TC_WINDOWS_BOOT
	PUT_64BIT_LE(count, ctx->count);
#else
	*(uint32 *) (count + 4) = 0;
	*(uint16 *) (count + 2) = 0;
	*(uint16 *) (count + 0) = ctx->count;
#endif
	/* Pad out to 56 mod 64. */
	padlen = RIPEMD160_BLOCK_LENGTH -
		(uint32)((ctx->count >> 3) & (RIPEMD160_BLOCK_LENGTH - 1));
	if (padlen < 1 + 8)
		padlen += RIPEMD160_BLOCK_LENGTH;
	}
	// ENDINCLUDE: void RMD160Pad(RMD160_CTX *ctx)
	
		
	
	
	// INCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input3, unsigned __int32 lenArg3) 
	// Call RMD160Update(ctx, PADDING, padlen - 8);
	{
	  unsigned char *input3;
	  input3=(unsigned char *)PADDING;
	  unsigned __int32 lenArg3=padlen-8;
#ifndef TC_WINDOWS_BOOT
	uint32 len = lenArg3, have, need;
#else
	uint16 len = (uint16) lenArg3, have, need;
#endif
	/* Check how many bytes we already have and how many more we need. */
	have = ((ctx->count >> 3) & (RIPEMD160_BLOCK_LENGTH - 1));
	need = RIPEMD160_BLOCK_LENGTH - have;
	/* Update bitcount */
	ctx->count += len << 3;
	if (len >= need) {
		if (have != 0) {
			memcpy (ctx->buffer + have, input3, (size_t) need);
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) ctx->buffer);
			input3 += need;
			len -= need;
			have = 0;
		}
		// Process data in RIPEMD160_BLOCK_LENGTH-byte chunks. 
		while (len >= RIPEMD160_BLOCK_LENGTH) {
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) input3);
			input3 += RIPEMD160_BLOCK_LENGTH;
			len -= RIPEMD160_BLOCK_LENGTH;
		}	  
	}
	/* Handle any remaining bytes of data. */
	if (len != 0)
		memcpy (ctx->buffer + have, input3, (size_t) len);
	}// ENDINCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input3, unsigned __int32 lenArg3)
	

	
	// INCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input4, unsigned __int32 lenArg4) 
	// Call RMD160Update(ctx, count, 8);
	{
	  unsigned char *input4;
	  input4=(unsigned char *)count;
	  unsigned __int32 lenArg4=8;
#ifndef TC_WINDOWS_BOOT
	uint32 len = lenArg4, have, need;
#else
	uint16 len = (uint16) lenArg4, have, need;
#endif
	/* Check how many bytes we already have and how many more we need. */
	have = ((ctx->count >> 3) & (RIPEMD160_BLOCK_LENGTH - 1));
	need = RIPEMD160_BLOCK_LENGTH - have;
	/* Update bitcount */
	ctx->count += len << 3;
	if (len >= need) {
		if (have != 0) {
			memcpy (ctx->buffer + have, input4, (size_t) need);
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) ctx->buffer);
			input4 += need;
			len -= need;
			have = 0;
		}
		// Process data in RIPEMD160_BLOCK_LENGTH-byte chunks. 
		while (len >= RIPEMD160_BLOCK_LENGTH) {
			cuda_RMD160Transform ((uint32 *) ctx->state, (const uint32 *) input4);
			input4 += RIPEMD160_BLOCK_LENGTH;
			len -= RIPEMD160_BLOCK_LENGTH;
		}	  
	}
	/* Handle any remaining bytes of data. */
	if (len != 0)
		memcpy (ctx->buffer + have, input4, (size_t) len);
	}// ENDINCLUDE: void RMD160Update (RMD160_CTX *ctx, const unsigned char *input4, unsigned __int32 lenArg4) 
		
		
		
	
	// INCLUDE: RMD160Final(unsigned char *digest, RMD160_CTX *ctx)	
	int i;
	if (digest) {
		for (i = 0; i < 5; i++)
			PUT_32BIT_LE(digest + i * 4, ctx->state[i]);
		memset (ctx, 0, sizeof(*ctx));
	}	
	// ENDINCLUDE: RMD160Final(unsigned char *digest, RMD160_CTX *ctx)
}



__device__ void cuda_RMD160Transform (unsigned __int32 *digest, const unsigned __int32 *data)
{
 
#if BYTE_ORDER == LITTLE_ENDIAN
	const unsigned __int32 *X = data;
#else
	unsigned __int32 X[16];
	int i;
#endif

	unsigned __int32 a1, b1, c1, d1, e1, a2, b2, c2, d2, e2;
	
	a1 = a2 = digest[0];
	b1 = b2 = digest[1];
	c1 = c2 = digest[2];
	d1 = d2 = digest[3];
	e1 = e2 = digest[4];
	

#if BYTE_ORDER == BIG_ENDIAN
	for (i = 0; i < 16; i++)
	{
		X[i] = LE32 (data[i]);
	}
#endif

	Subround(F, a1, b1, c1, d1, e1, X[ 0], 11, k0);
	Subround(F, e1, a1, b1, c1, d1, X[ 1], 14, k0);
	Subround(F, d1, e1, a1, b1, c1, X[ 2], 15, k0);
	Subround(F, c1, d1, e1, a1, b1, X[ 3], 12, k0);
	Subround(F, b1, c1, d1, e1, a1, X[ 4],  5, k0);
	Subround(F, a1, b1, c1, d1, e1, X[ 5],  8, k0);
	Subround(F, e1, a1, b1, c1, d1, X[ 6],  7, k0);
	Subround(F, d1, e1, a1, b1, c1, X[ 7],  9, k0);
	Subround(F, c1, d1, e1, a1, b1, X[ 8], 11, k0);
	Subround(F, b1, c1, d1, e1, a1, X[ 9], 13, k0);
	Subround(F, a1, b1, c1, d1, e1, X[10], 14, k0);
	Subround(F, e1, a1, b1, c1, d1, X[11], 15, k0);
	Subround(F, d1, e1, a1, b1, c1, X[12],  6, k0);
	Subround(F, c1, d1, e1, a1, b1, X[13],  7, k0);
	Subround(F, b1, c1, d1, e1, a1, X[14],  9, k0);
	Subround(F, a1, b1, c1, d1, e1, X[15],  8, k0);

	Subround(G, e1, a1, b1, c1, d1, X[ 7],  7, k1);
	Subround(G, d1, e1, a1, b1, c1, X[ 4],  6, k1);
	Subround(G, c1, d1, e1, a1, b1, X[13],  8, k1);
	Subround(G, b1, c1, d1, e1, a1, X[ 1], 13, k1);
	Subround(G, a1, b1, c1, d1, e1, X[10], 11, k1);
	Subround(G, e1, a1, b1, c1, d1, X[ 6],  9, k1);
	Subround(G, d1, e1, a1, b1, c1, X[15],  7, k1);
	Subround(G, c1, d1, e1, a1, b1, X[ 3], 15, k1);
	Subround(G, b1, c1, d1, e1, a1, X[12],  7, k1);
	Subround(G, a1, b1, c1, d1, e1, X[ 0], 12, k1);
	Subround(G, e1, a1, b1, c1, d1, X[ 9], 15, k1);
	Subround(G, d1, e1, a1, b1, c1, X[ 5],  9, k1);
	Subround(G, c1, d1, e1, a1, b1, X[ 2], 11, k1);
	Subround(G, b1, c1, d1, e1, a1, X[14],  7, k1);
	Subround(G, a1, b1, c1, d1, e1, X[11], 13, k1);
	Subround(G, e1, a1, b1, c1, d1, X[ 8], 12, k1);

	Subround(H, d1, e1, a1, b1, c1, X[ 3], 11, k2);
	Subround(H, c1, d1, e1, a1, b1, X[10], 13, k2);
	Subround(H, b1, c1, d1, e1, a1, X[14],  6, k2);
	Subround(H, a1, b1, c1, d1, e1, X[ 4],  7, k2);
	Subround(H, e1, a1, b1, c1, d1, X[ 9], 14, k2);
	Subround(H, d1, e1, a1, b1, c1, X[15],  9, k2);
	Subround(H, c1, d1, e1, a1, b1, X[ 8], 13, k2);
	Subround(H, b1, c1, d1, e1, a1, X[ 1], 15, k2);
	Subround(H, a1, b1, c1, d1, e1, X[ 2], 14, k2);
	Subround(H, e1, a1, b1, c1, d1, X[ 7],  8, k2);
	Subround(H, d1, e1, a1, b1, c1, X[ 0], 13, k2);
	Subround(H, c1, d1, e1, a1, b1, X[ 6],  6, k2);
	Subround(H, b1, c1, d1, e1, a1, X[13],  5, k2);
	Subround(H, a1, b1, c1, d1, e1, X[11], 12, k2);
	Subround(H, e1, a1, b1, c1, d1, X[ 5],  7, k2);
	Subround(H, d1, e1, a1, b1, c1, X[12],  5, k2);

	Subround(I, c1, d1, e1, a1, b1, X[ 1], 11, k3);
	Subround(I, b1, c1, d1, e1, a1, X[ 9], 12, k3);
	Subround(I, a1, b1, c1, d1, e1, X[11], 14, k3);
	Subround(I, e1, a1, b1, c1, d1, X[10], 15, k3);
	Subround(I, d1, e1, a1, b1, c1, X[ 0], 14, k3);
	Subround(I, c1, d1, e1, a1, b1, X[ 8], 15, k3);
	Subround(I, b1, c1, d1, e1, a1, X[12],  9, k3);
	Subround(I, a1, b1, c1, d1, e1, X[ 4],  8, k3);
	Subround(I, e1, a1, b1, c1, d1, X[13],  9, k3);
	Subround(I, d1, e1, a1, b1, c1, X[ 3], 14, k3);
	Subround(I, c1, d1, e1, a1, b1, X[ 7],  5, k3);
	Subround(I, b1, c1, d1, e1, a1, X[15],  6, k3);
	Subround(I, a1, b1, c1, d1, e1, X[14],  8, k3);
	Subround(I, e1, a1, b1, c1, d1, X[ 5],  6, k3);
	Subround(I, d1, e1, a1, b1, c1, X[ 6],  5, k3);
	Subround(I, c1, d1, e1, a1, b1, X[ 2], 12, k3);

	Subround(J, b1, c1, d1, e1, a1, X[ 4],  9, k4);
	Subround(J, a1, b1, c1, d1, e1, X[ 0], 15, k4);
	Subround(J, e1, a1, b1, c1, d1, X[ 5],  5, k4);
	Subround(J, d1, e1, a1, b1, c1, X[ 9], 11, k4);
	Subround(J, c1, d1, e1, a1, b1, X[ 7],  6, k4);
	Subround(J, b1, c1, d1, e1, a1, X[12],  8, k4);
	Subround(J, a1, b1, c1, d1, e1, X[ 2], 13, k4);
	Subround(J, e1, a1, b1, c1, d1, X[10], 12, k4);
	Subround(J, d1, e1, a1, b1, c1, X[14],  5, k4);
	Subround(J, c1, d1, e1, a1, b1, X[ 1], 12, k4);
	Subround(J, b1, c1, d1, e1, a1, X[ 3], 13, k4);
	Subround(J, a1, b1, c1, d1, e1, X[ 8], 14, k4);
	Subround(J, e1, a1, b1, c1, d1, X[11], 11, k4);
	Subround(J, d1, e1, a1, b1, c1, X[ 6],  8, k4);
	Subround(J, c1, d1, e1, a1, b1, X[15],  5, k4);
	Subround(J, b1, c1, d1, e1, a1, X[13],  6, k4);

	Subround(J, a2, b2, c2, d2, e2, X[ 5],  8, k5);
	Subround(J, e2, a2, b2, c2, d2, X[14],  9, k5);
	Subround(J, d2, e2, a2, b2, c2, X[ 7],  9, k5);
	Subround(J, c2, d2, e2, a2, b2, X[ 0], 11, k5);
	Subround(J, b2, c2, d2, e2, a2, X[ 9], 13, k5);
	Subround(J, a2, b2, c2, d2, e2, X[ 2], 15, k5);
	Subround(J, e2, a2, b2, c2, d2, X[11], 15, k5);
	Subround(J, d2, e2, a2, b2, c2, X[ 4],  5, k5);
	Subround(J, c2, d2, e2, a2, b2, X[13],  7, k5);
	Subround(J, b2, c2, d2, e2, a2, X[ 6],  7, k5);
	Subround(J, a2, b2, c2, d2, e2, X[15],  8, k5);
	Subround(J, e2, a2, b2, c2, d2, X[ 8], 11, k5);
	Subround(J, d2, e2, a2, b2, c2, X[ 1], 14, k5);
	Subround(J, c2, d2, e2, a2, b2, X[10], 14, k5);
	Subround(J, b2, c2, d2, e2, a2, X[ 3], 12, k5);
	Subround(J, a2, b2, c2, d2, e2, X[12],  6, k5);

	Subround(I, e2, a2, b2, c2, d2, X[ 6],  9, k6); 
	Subround(I, d2, e2, a2, b2, c2, X[11], 13, k6);
	Subround(I, c2, d2, e2, a2, b2, X[ 3], 15, k6);
	Subround(I, b2, c2, d2, e2, a2, X[ 7],  7, k6);
	Subround(I, a2, b2, c2, d2, e2, X[ 0], 12, k6);
	Subround(I, e2, a2, b2, c2, d2, X[13],  8, k6);
	Subround(I, d2, e2, a2, b2, c2, X[ 5],  9, k6);
	Subround(I, c2, d2, e2, a2, b2, X[10], 11, k6);
	Subround(I, b2, c2, d2, e2, a2, X[14],  7, k6);
	Subround(I, a2, b2, c2, d2, e2, X[15],  7, k6);
	Subround(I, e2, a2, b2, c2, d2, X[ 8], 12, k6);
	Subround(I, d2, e2, a2, b2, c2, X[12],  7, k6);
	Subround(I, c2, d2, e2, a2, b2, X[ 4],  6, k6);
	Subround(I, b2, c2, d2, e2, a2, X[ 9], 15, k6);
	Subround(I, a2, b2, c2, d2, e2, X[ 1], 13, k6);
	Subround(I, e2, a2, b2, c2, d2, X[ 2], 11, k6);

	Subround(H, d2, e2, a2, b2, c2, X[15],  9, k7);
	Subround(H, c2, d2, e2, a2, b2, X[ 5],  7, k7);
	Subround(H, b2, c2, d2, e2, a2, X[ 1], 15, k7);
	Subround(H, a2, b2, c2, d2, e2, X[ 3], 11, k7);
	Subround(H, e2, a2, b2, c2, d2, X[ 7],  8, k7);
	Subround(H, d2, e2, a2, b2, c2, X[14],  6, k7);
	Subround(H, c2, d2, e2, a2, b2, X[ 6],  6, k7);
	Subround(H, b2, c2, d2, e2, a2, X[ 9], 14, k7);
	Subround(H, a2, b2, c2, d2, e2, X[11], 12, k7);
	Subround(H, e2, a2, b2, c2, d2, X[ 8], 13, k7);
	Subround(H, d2, e2, a2, b2, c2, X[12],  5, k7);
	Subround(H, c2, d2, e2, a2, b2, X[ 2], 14, k7);
	Subround(H, b2, c2, d2, e2, a2, X[10], 13, k7);
	Subround(H, a2, b2, c2, d2, e2, X[ 0], 13, k7);
	Subround(H, e2, a2, b2, c2, d2, X[ 4],  7, k7);
	Subround(H, d2, e2, a2, b2, c2, X[13],  5, k7);

	Subround(G, c2, d2, e2, a2, b2, X[ 8], 15, k8);
	Subround(G, b2, c2, d2, e2, a2, X[ 6],  5, k8);
	Subround(G, a2, b2, c2, d2, e2, X[ 4],  8, k8);
	Subround(G, e2, a2, b2, c2, d2, X[ 1], 11, k8);
	Subround(G, d2, e2, a2, b2, c2, X[ 3], 14, k8);
	Subround(G, c2, d2, e2, a2, b2, X[11], 14, k8);
	Subround(G, b2, c2, d2, e2, a2, X[15],  6, k8);
	Subround(G, a2, b2, c2, d2, e2, X[ 0], 14, k8);
	Subround(G, e2, a2, b2, c2, d2, X[ 5],  6, k8);
	Subround(G, d2, e2, a2, b2, c2, X[12],  9, k8);
	Subround(G, c2, d2, e2, a2, b2, X[ 2], 12, k8);
	Subround(G, b2, c2, d2, e2, a2, X[13],  9, k8);
	Subround(G, a2, b2, c2, d2, e2, X[ 9], 12, k8);
	Subround(G, e2, a2, b2, c2, d2, X[ 7],  5, k8);
	Subround(G, d2, e2, a2, b2, c2, X[10], 15, k8);
	Subround(G, c2, d2, e2, a2, b2, X[14],  8, k8);

	Subround(F, b2, c2, d2, e2, a2, X[12],  8, k9);
	Subround(F, a2, b2, c2, d2, e2, X[15],  5, k9);
	Subround(F, e2, a2, b2, c2, d2, X[10], 12, k9);
	Subround(F, d2, e2, a2, b2, c2, X[ 4],  9, k9);
	Subround(F, c2, d2, e2, a2, b2, X[ 1], 12, k9);
	Subround(F, b2, c2, d2, e2, a2, X[ 5],  5, k9);
	Subround(F, a2, b2, c2, d2, e2, X[ 8], 14, k9);
	Subround(F, e2, a2, b2, c2, d2, X[ 7],  6, k9);
	Subround(F, d2, e2, a2, b2, c2, X[ 6],  8, k9);
	Subround(F, c2, d2, e2, a2, b2, X[ 2], 13, k9);
	Subround(F, b2, c2, d2, e2, a2, X[13],  6, k9);
	Subround(F, a2, b2, c2, d2, e2, X[14],  5, k9);
	Subround(F, e2, a2, b2, c2, d2, X[ 0], 15, k9);
	Subround(F, d2, e2, a2, b2, c2, X[ 3], 13, k9);
	Subround(F, c2, d2, e2, a2, b2, X[ 9], 11, k9);
	Subround(F, b2, c2, d2, e2, a2, X[11], 11, k9);

	c1        = digest[1] + c1 + d2;
	digest[1] = digest[2] + d1 + e2;
	digest[2] = digest[3] + e1 + a2;
	digest[3] = digest[4] + a1 + b2;
	digest[4] = digest[0] + b1 + c2;
	digest[0] = c1;
	
}
/*
 ---------------------------------------------------------------------------
 Copyright (c) 2002, Dr Brian Gladman, Worcester, UK.   All rights reserved.

 LICENSE TERMS

 The free distribution and use of this software is allowed (with or without
 changes) provided that:

  1. source code distributions include the above copyright notice, this
     list of conditions and the following disclaimer;

  2. binary distributions include the above copyright notice, this list
     of conditions and the following disclaimer in their documentation;

  3. the name of the copyright holder is not used to endorse products
     built using this software without specific written permission.

 DISCLAIMER

 This software is provided 'as is' with no explicit or implied warranties
 in respect of its properties, including, but not limited to, correctness
 and/or fitness for purpose.
 ---------------------------------------------------------------------------
 Issue Date: 01/08/2005

 This is a byte oriented version of SHA2 that operates on arrays of bytes
 stored in memory. This code implements sha256, sha384 and sha512 but the
 latter two functions rely on efficient 64-bit integer operations that
 may not be very efficient on 32-bit machines

 The sha256 functions use a type 'sha256_ctx' to hold details of the
 current hash state and uses the following three calls:

       void sha256_begin(sha256_ctx ctx[1])
       void sha256_hash(const unsigned char data[],
                            unsigned long len, sha256_ctx ctx[1])
       void sha_end1(unsigned char hval[], sha256_ctx ctx[1])

 The first subroutine initialises a hash computation by setting up the
 context in the sha256_ctx context. The second subroutine hashes 8-bit
 bytes from array data[] into the hash state withinh sha256_ctx context,
 the number of bytes to be hashed being given by the the unsigned long
 integer len.  The third subroutine completes the hash calculation and
 places the resulting digest value in the array of 8-bit bytes hval[].

 The sha384 and sha512 functions are similar and use the interfaces:

       void sha384_begin(sha384_ctx ctx[1]);
       void sha384_hash(const unsigned char data[],
                            unsigned long len, sha384_ctx ctx[1]);
       void sha384_end(unsigned char hval[], sha384_ctx ctx[1]);

       void sha512_begin(sha512_ctx ctx[1]);
       void sha512_hash(const unsigned char data[],
                            unsigned long len, sha512_ctx ctx[1]);
       void sha512_end(unsigned char hval[], sha512_ctx ctx[1]);

 In addition there is a function sha2 that can be used to call all these
 functions using a call with a hash length parameter as follows:

       int sha2_begin(unsigned long len, sha2_ctx ctx[1]);
       void sha2_hash(const unsigned char data[],
                            unsigned long len, sha2_ctx ctx[1]);
       void sha2_end(unsigned char hval[], sha2_ctx ctx[1]);

 My thanks to Erik Andersen <andersen@codepoet.org> for testing this code
 on big-endian systems and for his assistance with corrections
*/

#include "Common/Endian.h"
#ifndef PLATFORM_BYTE_ORDER
#define PLATFORM_BYTE_ORDER BYTE_ORDER
#endif
#ifndef IS_LITTLE_ENDIAN
#define IS_LITTLE_ENDIAN LITTLE_ENDIAN
#endif

#if 0
#define UNROLL_SHA2     /* for SHA2 loop unroll     */
#endif

#include <string.h>     /* for memcpy() etc.        */

#include "Sha2.cuh"

#if defined(__cplusplus)
extern "C"
{
#endif

#if defined( _MSC_VER ) && ( _MSC_VER > 800 )
#pragma intrinsic(memcpy)
#endif

#if 0 && defined(_MSC_VER)
#define rotl32 _lrotl
#define rotr32 _lrotr
#else
#define rotl32(x,n)   (((x) << n) | ((x) >> (32 - n)))
#define rotr32(x,n)   (((x) >> n) | ((x) << (32 - n)))
#endif

#if !defined(bswap_32)
#define bswap_32(x) ((rotr32((x), 24) & 0x00ff00ff) | (rotr32((x), 8) & 0xff00ff00))
#endif

#if (PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN)
#define SWAP_BYTES
#else
#undef  SWAP_BYTES
#endif

#if 0

#define ch(x,y,z)       (((x) & (y)) ^ (~(x) & (z)))
#define maj(x,y,z)      (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#else   /* Thanks to Rich Schroeppel and Colin Plumb for the following      */

#define ch(x,y,z)       ((z) ^ ((x) & ((y) ^ (z))))
#define maj(x,y,z)      (((x) & (y)) | ((z) & ((x) ^ (y))))

#endif

/* round transforms for SHA256 and SHA512 compression functions */

#define vf(n,i) v[(n - i) & 7]

#define hf(i) (p[i & 15] += \
    g_1(p[(i + 14) & 15]) + p[(i + 9) & 15] + g_0(p[(i + 1) & 15]))

#define v_cycle(i,j)                                \
    vf(7,i) += (j ? hf(i) : p[i]) + k_0[i+j]        \
    + s_1(vf(4,i)) + ch(vf(4,i),vf(5,i),vf(6,i));   \
    vf(3,i) += vf(7,i);                             \
    vf(7,i) += s_0(vf(0,i))+ maj(vf(0,i),vf(1,i),vf(2,i))

#if defined(SHA_224) || defined(SHA_256)

#define SHA256_MASK (SHA256_BLOCK_SIZE - 1)

#if defined(SWAP_BYTES)
#define bsw_32(p,n) \
    { int _i = (n); while(_i--) ((uint_32t*)p)[_i] = bswap_32(((uint_32t*)p)[_i]); }
#else
#define bsw_32(p,n)
#endif

#define s_0(x)  (rotr32((x),  2) ^ rotr32((x), 13) ^ rotr32((x), 22))
#define s_1(x)  (rotr32((x),  6) ^ rotr32((x), 11) ^ rotr32((x), 25))
#define g_0(x)  (rotr32((x),  7) ^ rotr32((x), 18) ^ ((x) >>  3))
#define g_1(x)  (rotr32((x), 17) ^ rotr32((x), 19) ^ ((x) >> 10))
#define k_0     k256

/* rotated SHA256 round definition. Rather than swapping variables as in    */
/* FIPS-180, different variables are 'rotated' on each round, returning     */
/* to their starting positions every eight rounds                           */

#define qq(n)  v##n

#define one_cycle(a,b,c,d,e,f,g,h,k,w)  \
    qq(h) += s_1(qq(e)) + ch(qq(e), qq(f), qq(g)) + k + w; \
    qq(d) += qq(h); qq(h) += s_0(qq(a)) + maj(qq(a), qq(b), qq(c))

/* SHA256 mixing data   */

__constant__ const uint_32t k256[64] =
{   0x428a2f98ul, 0x71374491ul, 0xb5c0fbcful, 0xe9b5dba5ul,
    0x3956c25bul, 0x59f111f1ul, 0x923f82a4ul, 0xab1c5ed5ul,
    0xd807aa98ul, 0x12835b01ul, 0x243185beul, 0x550c7dc3ul,
    0x72be5d74ul, 0x80deb1feul, 0x9bdc06a7ul, 0xc19bf174ul,
    0xe49b69c1ul, 0xefbe4786ul, 0x0fc19dc6ul, 0x240ca1ccul,
    0x2de92c6ful, 0x4a7484aaul, 0x5cb0a9dcul, 0x76f988daul,
    0x983e5152ul, 0xa831c66dul, 0xb00327c8ul, 0xbf597fc7ul,
    0xc6e00bf3ul, 0xd5a79147ul, 0x06ca6351ul, 0x14292967ul,
    0x27b70a85ul, 0x2e1b2138ul, 0x4d2c6dfcul, 0x53380d13ul,
    0x650a7354ul, 0x766a0abbul, 0x81c2c92eul, 0x92722c85ul,
    0xa2bfe8a1ul, 0xa81a664bul, 0xc24b8b70ul, 0xc76c51a3ul,
    0xd192e819ul, 0xd6990624ul, 0xf40e3585ul, 0x106aa070ul,
    0x19a4c116ul, 0x1e376c08ul, 0x2748774cul, 0x34b0bcb5ul,
    0x391c0cb3ul, 0x4ed8aa4aul, 0x5b9cca4ful, 0x682e6ff3ul,
    0x748f82eeul, 0x78a5636ful, 0x84c87814ul, 0x8cc70208ul,
    0x90befffaul, 0xa4506cebul, 0xbef9a3f7ul, 0xc67178f2ul,
};

/* Compile 64 bytes of hash data into SHA256 digest value   */
/* NOTE: this routine assumes that the byte order in the    */
/* ctx->wbuf[] at this point is such that low address bytes */
/* in the ORIGINAL byte stream will go into the high end of */
/* words on BOTH big and little endian systems              */

__device__ VOID_RETURN sha256_compile(sha256_ctx ctx[1])
{
#if !defined(UNROLL_SHA2)

    uint_32t j, *p = ctx->wbuf, v[8];

    memcpy(v, ctx->hash, 8 * sizeof(uint_32t));

    for(j = 0; j < 64; j += 16)
    {
        v_cycle( 0, j); v_cycle( 1, j);
        v_cycle( 2, j); v_cycle( 3, j);
        v_cycle( 4, j); v_cycle( 5, j);
        v_cycle( 6, j); v_cycle( 7, j);
        v_cycle( 8, j); v_cycle( 9, j);
        v_cycle(10, j); v_cycle(11, j);
        v_cycle(12, j); v_cycle(13, j);
        v_cycle(14, j); v_cycle(15, j);
    }

    ctx->hash[0] += v[0]; ctx->hash[1] += v[1];
    ctx->hash[2] += v[2]; ctx->hash[3] += v[3];
    ctx->hash[4] += v[4]; ctx->hash[5] += v[5];
    ctx->hash[6] += v[6]; ctx->hash[7] += v[7];

#else

    uint_32t *p = ctx->wbuf,v0,v1,v2,v3,v4,v5,v6,v7;

    v0 = ctx->hash[0]; v1 = ctx->hash[1];
    v2 = ctx->hash[2]; v3 = ctx->hash[3];
    v4 = ctx->hash[4]; v5 = ctx->hash[5];
    v6 = ctx->hash[6]; v7 = ctx->hash[7];

    one_cycle(0,1,2,3,4,5,6,7,k256[ 0],p[ 0]);
    one_cycle(7,0,1,2,3,4,5,6,k256[ 1],p[ 1]);
    one_cycle(6,7,0,1,2,3,4,5,k256[ 2],p[ 2]);
    one_cycle(5,6,7,0,1,2,3,4,k256[ 3],p[ 3]);
    one_cycle(4,5,6,7,0,1,2,3,k256[ 4],p[ 4]);
    one_cycle(3,4,5,6,7,0,1,2,k256[ 5],p[ 5]);
    one_cycle(2,3,4,5,6,7,0,1,k256[ 6],p[ 6]);
    one_cycle(1,2,3,4,5,6,7,0,k256[ 7],p[ 7]);
    one_cycle(0,1,2,3,4,5,6,7,k256[ 8],p[ 8]);
    one_cycle(7,0,1,2,3,4,5,6,k256[ 9],p[ 9]);
    one_cycle(6,7,0,1,2,3,4,5,k256[10],p[10]);
    one_cycle(5,6,7,0,1,2,3,4,k256[11],p[11]);
    one_cycle(4,5,6,7,0,1,2,3,k256[12],p[12]);
    one_cycle(3,4,5,6,7,0,1,2,k256[13],p[13]);
    one_cycle(2,3,4,5,6,7,0,1,k256[14],p[14]);
    one_cycle(1,2,3,4,5,6,7,0,k256[15],p[15]);

    one_cycle(0,1,2,3,4,5,6,7,k256[16],hf( 0));
    one_cycle(7,0,1,2,3,4,5,6,k256[17],hf( 1));
    one_cycle(6,7,0,1,2,3,4,5,k256[18],hf( 2));
    one_cycle(5,6,7,0,1,2,3,4,k256[19],hf( 3));
    one_cycle(4,5,6,7,0,1,2,3,k256[20],hf( 4));
    one_cycle(3,4,5,6,7,0,1,2,k256[21],hf( 5));
    one_cycle(2,3,4,5,6,7,0,1,k256[22],hf( 6));
    one_cycle(1,2,3,4,5,6,7,0,k256[23],hf( 7));
    one_cycle(0,1,2,3,4,5,6,7,k256[24],hf( 8));
    one_cycle(7,0,1,2,3,4,5,6,k256[25],hf( 9));
    one_cycle(6,7,0,1,2,3,4,5,k256[26],hf(10));
    one_cycle(5,6,7,0,1,2,3,4,k256[27],hf(11));
    one_cycle(4,5,6,7,0,1,2,3,k256[28],hf(12));
    one_cycle(3,4,5,6,7,0,1,2,k256[29],hf(13));
    one_cycle(2,3,4,5,6,7,0,1,k256[30],hf(14));
    one_cycle(1,2,3,4,5,6,7,0,k256[31],hf(15));

    one_cycle(0,1,2,3,4,5,6,7,k256[32],hf( 0));
    one_cycle(7,0,1,2,3,4,5,6,k256[33],hf( 1));
    one_cycle(6,7,0,1,2,3,4,5,k256[34],hf( 2));
    one_cycle(5,6,7,0,1,2,3,4,k256[35],hf( 3));
    one_cycle(4,5,6,7,0,1,2,3,k256[36],hf( 4));
    one_cycle(3,4,5,6,7,0,1,2,k256[37],hf( 5));
    one_cycle(2,3,4,5,6,7,0,1,k256[38],hf( 6));
    one_cycle(1,2,3,4,5,6,7,0,k256[39],hf( 7));
    one_cycle(0,1,2,3,4,5,6,7,k256[40],hf( 8));
    one_cycle(7,0,1,2,3,4,5,6,k256[41],hf( 9));
    one_cycle(6,7,0,1,2,3,4,5,k256[42],hf(10));
    one_cycle(5,6,7,0,1,2,3,4,k256[43],hf(11));
    one_cycle(4,5,6,7,0,1,2,3,k256[44],hf(12));
    one_cycle(3,4,5,6,7,0,1,2,k256[45],hf(13));
    one_cycle(2,3,4,5,6,7,0,1,k256[46],hf(14));
    one_cycle(1,2,3,4,5,6,7,0,k256[47],hf(15));

    one_cycle(0,1,2,3,4,5,6,7,k256[48],hf( 0));
    one_cycle(7,0,1,2,3,4,5,6,k256[49],hf( 1));
    one_cycle(6,7,0,1,2,3,4,5,k256[50],hf( 2));
    one_cycle(5,6,7,0,1,2,3,4,k256[51],hf( 3));
    one_cycle(4,5,6,7,0,1,2,3,k256[52],hf( 4));
    one_cycle(3,4,5,6,7,0,1,2,k256[53],hf( 5));
    one_cycle(2,3,4,5,6,7,0,1,k256[54],hf( 6));
    one_cycle(1,2,3,4,5,6,7,0,k256[55],hf( 7));
    one_cycle(0,1,2,3,4,5,6,7,k256[56],hf( 8));
    one_cycle(7,0,1,2,3,4,5,6,k256[57],hf( 9));
    one_cycle(6,7,0,1,2,3,4,5,k256[58],hf(10));
    one_cycle(5,6,7,0,1,2,3,4,k256[59],hf(11));
    one_cycle(4,5,6,7,0,1,2,3,k256[60],hf(12));
    one_cycle(3,4,5,6,7,0,1,2,k256[61],hf(13));
    one_cycle(2,3,4,5,6,7,0,1,k256[62],hf(14));
    one_cycle(1,2,3,4,5,6,7,0,k256[63],hf(15));

    ctx->hash[0] += v0; ctx->hash[1] += v1;
    ctx->hash[2] += v2; ctx->hash[3] += v3;
    ctx->hash[4] += v4; ctx->hash[5] += v5;
    ctx->hash[6] += v6; ctx->hash[7] += v7;
#endif
}

/* SHA256 hash data in an array of bytes into hash buffer   */
/* and call the hash_compile function as required.          */

__device__ VOID_RETURN sha256_hash(const unsigned char data[], unsigned long len, sha256_ctx ctx[1])
{   uint_32t pos = (uint_32t)(ctx->count[0] & SHA256_MASK),
             space = SHA256_BLOCK_SIZE - pos;
    const unsigned char *sp = data;

    if((ctx->count[0] += len) < len)
        ++(ctx->count[1]);

    while(len >= space)     /* tranfer whole blocks while possible  */
    {
        memcpy(((unsigned char*)ctx->wbuf) + pos, sp, space);
        sp += space; len -= space; space = SHA256_BLOCK_SIZE; pos = 0;
        bsw_32(ctx->wbuf, SHA256_BLOCK_SIZE >> 2)
        sha256_compile(ctx);
    }

    memcpy(((unsigned char*)ctx->wbuf) + pos, sp, len);
}

/* SHA256 Final padding and digest calculation  */

__device__ static void sha_end1(unsigned char hval[], sha256_ctx ctx[1], const unsigned int hlen)
{   uint_32t    i = (uint_32t)(ctx->count[0] & SHA256_MASK);

    /* put bytes in the buffer in an order in which references to   */
    /* 32-bit words will put bytes with lower addresses into the    */
    /* top of 32 bit words on BOTH big and little endian machines   */
    bsw_32(ctx->wbuf, (i + 3) >> 2)

    /* we now need to mask valid bytes and add the padding which is */
    /* a single 1 bit and as many zero bits as necessary. Note that */
    /* we can always add the first padding byte here because the    */
    /* buffer always has at least one empty slot                    */
    ctx->wbuf[i >> 2] &= 0xffffff80 << 8 * (~i & 3);
    ctx->wbuf[i >> 2] |= 0x00000080 << 8 * (~i & 3);

    /* we need 9 or more empty positions, one for the padding byte  */
    /* (above) and eight for the length count.  If there is not     */
    /* enough space pad and empty the buffer                        */
    if(i > SHA256_BLOCK_SIZE - 9)
    {
        if(i < 60) ctx->wbuf[15] = 0;
        sha256_compile(ctx);
        i = 0;
    }
    else    /* compute a word index for the empty buffer positions  */
        i = (i >> 2) + 1;

    while(i < 14) /* and zero pad all but last two positions        */
        ctx->wbuf[i++] = 0;

    /* the following 32-bit length fields are assembled in the      */
    /* wrong byte order on little endian machines but this is       */
    /* corrected later since they are only ever used as 32-bit      */
    /* word values.                                                 */
    ctx->wbuf[14] = (ctx->count[1] << 3) | (ctx->count[0] >> 29);
    ctx->wbuf[15] = ctx->count[0] << 3;
    sha256_compile(ctx);

    /* extract the hash value as bytes in case the hash buffer is   */
    /* mislaigned for 32-bit words                                  */
    for(i = 0; i < hlen; ++i)
        hval[i] = (unsigned char)(ctx->hash[i >> 2] >> (8 * (~i & 3)));
}

#endif

#if defined(SHA_224)

__constant__ const uint_32t i224[8] =
{
    0xc1059ed8ul, 0x367cd507ul, 0x3070dd17ul, 0xf70e5939ul,
    0xffc00b31ul, 0x68581511ul, 0x64f98fa7ul, 0xbefa4fa4ul
};

__device__ VOID_RETURN sha224_begin(sha224_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i224, 8 * sizeof(uint_32t));
}

__device__ VOID_RETURN sha224_end(unsigned char hval[], sha224_ctx ctx[1])
{
    sha_end1(hval, ctx, SHA224_DIGEST_SIZE);
}

__device__ VOID_RETURN sha224(unsigned char hval[], const unsigned char data[], unsigned long len)
{   sha224_ctx  cx[1];

    sha224_begin(cx);
    sha224_hash(data, len, cx);
    sha_end1(hval, cx, SHA224_DIGEST_SIZE);
}

#endif

#if defined(SHA_256)

__constant__ const uint_32t i256[8] =
{
    0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
    0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
};

__device__ VOID_RETURN sha256_begin(sha256_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i256, 8 * sizeof(uint_32t));
}

__device__ VOID_RETURN sha256_end(unsigned char hval[], sha256_ctx ctx[1])
{
    sha_end1(hval, ctx, SHA256_DIGEST_SIZE);
}

__device__ VOID_RETURN sha256(unsigned char hval[], const unsigned char data[], unsigned long len)
{   sha256_ctx  cx[1];

    sha256_begin(cx);
    sha256_hash(data, len, cx);
    sha_end1(hval, cx, SHA256_DIGEST_SIZE);
}

#endif

#if defined(SHA_384) || defined(SHA_512)

#define SHA512_MASK (SHA512_BLOCK_SIZE - 1)

#define rotr64(x,n)   (((x) >> n) | ((x) << (64 - n)))

#if !defined(bswap_64)
#define bswap_64(x) (((uint_64t)(bswap_32((uint_32t)(x)))) << 32 | bswap_32((uint_32t)((x) >> 32)))
#endif

#if defined(SWAP_BYTES)
#define bsw_64(p,n) \
    { int _i = (n); while(_i--) ((uint_64t*)p)[_i] = bswap_64(((uint_64t*)p)[_i]); }
#else
#define bsw_64(p,n)
#endif

/* SHA512 mixing function definitions   */

#ifdef   s_0
# undef  s_0
# undef  s_1
# undef  g_0
# undef  g_1
# undef  k_0
#endif

#define s_0(x)  (rotr64((x), 28) ^ rotr64((x), 34) ^ rotr64((x), 39))
#define s_1(x)  (rotr64((x), 14) ^ rotr64((x), 18) ^ rotr64((x), 41))
#define g_0(x)  (rotr64((x),  1) ^ rotr64((x),  8) ^ ((x) >>  7))
#define g_1(x)  (rotr64((x), 19) ^ rotr64((x), 61) ^ ((x) >>  6))
#define k_0     k512

/* SHA384/SHA512 mixing data    */

__constant__ const uint_64t  k512[80] =
{
    li_64(428a2f98d728ae22), li_64(7137449123ef65cd),
    li_64(b5c0fbcfec4d3b2f), li_64(e9b5dba58189dbbc),
    li_64(3956c25bf348b538), li_64(59f111f1b605d019),
    li_64(923f82a4af194f9b), li_64(ab1c5ed5da6d8118),
    li_64(d807aa98a3030242), li_64(12835b0145706fbe),
    li_64(243185be4ee4b28c), li_64(550c7dc3d5ffb4e2),
    li_64(72be5d74f27b896f), li_64(80deb1fe3b1696b1),
    li_64(9bdc06a725c71235), li_64(c19bf174cf692694),
    li_64(e49b69c19ef14ad2), li_64(efbe4786384f25e3),
    li_64(0fc19dc68b8cd5b5), li_64(240ca1cc77ac9c65),
    li_64(2de92c6f592b0275), li_64(4a7484aa6ea6e483),
    li_64(5cb0a9dcbd41fbd4), li_64(76f988da831153b5),
    li_64(983e5152ee66dfab), li_64(a831c66d2db43210),
    li_64(b00327c898fb213f), li_64(bf597fc7beef0ee4),
    li_64(c6e00bf33da88fc2), li_64(d5a79147930aa725),
    li_64(06ca6351e003826f), li_64(142929670a0e6e70),
    li_64(27b70a8546d22ffc), li_64(2e1b21385c26c926),
    li_64(4d2c6dfc5ac42aed), li_64(53380d139d95b3df),
    li_64(650a73548baf63de), li_64(766a0abb3c77b2a8),
    li_64(81c2c92e47edaee6), li_64(92722c851482353b),
    li_64(a2bfe8a14cf10364), li_64(a81a664bbc423001),
    li_64(c24b8b70d0f89791), li_64(c76c51a30654be30),
    li_64(d192e819d6ef5218), li_64(d69906245565a910),
    li_64(f40e35855771202a), li_64(106aa07032bbd1b8),
    li_64(19a4c116b8d2d0c8), li_64(1e376c085141ab53),
    li_64(2748774cdf8eeb99), li_64(34b0bcb5e19b48a8),
    li_64(391c0cb3c5c95a63), li_64(4ed8aa4ae3418acb),
    li_64(5b9cca4f7763e373), li_64(682e6ff3d6b2b8a3),
    li_64(748f82ee5defb2fc), li_64(78a5636f43172f60),
    li_64(84c87814a1f0ab72), li_64(8cc702081a6439ec),
    li_64(90befffa23631e28), li_64(a4506cebde82bde9),
    li_64(bef9a3f7b2c67915), li_64(c67178f2e372532b),
    li_64(ca273eceea26619c), li_64(d186b8c721c0c207),
    li_64(eada7dd6cde0eb1e), li_64(f57d4f7fee6ed178),
    li_64(06f067aa72176fba), li_64(0a637dc5a2c898a6),
    li_64(113f9804bef90dae), li_64(1b710b35131c471b),
    li_64(28db77f523047d84), li_64(32caab7b40c72493),
    li_64(3c9ebe0a15c9bebc), li_64(431d67c49c100d4c),
    li_64(4cc5d4becb3e42b6), li_64(597f299cfc657e2a),
    li_64(5fcb6fab3ad6faec), li_64(6c44198c4a475817)
};

/* Compile 128 bytes of hash data into SHA384/512 digest    */
/* NOTE: this routine assumes that the byte order in the    */
/* ctx->wbuf[] at this point is such that low address bytes */
/* in the ORIGINAL byte stream will go into the high end of */
/* words on BOTH big and little endian systems              */

__device__ VOID_RETURN sha512_compile(sha512_ctx ctx[1])
{   uint_64t    v[8], *p = ctx->wbuf;
    uint_32t    j;

    memcpy(v, ctx->hash, 8 * sizeof(uint_64t));

    for(j = 0; j < 80; j += 16)
    {
        v_cycle( 0, j); v_cycle( 1, j);
        v_cycle( 2, j); v_cycle( 3, j);
        v_cycle( 4, j); v_cycle( 5, j);
        v_cycle( 6, j); v_cycle( 7, j);
        v_cycle( 8, j); v_cycle( 9, j);
        v_cycle(10, j); v_cycle(11, j);
        v_cycle(12, j); v_cycle(13, j);
        v_cycle(14, j); v_cycle(15, j);
    }

    ctx->hash[0] += v[0]; ctx->hash[1] += v[1];
    ctx->hash[2] += v[2]; ctx->hash[3] += v[3];
    ctx->hash[4] += v[4]; ctx->hash[5] += v[5];
    ctx->hash[6] += v[6]; ctx->hash[7] += v[7];
}

/* Compile 128 bytes of hash data into SHA256 digest value  */
/* NOTE: this routine assumes that the byte order in the    */
/* ctx->wbuf[] at this point is in such an order that low   */
/* address bytes in the ORIGINAL byte stream placed in this */
/* buffer will now go to the high end of words on BOTH big  */
/* and little endian systems                                */

__device__ VOID_RETURN sha512_hash(const unsigned char data[], unsigned long len, sha512_ctx ctx[1])
{   uint_32t pos = (uint_32t)(ctx->count[0] & SHA512_MASK),
             space = SHA512_BLOCK_SIZE - pos;
    const unsigned char *sp = data;

    if((ctx->count[0] += len) < len)
        ++(ctx->count[1]);

    while(len >= space)     /* tranfer whole blocks while possible  */
    {
        memcpy(((unsigned char*)ctx->wbuf) + pos, sp, space);
        sp += space; len -= space; space = SHA512_BLOCK_SIZE; pos = 0;
        bsw_64(ctx->wbuf, SHA512_BLOCK_SIZE >> 3);
        sha512_compile(ctx);
    }

    memcpy(((unsigned char*)ctx->wbuf) + pos, sp, len);
}

/* SHA384/512 Final padding and digest calculation  */

__device__ static void sha_end2(unsigned char hval[], sha512_ctx ctx[1], const unsigned int hlen)
{   uint_32t    i = (uint_32t)(ctx->count[0] & SHA512_MASK);

    /* put bytes in the buffer in an order in which references to   */
    /* 32-bit words will put bytes with lower addresses into the    */
    /* top of 32 bit words on BOTH big and little endian machines   */
    bsw_64(ctx->wbuf, (i + 7) >> 3);

    /* we now need to mask valid bytes and add the padding which is */
    /* a single 1 bit and as many zero bits as necessary. Note that */
    /* we can always add the first padding byte here because the    */
    /* buffer always has at least one empty slot                    */
    ctx->wbuf[i >> 3] &= li_64(ffffffffffffff00) << 8 * (~i & 7);
    ctx->wbuf[i >> 3] |= li_64(0000000000000080) << 8 * (~i & 7);

    /* we need 17 or more empty byte positions, one for the padding */
    /* byte (above) and sixteen for the length count.  If there is  */
    /* not enough space pad and empty the buffer                    */
    if(i > SHA512_BLOCK_SIZE - 17)
    {
        if(i < 120) ctx->wbuf[15] = 0;
        sha512_compile(ctx);
        i = 0;
    }
    else
        i = (i >> 3) + 1;

    while(i < 14)
        ctx->wbuf[i++] = 0;

    /* the following 64-bit length fields are assembled in the      */
    /* wrong byte order on little endian machines but this is       */
    /* corrected later since they are only ever used as 64-bit      */
    /* word values.                                                 */
    ctx->wbuf[14] = (ctx->count[1] << 3) | (ctx->count[0] >> 61);
    ctx->wbuf[15] = ctx->count[0] << 3;
    sha512_compile(ctx);

    /* extract the hash value as bytes in case the hash buffer is   */
    /* misaligned for 32-bit words                                  */
    for(i = 0; i < hlen; ++i)
        hval[i] = (unsigned char)(ctx->hash[i >> 3] >> (8 * (~i & 7)));
}

#endif

#if defined(SHA_384)

/* SHA384 initialisation data   */

__constant__ const uint_64t  i384[80] =
{
    li_64(cbbb9d5dc1059ed8), li_64(629a292a367cd507),
    li_64(9159015a3070dd17), li_64(152fecd8f70e5939),
    li_64(67332667ffc00b31), li_64(8eb44a8768581511),
    li_64(db0c2e0d64f98fa7), li_64(47b5481dbefa4fa4)
};

__device__ VOID_RETURN sha384_begin(sha384_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i384, 8 * sizeof(uint_64t));
}

__device__ VOID_RETURN sha384_end(unsigned char hval[], sha384_ctx ctx[1])
{
    sha_end2(hval, ctx, SHA384_DIGEST_SIZE);
}

__device__ VOID_RETURN sha384(unsigned char hval[], const unsigned char data[], unsigned long len)
{   sha384_ctx  cx[1];

    sha384_begin(cx);
    sha384_hash(data, len, cx);
    sha_end2(hval, cx, SHA384_DIGEST_SIZE);
}

#endif

#if defined(SHA_512)

/* SHA512 initialisation data   */

__constant__ const uint_64t  i512[80] =
{
    li_64(6a09e667f3bcc908), li_64(bb67ae8584caa73b),
    li_64(3c6ef372fe94f82b), li_64(a54ff53a5f1d36f1),
    li_64(510e527fade682d1), li_64(9b05688c2b3e6c1f),
    li_64(1f83d9abfb41bd6b), li_64(5be0cd19137e2179)
};

__device__ VOID_RETURN sha512_begin(sha512_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i512, 8 * sizeof(uint_64t));
}

__device__ VOID_RETURN sha512_end(unsigned char hval[], sha512_ctx ctx[1])
{
    sha_end2(hval, ctx, SHA512_DIGEST_SIZE);
}

__device__ VOID_RETURN sha512(unsigned char hval[], const unsigned char data[], unsigned long len)
{   sha512_ctx  cx[1];

    sha512_begin(cx);
    sha512_hash(data, len, cx);
    sha_end2(hval, cx, SHA512_DIGEST_SIZE);
}

#endif

#if defined(SHA_2)

#define CTX_224(x)  ((x)->uu->ctx256)
#define CTX_256(x)  ((x)->uu->ctx256)
#define CTX_384(x)  ((x)->uu->ctx512)
#define CTX_512(x)  ((x)->uu->ctx512)

/* SHA2 initialisation */

__device__ INT_RETURN sha2_begin(unsigned long len, sha2_ctx ctx[1])
{
    switch(len)
    {
#if defined(SHA_224)
        case 224:
        case  28:   CTX_256(ctx)->count[0] = CTX_256(ctx)->count[1] = 0;
                    memcpy(CTX_256(ctx)->hash, i224, 32);
                    ctx->sha2_len = 28; return EXIT_SUCCESS;
#endif
#if defined(SHA_256)
        case 256:
        case  32:   CTX_256(ctx)->count[0] = CTX_256(ctx)->count[1] = 0;
                    memcpy(CTX_256(ctx)->hash, i256, 32);
                    ctx->sha2_len = 32; return EXIT_SUCCESS;
#endif
#if defined(SHA_384)
        case 384:
        case  48:   CTX_384(ctx)->count[0] = CTX_384(ctx)->count[1] = 0;
                    memcpy(CTX_384(ctx)->hash, i384, 64);
                    ctx->sha2_len = 48; return EXIT_SUCCESS;
#endif
#if defined(SHA_512)
        case 512:
        case  64:   CTX_512(ctx)->count[0] = CTX_512(ctx)->count[1] = 0;
                    memcpy(CTX_512(ctx)->hash, i512, 64);
                    ctx->sha2_len = 64; return EXIT_SUCCESS;
#endif
        default:    return EXIT_FAILURE;
    }
}

__device__ VOID_RETURN sha2_hash(const unsigned char data[], unsigned long len, sha2_ctx ctx[1])
{
    switch(ctx->sha2_len)
    {
#if defined(SHA_224)
        case 28: sha224_hash(data, len, CTX_224(ctx)); return;
#endif
#if defined(SHA_256)
        case 32: sha256_hash(data, len, CTX_256(ctx)); return;
#endif
#if defined(SHA_384)
        case 48: sha384_hash(data, len, CTX_384(ctx)); return;
#endif
#if defined(SHA_512)
        case 64: sha512_hash(data, len, CTX_512(ctx)); return;
#endif
    }
}

__device__ VOID_RETURN sha2_end(unsigned char hval[], sha2_ctx ctx[1])
{
    switch(ctx->sha2_len)
    {
#if defined(SHA_224)
        case 28: sha_end1(hval, CTX_224(ctx), SHA224_DIGEST_SIZE); return;
#endif
#if defined(SHA_256)
        case 32: sha_end1(hval, CTX_256(ctx), SHA256_DIGEST_SIZE); return;
#endif
#if defined(SHA_384)
        case 48: sha_end2(hval, CTX_384(ctx), SHA384_DIGEST_SIZE); return;
#endif
#if defined(SHA_512)
        case 64: sha_end2(hval, CTX_512(ctx), SHA512_DIGEST_SIZE); return;
#endif
    }
}

__device__ INT_RETURN sha2(unsigned char hval[], unsigned long size,
                                const unsigned char data[], unsigned long len)
{   sha2_ctx    cx[1];

    if(sha2_begin(size, cx) == EXIT_SUCCESS)
    {
        sha2_hash(data, len, cx); sha2_end(hval, cx); return EXIT_SUCCESS;
    }
    else
        return EXIT_FAILURE;
}

#endif

#if defined(__cplusplus)
}
#endif
/**
 * The Whirlpool hashing function.
 *
 * <P>
 * <b>References</b>
 *
 * <P>
 * The Whirlpool algorithm was developed by
 * <a href="mailto:pbarreto@scopus.com.br">Paulo S. L. M. Barreto</a> and
 * <a href="mailto:vincent.rijmen@cryptomathic.com">Vincent Rijmen</a>.
 *
 * See
 *      P.S.L.M. Barreto, V. Rijmen,
 *      ``The Whirlpool hashing function,''
 *      NESSIE submission, 2000 (tweaked version, 2001),
 *      <https://www.cosic.esat.kuleuven.ac.be/nessie/workshop/submissions/whirlpool.zip>
 * 
 * @author  Paulo S.L.M. Barreto
 * @author  Vincent Rijmen.
 * Adapted for TrueCrypt.
 *
 * @version 3.0 (2003.03.12)
 *
 * =============================================================================
 *
 * Differences from version 2.1:
 *
 * - Suboptimal diffusion matrix replaced by cir(1, 1, 4, 1, 8, 5, 2, 9).
 *
 * =============================================================================
 *
 * Differences from version 2.0:
 *
 * - Generation of ISO/IEC 10118-3 test vectors.
 * - Bug fix: nonzero carry was ignored when tallying the data length
 *      (this bug apparently only manifested itself when feeding data
 *      in pieces rather than in a single chunk at once).
 * - Support for MS Visual C++ 64-bit integer arithmetic.
 *
 * Differences from version 1.0:
 *
 * - Original S-box replaced by the tweaked, hardware-efficient version.
 *
 * =============================================================================
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
 /* The code contained in this file (Whirlpool.c) is in the public domain. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Whirlpool.cuh"

/* #define TRACE_INTERMEDIATE_VALUES */

/*
 * The number of rounds of the internal dedicated block cipher.
 */
#define R 10

/*
 * Though Whirlpool is endianness-neutral, the encryption tables are listed
 * in BIG-ENDIAN format, which is adopted throughout this implementation
 * (but little-endian notation would be equally suitable if consistently
 * employed).
 */

__constant__ static const u64 C0[256] = {
    LL(0x18186018c07830d8), LL(0x23238c2305af4626), LL(0xc6c63fc67ef991b8), LL(0xe8e887e8136fcdfb),
    LL(0x878726874ca113cb), LL(0xb8b8dab8a9626d11), LL(0x0101040108050209), LL(0x4f4f214f426e9e0d),
    LL(0x3636d836adee6c9b), LL(0xa6a6a2a6590451ff), LL(0xd2d26fd2debdb90c), LL(0xf5f5f3f5fb06f70e),
    LL(0x7979f979ef80f296), LL(0x6f6fa16f5fcede30), LL(0x91917e91fcef3f6d), LL(0x52525552aa07a4f8),
    LL(0x60609d6027fdc047), LL(0xbcbccabc89766535), LL(0x9b9b569baccd2b37), LL(0x8e8e028e048c018a),
    LL(0xa3a3b6a371155bd2), LL(0x0c0c300c603c186c), LL(0x7b7bf17bff8af684), LL(0x3535d435b5e16a80),
    LL(0x1d1d741de8693af5), LL(0xe0e0a7e05347ddb3), LL(0xd7d77bd7f6acb321), LL(0xc2c22fc25eed999c),
    LL(0x2e2eb82e6d965c43), LL(0x4b4b314b627a9629), LL(0xfefedffea321e15d), LL(0x575741578216aed5),
    LL(0x15155415a8412abd), LL(0x7777c1779fb6eee8), LL(0x3737dc37a5eb6e92), LL(0xe5e5b3e57b56d79e),
    LL(0x9f9f469f8cd92313), LL(0xf0f0e7f0d317fd23), LL(0x4a4a354a6a7f9420), LL(0xdada4fda9e95a944),
    LL(0x58587d58fa25b0a2), LL(0xc9c903c906ca8fcf), LL(0x2929a429558d527c), LL(0x0a0a280a5022145a),
    LL(0xb1b1feb1e14f7f50), LL(0xa0a0baa0691a5dc9), LL(0x6b6bb16b7fdad614), LL(0x85852e855cab17d9),
    LL(0xbdbdcebd8173673c), LL(0x5d5d695dd234ba8f), LL(0x1010401080502090), LL(0xf4f4f7f4f303f507),
    LL(0xcbcb0bcb16c08bdd), LL(0x3e3ef83eedc67cd3), LL(0x0505140528110a2d), LL(0x676781671fe6ce78),
    LL(0xe4e4b7e47353d597), LL(0x27279c2725bb4e02), LL(0x4141194132588273), LL(0x8b8b168b2c9d0ba7),
    LL(0xa7a7a6a7510153f6), LL(0x7d7de97dcf94fab2), LL(0x95956e95dcfb3749), LL(0xd8d847d88e9fad56),
    LL(0xfbfbcbfb8b30eb70), LL(0xeeee9fee2371c1cd), LL(0x7c7ced7cc791f8bb), LL(0x6666856617e3cc71),
    LL(0xdddd53dda68ea77b), LL(0x17175c17b84b2eaf), LL(0x4747014702468e45), LL(0x9e9e429e84dc211a),
    LL(0xcaca0fca1ec589d4), LL(0x2d2db42d75995a58), LL(0xbfbfc6bf9179632e), LL(0x07071c07381b0e3f),
    LL(0xadad8ead012347ac), LL(0x5a5a755aea2fb4b0), LL(0x838336836cb51bef), LL(0x3333cc3385ff66b6),
    LL(0x636391633ff2c65c), LL(0x02020802100a0412), LL(0xaaaa92aa39384993), LL(0x7171d971afa8e2de),
    LL(0xc8c807c80ecf8dc6), LL(0x19196419c87d32d1), LL(0x494939497270923b), LL(0xd9d943d9869aaf5f),
    LL(0xf2f2eff2c31df931), LL(0xe3e3abe34b48dba8), LL(0x5b5b715be22ab6b9), LL(0x88881a8834920dbc),
    LL(0x9a9a529aa4c8293e), LL(0x262698262dbe4c0b), LL(0x3232c8328dfa64bf), LL(0xb0b0fab0e94a7d59),
    LL(0xe9e983e91b6acff2), LL(0x0f0f3c0f78331e77), LL(0xd5d573d5e6a6b733), LL(0x80803a8074ba1df4),
    LL(0xbebec2be997c6127), LL(0xcdcd13cd26de87eb), LL(0x3434d034bde46889), LL(0x48483d487a759032),
    LL(0xffffdbffab24e354), LL(0x7a7af57af78ff48d), LL(0x90907a90f4ea3d64), LL(0x5f5f615fc23ebe9d),
    LL(0x202080201da0403d), LL(0x6868bd6867d5d00f), LL(0x1a1a681ad07234ca), LL(0xaeae82ae192c41b7),
    LL(0xb4b4eab4c95e757d), LL(0x54544d549a19a8ce), LL(0x93937693ece53b7f), LL(0x222288220daa442f),
    LL(0x64648d6407e9c863), LL(0xf1f1e3f1db12ff2a), LL(0x7373d173bfa2e6cc), LL(0x12124812905a2482),
    LL(0x40401d403a5d807a), LL(0x0808200840281048), LL(0xc3c32bc356e89b95), LL(0xecec97ec337bc5df),
    LL(0xdbdb4bdb9690ab4d), LL(0xa1a1bea1611f5fc0), LL(0x8d8d0e8d1c830791), LL(0x3d3df43df5c97ac8),
    LL(0x97976697ccf1335b), LL(0x0000000000000000), LL(0xcfcf1bcf36d483f9), LL(0x2b2bac2b4587566e),
    LL(0x7676c57697b3ece1), LL(0x8282328264b019e6), LL(0xd6d67fd6fea9b128), LL(0x1b1b6c1bd87736c3),
    LL(0xb5b5eeb5c15b7774), LL(0xafaf86af112943be), LL(0x6a6ab56a77dfd41d), LL(0x50505d50ba0da0ea),
    LL(0x45450945124c8a57), LL(0xf3f3ebf3cb18fb38), LL(0x3030c0309df060ad), LL(0xefef9bef2b74c3c4),
    LL(0x3f3ffc3fe5c37eda), LL(0x55554955921caac7), LL(0xa2a2b2a2791059db), LL(0xeaea8fea0365c9e9),
    LL(0x656589650fecca6a), LL(0xbabad2bab9686903), LL(0x2f2fbc2f65935e4a), LL(0xc0c027c04ee79d8e),
    LL(0xdede5fdebe81a160), LL(0x1c1c701ce06c38fc), LL(0xfdfdd3fdbb2ee746), LL(0x4d4d294d52649a1f),
    LL(0x92927292e4e03976), LL(0x7575c9758fbceafa), LL(0x06061806301e0c36), LL(0x8a8a128a249809ae),
    LL(0xb2b2f2b2f940794b), LL(0xe6e6bfe66359d185), LL(0x0e0e380e70361c7e), LL(0x1f1f7c1ff8633ee7),
    LL(0x6262956237f7c455), LL(0xd4d477d4eea3b53a), LL(0xa8a89aa829324d81), LL(0x96966296c4f43152),
    LL(0xf9f9c3f99b3aef62), LL(0xc5c533c566f697a3), LL(0x2525942535b14a10), LL(0x59597959f220b2ab),
    LL(0x84842a8454ae15d0), LL(0x7272d572b7a7e4c5), LL(0x3939e439d5dd72ec), LL(0x4c4c2d4c5a619816),
    LL(0x5e5e655eca3bbc94), LL(0x7878fd78e785f09f), LL(0x3838e038ddd870e5), LL(0x8c8c0a8c14860598),
    LL(0xd1d163d1c6b2bf17), LL(0xa5a5aea5410b57e4), LL(0xe2e2afe2434dd9a1), LL(0x616199612ff8c24e),
    LL(0xb3b3f6b3f1457b42), LL(0x2121842115a54234), LL(0x9c9c4a9c94d62508), LL(0x1e1e781ef0663cee),
    LL(0x4343114322528661), LL(0xc7c73bc776fc93b1), LL(0xfcfcd7fcb32be54f), LL(0x0404100420140824),
    LL(0x51515951b208a2e3), LL(0x99995e99bcc72f25), LL(0x6d6da96d4fc4da22), LL(0x0d0d340d68391a65),
    LL(0xfafacffa8335e979), LL(0xdfdf5bdfb684a369), LL(0x7e7ee57ed79bfca9), LL(0x242490243db44819),
    LL(0x3b3bec3bc5d776fe), LL(0xabab96ab313d4b9a), LL(0xcece1fce3ed181f0), LL(0x1111441188552299),
    LL(0x8f8f068f0c890383), LL(0x4e4e254e4a6b9c04), LL(0xb7b7e6b7d1517366), LL(0xebeb8beb0b60cbe0),
    LL(0x3c3cf03cfdcc78c1), LL(0x81813e817cbf1ffd), LL(0x94946a94d4fe3540), LL(0xf7f7fbf7eb0cf31c),
    LL(0xb9b9deb9a1676f18), LL(0x13134c13985f268b), LL(0x2c2cb02c7d9c5851), LL(0xd3d36bd3d6b8bb05),
    LL(0xe7e7bbe76b5cd38c), LL(0x6e6ea56e57cbdc39), LL(0xc4c437c46ef395aa), LL(0x03030c03180f061b),
    LL(0x565645568a13acdc), LL(0x44440d441a49885e), LL(0x7f7fe17fdf9efea0), LL(0xa9a99ea921374f88),
    LL(0x2a2aa82a4d825467), LL(0xbbbbd6bbb16d6b0a), LL(0xc1c123c146e29f87), LL(0x53535153a202a6f1),
    LL(0xdcdc57dcae8ba572), LL(0x0b0b2c0b58271653), LL(0x9d9d4e9d9cd32701), LL(0x6c6cad6c47c1d82b),
    LL(0x3131c43195f562a4), LL(0x7474cd7487b9e8f3), LL(0xf6f6fff6e309f115), LL(0x464605460a438c4c),
    LL(0xacac8aac092645a5), LL(0x89891e893c970fb5), LL(0x14145014a04428b4), LL(0xe1e1a3e15b42dfba),
    LL(0x16165816b04e2ca6), LL(0x3a3ae83acdd274f7), LL(0x6969b9696fd0d206), LL(0x09092409482d1241),
    LL(0x7070dd70a7ade0d7), LL(0xb6b6e2b6d954716f), LL(0xd0d067d0ceb7bd1e), LL(0xeded93ed3b7ec7d6),
    LL(0xcccc17cc2edb85e2), LL(0x424215422a578468), LL(0x98985a98b4c22d2c), LL(0xa4a4aaa4490e55ed),
    LL(0x2828a0285d885075), LL(0x5c5c6d5cda31b886), LL(0xf8f8c7f8933fed6b), LL(0x8686228644a411c2),
};

__constant__ static const u64 C1[256] = {
    LL(0xd818186018c07830), LL(0x2623238c2305af46), LL(0xb8c6c63fc67ef991), LL(0xfbe8e887e8136fcd),
    LL(0xcb878726874ca113), LL(0x11b8b8dab8a9626d), LL(0x0901010401080502), LL(0x0d4f4f214f426e9e),
    LL(0x9b3636d836adee6c), LL(0xffa6a6a2a6590451), LL(0x0cd2d26fd2debdb9), LL(0x0ef5f5f3f5fb06f7),
    LL(0x967979f979ef80f2), LL(0x306f6fa16f5fcede), LL(0x6d91917e91fcef3f), LL(0xf852525552aa07a4),
    LL(0x4760609d6027fdc0), LL(0x35bcbccabc897665), LL(0x379b9b569baccd2b), LL(0x8a8e8e028e048c01),
    LL(0xd2a3a3b6a371155b), LL(0x6c0c0c300c603c18), LL(0x847b7bf17bff8af6), LL(0x803535d435b5e16a),
    LL(0xf51d1d741de8693a), LL(0xb3e0e0a7e05347dd), LL(0x21d7d77bd7f6acb3), LL(0x9cc2c22fc25eed99),
    LL(0x432e2eb82e6d965c), LL(0x294b4b314b627a96), LL(0x5dfefedffea321e1), LL(0xd5575741578216ae),
    LL(0xbd15155415a8412a), LL(0xe87777c1779fb6ee), LL(0x923737dc37a5eb6e), LL(0x9ee5e5b3e57b56d7),
    LL(0x139f9f469f8cd923), LL(0x23f0f0e7f0d317fd), LL(0x204a4a354a6a7f94), LL(0x44dada4fda9e95a9),
    LL(0xa258587d58fa25b0), LL(0xcfc9c903c906ca8f), LL(0x7c2929a429558d52), LL(0x5a0a0a280a502214),
    LL(0x50b1b1feb1e14f7f), LL(0xc9a0a0baa0691a5d), LL(0x146b6bb16b7fdad6), LL(0xd985852e855cab17),
    LL(0x3cbdbdcebd817367), LL(0x8f5d5d695dd234ba), LL(0x9010104010805020), LL(0x07f4f4f7f4f303f5),
    LL(0xddcbcb0bcb16c08b), LL(0xd33e3ef83eedc67c), LL(0x2d0505140528110a), LL(0x78676781671fe6ce),
    LL(0x97e4e4b7e47353d5), LL(0x0227279c2725bb4e), LL(0x7341411941325882), LL(0xa78b8b168b2c9d0b),
    LL(0xf6a7a7a6a7510153), LL(0xb27d7de97dcf94fa), LL(0x4995956e95dcfb37), LL(0x56d8d847d88e9fad),
    LL(0x70fbfbcbfb8b30eb), LL(0xcdeeee9fee2371c1), LL(0xbb7c7ced7cc791f8), LL(0x716666856617e3cc),
    LL(0x7bdddd53dda68ea7), LL(0xaf17175c17b84b2e), LL(0x454747014702468e), LL(0x1a9e9e429e84dc21),
    LL(0xd4caca0fca1ec589), LL(0x582d2db42d75995a), LL(0x2ebfbfc6bf917963), LL(0x3f07071c07381b0e),
    LL(0xacadad8ead012347), LL(0xb05a5a755aea2fb4), LL(0xef838336836cb51b), LL(0xb63333cc3385ff66),
    LL(0x5c636391633ff2c6), LL(0x1202020802100a04), LL(0x93aaaa92aa393849), LL(0xde7171d971afa8e2),
    LL(0xc6c8c807c80ecf8d), LL(0xd119196419c87d32), LL(0x3b49493949727092), LL(0x5fd9d943d9869aaf),
    LL(0x31f2f2eff2c31df9), LL(0xa8e3e3abe34b48db), LL(0xb95b5b715be22ab6), LL(0xbc88881a8834920d),
    LL(0x3e9a9a529aa4c829), LL(0x0b262698262dbe4c), LL(0xbf3232c8328dfa64), LL(0x59b0b0fab0e94a7d),
    LL(0xf2e9e983e91b6acf), LL(0x770f0f3c0f78331e), LL(0x33d5d573d5e6a6b7), LL(0xf480803a8074ba1d),
    LL(0x27bebec2be997c61), LL(0xebcdcd13cd26de87), LL(0x893434d034bde468), LL(0x3248483d487a7590),
    LL(0x54ffffdbffab24e3), LL(0x8d7a7af57af78ff4), LL(0x6490907a90f4ea3d), LL(0x9d5f5f615fc23ebe),
    LL(0x3d202080201da040), LL(0x0f6868bd6867d5d0), LL(0xca1a1a681ad07234), LL(0xb7aeae82ae192c41),
    LL(0x7db4b4eab4c95e75), LL(0xce54544d549a19a8), LL(0x7f93937693ece53b), LL(0x2f222288220daa44),
    LL(0x6364648d6407e9c8), LL(0x2af1f1e3f1db12ff), LL(0xcc7373d173bfa2e6), LL(0x8212124812905a24),
    LL(0x7a40401d403a5d80), LL(0x4808082008402810), LL(0x95c3c32bc356e89b), LL(0xdfecec97ec337bc5),
    LL(0x4ddbdb4bdb9690ab), LL(0xc0a1a1bea1611f5f), LL(0x918d8d0e8d1c8307), LL(0xc83d3df43df5c97a),
    LL(0x5b97976697ccf133), LL(0x0000000000000000), LL(0xf9cfcf1bcf36d483), LL(0x6e2b2bac2b458756),
    LL(0xe17676c57697b3ec), LL(0xe68282328264b019), LL(0x28d6d67fd6fea9b1), LL(0xc31b1b6c1bd87736),
    LL(0x74b5b5eeb5c15b77), LL(0xbeafaf86af112943), LL(0x1d6a6ab56a77dfd4), LL(0xea50505d50ba0da0),
    LL(0x5745450945124c8a), LL(0x38f3f3ebf3cb18fb), LL(0xad3030c0309df060), LL(0xc4efef9bef2b74c3),
    LL(0xda3f3ffc3fe5c37e), LL(0xc755554955921caa), LL(0xdba2a2b2a2791059), LL(0xe9eaea8fea0365c9),
    LL(0x6a656589650fecca), LL(0x03babad2bab96869), LL(0x4a2f2fbc2f65935e), LL(0x8ec0c027c04ee79d),
    LL(0x60dede5fdebe81a1), LL(0xfc1c1c701ce06c38), LL(0x46fdfdd3fdbb2ee7), LL(0x1f4d4d294d52649a),
    LL(0x7692927292e4e039), LL(0xfa7575c9758fbcea), LL(0x3606061806301e0c), LL(0xae8a8a128a249809),
    LL(0x4bb2b2f2b2f94079), LL(0x85e6e6bfe66359d1), LL(0x7e0e0e380e70361c), LL(0xe71f1f7c1ff8633e),
    LL(0x556262956237f7c4), LL(0x3ad4d477d4eea3b5), LL(0x81a8a89aa829324d), LL(0x5296966296c4f431),
    LL(0x62f9f9c3f99b3aef), LL(0xa3c5c533c566f697), LL(0x102525942535b14a), LL(0xab59597959f220b2),
    LL(0xd084842a8454ae15), LL(0xc57272d572b7a7e4), LL(0xec3939e439d5dd72), LL(0x164c4c2d4c5a6198),
    LL(0x945e5e655eca3bbc), LL(0x9f7878fd78e785f0), LL(0xe53838e038ddd870), LL(0x988c8c0a8c148605),
    LL(0x17d1d163d1c6b2bf), LL(0xe4a5a5aea5410b57), LL(0xa1e2e2afe2434dd9), LL(0x4e616199612ff8c2),
    LL(0x42b3b3f6b3f1457b), LL(0x342121842115a542), LL(0x089c9c4a9c94d625), LL(0xee1e1e781ef0663c),
    LL(0x6143431143225286), LL(0xb1c7c73bc776fc93), LL(0x4ffcfcd7fcb32be5), LL(0x2404041004201408),
    LL(0xe351515951b208a2), LL(0x2599995e99bcc72f), LL(0x226d6da96d4fc4da), LL(0x650d0d340d68391a),
    LL(0x79fafacffa8335e9), LL(0x69dfdf5bdfb684a3), LL(0xa97e7ee57ed79bfc), LL(0x19242490243db448),
    LL(0xfe3b3bec3bc5d776), LL(0x9aabab96ab313d4b), LL(0xf0cece1fce3ed181), LL(0x9911114411885522),
    LL(0x838f8f068f0c8903), LL(0x044e4e254e4a6b9c), LL(0x66b7b7e6b7d15173), LL(0xe0ebeb8beb0b60cb),
    LL(0xc13c3cf03cfdcc78), LL(0xfd81813e817cbf1f), LL(0x4094946a94d4fe35), LL(0x1cf7f7fbf7eb0cf3),
    LL(0x18b9b9deb9a1676f), LL(0x8b13134c13985f26), LL(0x512c2cb02c7d9c58), LL(0x05d3d36bd3d6b8bb),
    LL(0x8ce7e7bbe76b5cd3), LL(0x396e6ea56e57cbdc), LL(0xaac4c437c46ef395), LL(0x1b03030c03180f06),
    LL(0xdc565645568a13ac), LL(0x5e44440d441a4988), LL(0xa07f7fe17fdf9efe), LL(0x88a9a99ea921374f),
    LL(0x672a2aa82a4d8254), LL(0x0abbbbd6bbb16d6b), LL(0x87c1c123c146e29f), LL(0xf153535153a202a6),
    LL(0x72dcdc57dcae8ba5), LL(0x530b0b2c0b582716), LL(0x019d9d4e9d9cd327), LL(0x2b6c6cad6c47c1d8),
    LL(0xa43131c43195f562), LL(0xf37474cd7487b9e8), LL(0x15f6f6fff6e309f1), LL(0x4c464605460a438c),
    LL(0xa5acac8aac092645), LL(0xb589891e893c970f), LL(0xb414145014a04428), LL(0xbae1e1a3e15b42df),
    LL(0xa616165816b04e2c), LL(0xf73a3ae83acdd274), LL(0x066969b9696fd0d2), LL(0x4109092409482d12),
    LL(0xd77070dd70a7ade0), LL(0x6fb6b6e2b6d95471), LL(0x1ed0d067d0ceb7bd), LL(0xd6eded93ed3b7ec7),
    LL(0xe2cccc17cc2edb85), LL(0x68424215422a5784), LL(0x2c98985a98b4c22d), LL(0xeda4a4aaa4490e55),
    LL(0x752828a0285d8850), LL(0x865c5c6d5cda31b8), LL(0x6bf8f8c7f8933fed), LL(0xc28686228644a411),
};

__constant__ static const u64 C2[256] = {
    LL(0x30d818186018c078), LL(0x462623238c2305af), LL(0x91b8c6c63fc67ef9), LL(0xcdfbe8e887e8136f),
    LL(0x13cb878726874ca1), LL(0x6d11b8b8dab8a962), LL(0x0209010104010805), LL(0x9e0d4f4f214f426e),
    LL(0x6c9b3636d836adee), LL(0x51ffa6a6a2a65904), LL(0xb90cd2d26fd2debd), LL(0xf70ef5f5f3f5fb06),
    LL(0xf2967979f979ef80), LL(0xde306f6fa16f5fce), LL(0x3f6d91917e91fcef), LL(0xa4f852525552aa07),
    LL(0xc04760609d6027fd), LL(0x6535bcbccabc8976), LL(0x2b379b9b569baccd), LL(0x018a8e8e028e048c),
    LL(0x5bd2a3a3b6a37115), LL(0x186c0c0c300c603c), LL(0xf6847b7bf17bff8a), LL(0x6a803535d435b5e1),
    LL(0x3af51d1d741de869), LL(0xddb3e0e0a7e05347), LL(0xb321d7d77bd7f6ac), LL(0x999cc2c22fc25eed),
    LL(0x5c432e2eb82e6d96), LL(0x96294b4b314b627a), LL(0xe15dfefedffea321), LL(0xaed5575741578216),
    LL(0x2abd15155415a841), LL(0xeee87777c1779fb6), LL(0x6e923737dc37a5eb), LL(0xd79ee5e5b3e57b56),
    LL(0x23139f9f469f8cd9), LL(0xfd23f0f0e7f0d317), LL(0x94204a4a354a6a7f), LL(0xa944dada4fda9e95),
    LL(0xb0a258587d58fa25), LL(0x8fcfc9c903c906ca), LL(0x527c2929a429558d), LL(0x145a0a0a280a5022),
    LL(0x7f50b1b1feb1e14f), LL(0x5dc9a0a0baa0691a), LL(0xd6146b6bb16b7fda), LL(0x17d985852e855cab),
    LL(0x673cbdbdcebd8173), LL(0xba8f5d5d695dd234), LL(0x2090101040108050), LL(0xf507f4f4f7f4f303),
    LL(0x8bddcbcb0bcb16c0), LL(0x7cd33e3ef83eedc6), LL(0x0a2d050514052811), LL(0xce78676781671fe6),
    LL(0xd597e4e4b7e47353), LL(0x4e0227279c2725bb), LL(0x8273414119413258), LL(0x0ba78b8b168b2c9d),
    LL(0x53f6a7a7a6a75101), LL(0xfab27d7de97dcf94), LL(0x374995956e95dcfb), LL(0xad56d8d847d88e9f),
    LL(0xeb70fbfbcbfb8b30), LL(0xc1cdeeee9fee2371), LL(0xf8bb7c7ced7cc791), LL(0xcc716666856617e3),
    LL(0xa77bdddd53dda68e), LL(0x2eaf17175c17b84b), LL(0x8e45474701470246), LL(0x211a9e9e429e84dc),
    LL(0x89d4caca0fca1ec5), LL(0x5a582d2db42d7599), LL(0x632ebfbfc6bf9179), LL(0x0e3f07071c07381b),
    LL(0x47acadad8ead0123), LL(0xb4b05a5a755aea2f), LL(0x1bef838336836cb5), LL(0x66b63333cc3385ff),
    LL(0xc65c636391633ff2), LL(0x041202020802100a), LL(0x4993aaaa92aa3938), LL(0xe2de7171d971afa8),
    LL(0x8dc6c8c807c80ecf), LL(0x32d119196419c87d), LL(0x923b494939497270), LL(0xaf5fd9d943d9869a),
    LL(0xf931f2f2eff2c31d), LL(0xdba8e3e3abe34b48), LL(0xb6b95b5b715be22a), LL(0x0dbc88881a883492),
    LL(0x293e9a9a529aa4c8), LL(0x4c0b262698262dbe), LL(0x64bf3232c8328dfa), LL(0x7d59b0b0fab0e94a),
    LL(0xcff2e9e983e91b6a), LL(0x1e770f0f3c0f7833), LL(0xb733d5d573d5e6a6), LL(0x1df480803a8074ba),
    LL(0x6127bebec2be997c), LL(0x87ebcdcd13cd26de), LL(0x68893434d034bde4), LL(0x903248483d487a75),
    LL(0xe354ffffdbffab24), LL(0xf48d7a7af57af78f), LL(0x3d6490907a90f4ea), LL(0xbe9d5f5f615fc23e),
    LL(0x403d202080201da0), LL(0xd00f6868bd6867d5), LL(0x34ca1a1a681ad072), LL(0x41b7aeae82ae192c),
    LL(0x757db4b4eab4c95e), LL(0xa8ce54544d549a19), LL(0x3b7f93937693ece5), LL(0x442f222288220daa),
    LL(0xc86364648d6407e9), LL(0xff2af1f1e3f1db12), LL(0xe6cc7373d173bfa2), LL(0x248212124812905a),
    LL(0x807a40401d403a5d), LL(0x1048080820084028), LL(0x9b95c3c32bc356e8), LL(0xc5dfecec97ec337b),
    LL(0xab4ddbdb4bdb9690), LL(0x5fc0a1a1bea1611f), LL(0x07918d8d0e8d1c83), LL(0x7ac83d3df43df5c9),
    LL(0x335b97976697ccf1), LL(0x0000000000000000), LL(0x83f9cfcf1bcf36d4), LL(0x566e2b2bac2b4587),
    LL(0xece17676c57697b3), LL(0x19e68282328264b0), LL(0xb128d6d67fd6fea9), LL(0x36c31b1b6c1bd877),
    LL(0x7774b5b5eeb5c15b), LL(0x43beafaf86af1129), LL(0xd41d6a6ab56a77df), LL(0xa0ea50505d50ba0d),
    LL(0x8a5745450945124c), LL(0xfb38f3f3ebf3cb18), LL(0x60ad3030c0309df0), LL(0xc3c4efef9bef2b74),
    LL(0x7eda3f3ffc3fe5c3), LL(0xaac755554955921c), LL(0x59dba2a2b2a27910), LL(0xc9e9eaea8fea0365),
    LL(0xca6a656589650fec), LL(0x6903babad2bab968), LL(0x5e4a2f2fbc2f6593), LL(0x9d8ec0c027c04ee7),
    LL(0xa160dede5fdebe81), LL(0x38fc1c1c701ce06c), LL(0xe746fdfdd3fdbb2e), LL(0x9a1f4d4d294d5264),
    LL(0x397692927292e4e0), LL(0xeafa7575c9758fbc), LL(0x0c3606061806301e), LL(0x09ae8a8a128a2498),
    LL(0x794bb2b2f2b2f940), LL(0xd185e6e6bfe66359), LL(0x1c7e0e0e380e7036), LL(0x3ee71f1f7c1ff863),
    LL(0xc4556262956237f7), LL(0xb53ad4d477d4eea3), LL(0x4d81a8a89aa82932), LL(0x315296966296c4f4),
    LL(0xef62f9f9c3f99b3a), LL(0x97a3c5c533c566f6), LL(0x4a102525942535b1), LL(0xb2ab59597959f220),
    LL(0x15d084842a8454ae), LL(0xe4c57272d572b7a7), LL(0x72ec3939e439d5dd), LL(0x98164c4c2d4c5a61),
    LL(0xbc945e5e655eca3b), LL(0xf09f7878fd78e785), LL(0x70e53838e038ddd8), LL(0x05988c8c0a8c1486),
    LL(0xbf17d1d163d1c6b2), LL(0x57e4a5a5aea5410b), LL(0xd9a1e2e2afe2434d), LL(0xc24e616199612ff8),
    LL(0x7b42b3b3f6b3f145), LL(0x42342121842115a5), LL(0x25089c9c4a9c94d6), LL(0x3cee1e1e781ef066),
    LL(0x8661434311432252), LL(0x93b1c7c73bc776fc), LL(0xe54ffcfcd7fcb32b), LL(0x0824040410042014),
    LL(0xa2e351515951b208), LL(0x2f2599995e99bcc7), LL(0xda226d6da96d4fc4), LL(0x1a650d0d340d6839),
    LL(0xe979fafacffa8335), LL(0xa369dfdf5bdfb684), LL(0xfca97e7ee57ed79b), LL(0x4819242490243db4),
    LL(0x76fe3b3bec3bc5d7), LL(0x4b9aabab96ab313d), LL(0x81f0cece1fce3ed1), LL(0x2299111144118855),
    LL(0x03838f8f068f0c89), LL(0x9c044e4e254e4a6b), LL(0x7366b7b7e6b7d151), LL(0xcbe0ebeb8beb0b60),
    LL(0x78c13c3cf03cfdcc), LL(0x1ffd81813e817cbf), LL(0x354094946a94d4fe), LL(0xf31cf7f7fbf7eb0c),
    LL(0x6f18b9b9deb9a167), LL(0x268b13134c13985f), LL(0x58512c2cb02c7d9c), LL(0xbb05d3d36bd3d6b8),
    LL(0xd38ce7e7bbe76b5c), LL(0xdc396e6ea56e57cb), LL(0x95aac4c437c46ef3), LL(0x061b03030c03180f),
    LL(0xacdc565645568a13), LL(0x885e44440d441a49), LL(0xfea07f7fe17fdf9e), LL(0x4f88a9a99ea92137),
    LL(0x54672a2aa82a4d82), LL(0x6b0abbbbd6bbb16d), LL(0x9f87c1c123c146e2), LL(0xa6f153535153a202),
    LL(0xa572dcdc57dcae8b), LL(0x16530b0b2c0b5827), LL(0x27019d9d4e9d9cd3), LL(0xd82b6c6cad6c47c1),
    LL(0x62a43131c43195f5), LL(0xe8f37474cd7487b9), LL(0xf115f6f6fff6e309), LL(0x8c4c464605460a43),
    LL(0x45a5acac8aac0926), LL(0x0fb589891e893c97), LL(0x28b414145014a044), LL(0xdfbae1e1a3e15b42),
    LL(0x2ca616165816b04e), LL(0x74f73a3ae83acdd2), LL(0xd2066969b9696fd0), LL(0x124109092409482d),
    LL(0xe0d77070dd70a7ad), LL(0x716fb6b6e2b6d954), LL(0xbd1ed0d067d0ceb7), LL(0xc7d6eded93ed3b7e),
    LL(0x85e2cccc17cc2edb), LL(0x8468424215422a57), LL(0x2d2c98985a98b4c2), LL(0x55eda4a4aaa4490e),
    LL(0x50752828a0285d88), LL(0xb8865c5c6d5cda31), LL(0xed6bf8f8c7f8933f), LL(0x11c28686228644a4),
};

__constant__ static const u64 C3[256] = {
    LL(0x7830d818186018c0), LL(0xaf462623238c2305), LL(0xf991b8c6c63fc67e), LL(0x6fcdfbe8e887e813),
    LL(0xa113cb878726874c), LL(0x626d11b8b8dab8a9), LL(0x0502090101040108), LL(0x6e9e0d4f4f214f42),
    LL(0xee6c9b3636d836ad), LL(0x0451ffa6a6a2a659), LL(0xbdb90cd2d26fd2de), LL(0x06f70ef5f5f3f5fb),
    LL(0x80f2967979f979ef), LL(0xcede306f6fa16f5f), LL(0xef3f6d91917e91fc), LL(0x07a4f852525552aa),
    LL(0xfdc04760609d6027), LL(0x766535bcbccabc89), LL(0xcd2b379b9b569bac), LL(0x8c018a8e8e028e04),
    LL(0x155bd2a3a3b6a371), LL(0x3c186c0c0c300c60), LL(0x8af6847b7bf17bff), LL(0xe16a803535d435b5),
    LL(0x693af51d1d741de8), LL(0x47ddb3e0e0a7e053), LL(0xacb321d7d77bd7f6), LL(0xed999cc2c22fc25e),
    LL(0x965c432e2eb82e6d), LL(0x7a96294b4b314b62), LL(0x21e15dfefedffea3), LL(0x16aed55757415782),
    LL(0x412abd15155415a8), LL(0xb6eee87777c1779f), LL(0xeb6e923737dc37a5), LL(0x56d79ee5e5b3e57b),
    LL(0xd923139f9f469f8c), LL(0x17fd23f0f0e7f0d3), LL(0x7f94204a4a354a6a), LL(0x95a944dada4fda9e),
    LL(0x25b0a258587d58fa), LL(0xca8fcfc9c903c906), LL(0x8d527c2929a42955), LL(0x22145a0a0a280a50),
    LL(0x4f7f50b1b1feb1e1), LL(0x1a5dc9a0a0baa069), LL(0xdad6146b6bb16b7f), LL(0xab17d985852e855c),
    LL(0x73673cbdbdcebd81), LL(0x34ba8f5d5d695dd2), LL(0x5020901010401080), LL(0x03f507f4f4f7f4f3),
    LL(0xc08bddcbcb0bcb16), LL(0xc67cd33e3ef83eed), LL(0x110a2d0505140528), LL(0xe6ce78676781671f),
    LL(0x53d597e4e4b7e473), LL(0xbb4e0227279c2725), LL(0x5882734141194132), LL(0x9d0ba78b8b168b2c),
    LL(0x0153f6a7a7a6a751), LL(0x94fab27d7de97dcf), LL(0xfb374995956e95dc), LL(0x9fad56d8d847d88e),
    LL(0x30eb70fbfbcbfb8b), LL(0x71c1cdeeee9fee23), LL(0x91f8bb7c7ced7cc7), LL(0xe3cc716666856617),
    LL(0x8ea77bdddd53dda6), LL(0x4b2eaf17175c17b8), LL(0x468e454747014702), LL(0xdc211a9e9e429e84),
    LL(0xc589d4caca0fca1e), LL(0x995a582d2db42d75), LL(0x79632ebfbfc6bf91), LL(0x1b0e3f07071c0738),
    LL(0x2347acadad8ead01), LL(0x2fb4b05a5a755aea), LL(0xb51bef838336836c), LL(0xff66b63333cc3385),
    LL(0xf2c65c636391633f), LL(0x0a04120202080210), LL(0x384993aaaa92aa39), LL(0xa8e2de7171d971af),
    LL(0xcf8dc6c8c807c80e), LL(0x7d32d119196419c8), LL(0x70923b4949394972), LL(0x9aaf5fd9d943d986),
    LL(0x1df931f2f2eff2c3), LL(0x48dba8e3e3abe34b), LL(0x2ab6b95b5b715be2), LL(0x920dbc88881a8834),
    LL(0xc8293e9a9a529aa4), LL(0xbe4c0b262698262d), LL(0xfa64bf3232c8328d), LL(0x4a7d59b0b0fab0e9),
    LL(0x6acff2e9e983e91b), LL(0x331e770f0f3c0f78), LL(0xa6b733d5d573d5e6), LL(0xba1df480803a8074),
    LL(0x7c6127bebec2be99), LL(0xde87ebcdcd13cd26), LL(0xe468893434d034bd), LL(0x75903248483d487a),
    LL(0x24e354ffffdbffab), LL(0x8ff48d7a7af57af7), LL(0xea3d6490907a90f4), LL(0x3ebe9d5f5f615fc2),
    LL(0xa0403d202080201d), LL(0xd5d00f6868bd6867), LL(0x7234ca1a1a681ad0), LL(0x2c41b7aeae82ae19),
    LL(0x5e757db4b4eab4c9), LL(0x19a8ce54544d549a), LL(0xe53b7f93937693ec), LL(0xaa442f222288220d),
    LL(0xe9c86364648d6407), LL(0x12ff2af1f1e3f1db), LL(0xa2e6cc7373d173bf), LL(0x5a24821212481290),
    LL(0x5d807a40401d403a), LL(0x2810480808200840), LL(0xe89b95c3c32bc356), LL(0x7bc5dfecec97ec33),
    LL(0x90ab4ddbdb4bdb96), LL(0x1f5fc0a1a1bea161), LL(0x8307918d8d0e8d1c), LL(0xc97ac83d3df43df5),
    LL(0xf1335b97976697cc), LL(0x0000000000000000), LL(0xd483f9cfcf1bcf36), LL(0x87566e2b2bac2b45),
    LL(0xb3ece17676c57697), LL(0xb019e68282328264), LL(0xa9b128d6d67fd6fe), LL(0x7736c31b1b6c1bd8),
    LL(0x5b7774b5b5eeb5c1), LL(0x2943beafaf86af11), LL(0xdfd41d6a6ab56a77), LL(0x0da0ea50505d50ba),
    LL(0x4c8a574545094512), LL(0x18fb38f3f3ebf3cb), LL(0xf060ad3030c0309d), LL(0x74c3c4efef9bef2b),
    LL(0xc37eda3f3ffc3fe5), LL(0x1caac75555495592), LL(0x1059dba2a2b2a279), LL(0x65c9e9eaea8fea03),
    LL(0xecca6a656589650f), LL(0x686903babad2bab9), LL(0x935e4a2f2fbc2f65), LL(0xe79d8ec0c027c04e),
    LL(0x81a160dede5fdebe), LL(0x6c38fc1c1c701ce0), LL(0x2ee746fdfdd3fdbb), LL(0x649a1f4d4d294d52),
    LL(0xe0397692927292e4), LL(0xbceafa7575c9758f), LL(0x1e0c360606180630), LL(0x9809ae8a8a128a24),
    LL(0x40794bb2b2f2b2f9), LL(0x59d185e6e6bfe663), LL(0x361c7e0e0e380e70), LL(0x633ee71f1f7c1ff8),
    LL(0xf7c4556262956237), LL(0xa3b53ad4d477d4ee), LL(0x324d81a8a89aa829), LL(0xf4315296966296c4),
    LL(0x3aef62f9f9c3f99b), LL(0xf697a3c5c533c566), LL(0xb14a102525942535), LL(0x20b2ab59597959f2),
    LL(0xae15d084842a8454), LL(0xa7e4c57272d572b7), LL(0xdd72ec3939e439d5), LL(0x6198164c4c2d4c5a),
    LL(0x3bbc945e5e655eca), LL(0x85f09f7878fd78e7), LL(0xd870e53838e038dd), LL(0x8605988c8c0a8c14),
    LL(0xb2bf17d1d163d1c6), LL(0x0b57e4a5a5aea541), LL(0x4dd9a1e2e2afe243), LL(0xf8c24e616199612f),
    LL(0x457b42b3b3f6b3f1), LL(0xa542342121842115), LL(0xd625089c9c4a9c94), LL(0x663cee1e1e781ef0),
    LL(0x5286614343114322), LL(0xfc93b1c7c73bc776), LL(0x2be54ffcfcd7fcb3), LL(0x1408240404100420),
    LL(0x08a2e351515951b2), LL(0xc72f2599995e99bc), LL(0xc4da226d6da96d4f), LL(0x391a650d0d340d68),
    LL(0x35e979fafacffa83), LL(0x84a369dfdf5bdfb6), LL(0x9bfca97e7ee57ed7), LL(0xb44819242490243d),
    LL(0xd776fe3b3bec3bc5), LL(0x3d4b9aabab96ab31), LL(0xd181f0cece1fce3e), LL(0x5522991111441188),
    LL(0x8903838f8f068f0c), LL(0x6b9c044e4e254e4a), LL(0x517366b7b7e6b7d1), LL(0x60cbe0ebeb8beb0b),
    LL(0xcc78c13c3cf03cfd), LL(0xbf1ffd81813e817c), LL(0xfe354094946a94d4), LL(0x0cf31cf7f7fbf7eb),
    LL(0x676f18b9b9deb9a1), LL(0x5f268b13134c1398), LL(0x9c58512c2cb02c7d), LL(0xb8bb05d3d36bd3d6),
    LL(0x5cd38ce7e7bbe76b), LL(0xcbdc396e6ea56e57), LL(0xf395aac4c437c46e), LL(0x0f061b03030c0318),
    LL(0x13acdc565645568a), LL(0x49885e44440d441a), LL(0x9efea07f7fe17fdf), LL(0x374f88a9a99ea921),
    LL(0x8254672a2aa82a4d), LL(0x6d6b0abbbbd6bbb1), LL(0xe29f87c1c123c146), LL(0x02a6f153535153a2),
    LL(0x8ba572dcdc57dcae), LL(0x2716530b0b2c0b58), LL(0xd327019d9d4e9d9c), LL(0xc1d82b6c6cad6c47),
    LL(0xf562a43131c43195), LL(0xb9e8f37474cd7487), LL(0x09f115f6f6fff6e3), LL(0x438c4c464605460a),
    LL(0x2645a5acac8aac09), LL(0x970fb589891e893c), LL(0x4428b414145014a0), LL(0x42dfbae1e1a3e15b),
    LL(0x4e2ca616165816b0), LL(0xd274f73a3ae83acd), LL(0xd0d2066969b9696f), LL(0x2d12410909240948),
    LL(0xade0d77070dd70a7), LL(0x54716fb6b6e2b6d9), LL(0xb7bd1ed0d067d0ce), LL(0x7ec7d6eded93ed3b),
    LL(0xdb85e2cccc17cc2e), LL(0x578468424215422a), LL(0xc22d2c98985a98b4), LL(0x0e55eda4a4aaa449),
    LL(0x8850752828a0285d), LL(0x31b8865c5c6d5cda), LL(0x3fed6bf8f8c7f893), LL(0xa411c28686228644),
};

__constant__ static const u64 C4[256] = {
    LL(0xc07830d818186018), LL(0x05af462623238c23), LL(0x7ef991b8c6c63fc6), LL(0x136fcdfbe8e887e8),
    LL(0x4ca113cb87872687), LL(0xa9626d11b8b8dab8), LL(0x0805020901010401), LL(0x426e9e0d4f4f214f),
    LL(0xadee6c9b3636d836), LL(0x590451ffa6a6a2a6), LL(0xdebdb90cd2d26fd2), LL(0xfb06f70ef5f5f3f5),
    LL(0xef80f2967979f979), LL(0x5fcede306f6fa16f), LL(0xfcef3f6d91917e91), LL(0xaa07a4f852525552),
    LL(0x27fdc04760609d60), LL(0x89766535bcbccabc), LL(0xaccd2b379b9b569b), LL(0x048c018a8e8e028e),
    LL(0x71155bd2a3a3b6a3), LL(0x603c186c0c0c300c), LL(0xff8af6847b7bf17b), LL(0xb5e16a803535d435),
    LL(0xe8693af51d1d741d), LL(0x5347ddb3e0e0a7e0), LL(0xf6acb321d7d77bd7), LL(0x5eed999cc2c22fc2),
    LL(0x6d965c432e2eb82e), LL(0x627a96294b4b314b), LL(0xa321e15dfefedffe), LL(0x8216aed557574157),
    LL(0xa8412abd15155415), LL(0x9fb6eee87777c177), LL(0xa5eb6e923737dc37), LL(0x7b56d79ee5e5b3e5),
    LL(0x8cd923139f9f469f), LL(0xd317fd23f0f0e7f0), LL(0x6a7f94204a4a354a), LL(0x9e95a944dada4fda),
    LL(0xfa25b0a258587d58), LL(0x06ca8fcfc9c903c9), LL(0x558d527c2929a429), LL(0x5022145a0a0a280a),
    LL(0xe14f7f50b1b1feb1), LL(0x691a5dc9a0a0baa0), LL(0x7fdad6146b6bb16b), LL(0x5cab17d985852e85),
    LL(0x8173673cbdbdcebd), LL(0xd234ba8f5d5d695d), LL(0x8050209010104010), LL(0xf303f507f4f4f7f4),
    LL(0x16c08bddcbcb0bcb), LL(0xedc67cd33e3ef83e), LL(0x28110a2d05051405), LL(0x1fe6ce7867678167),
    LL(0x7353d597e4e4b7e4), LL(0x25bb4e0227279c27), LL(0x3258827341411941), LL(0x2c9d0ba78b8b168b),
    LL(0x510153f6a7a7a6a7), LL(0xcf94fab27d7de97d), LL(0xdcfb374995956e95), LL(0x8e9fad56d8d847d8),
    LL(0x8b30eb70fbfbcbfb), LL(0x2371c1cdeeee9fee), LL(0xc791f8bb7c7ced7c), LL(0x17e3cc7166668566),
    LL(0xa68ea77bdddd53dd), LL(0xb84b2eaf17175c17), LL(0x02468e4547470147), LL(0x84dc211a9e9e429e),
    LL(0x1ec589d4caca0fca), LL(0x75995a582d2db42d), LL(0x9179632ebfbfc6bf), LL(0x381b0e3f07071c07),
    LL(0x012347acadad8ead), LL(0xea2fb4b05a5a755a), LL(0x6cb51bef83833683), LL(0x85ff66b63333cc33),
    LL(0x3ff2c65c63639163), LL(0x100a041202020802), LL(0x39384993aaaa92aa), LL(0xafa8e2de7171d971),
    LL(0x0ecf8dc6c8c807c8), LL(0xc87d32d119196419), LL(0x7270923b49493949), LL(0x869aaf5fd9d943d9),
    LL(0xc31df931f2f2eff2), LL(0x4b48dba8e3e3abe3), LL(0xe22ab6b95b5b715b), LL(0x34920dbc88881a88),
    LL(0xa4c8293e9a9a529a), LL(0x2dbe4c0b26269826), LL(0x8dfa64bf3232c832), LL(0xe94a7d59b0b0fab0),
    LL(0x1b6acff2e9e983e9), LL(0x78331e770f0f3c0f), LL(0xe6a6b733d5d573d5), LL(0x74ba1df480803a80),
    LL(0x997c6127bebec2be), LL(0x26de87ebcdcd13cd), LL(0xbde468893434d034), LL(0x7a75903248483d48),
    LL(0xab24e354ffffdbff), LL(0xf78ff48d7a7af57a), LL(0xf4ea3d6490907a90), LL(0xc23ebe9d5f5f615f),
    LL(0x1da0403d20208020), LL(0x67d5d00f6868bd68), LL(0xd07234ca1a1a681a), LL(0x192c41b7aeae82ae),
    LL(0xc95e757db4b4eab4), LL(0x9a19a8ce54544d54), LL(0xece53b7f93937693), LL(0x0daa442f22228822),
    LL(0x07e9c86364648d64), LL(0xdb12ff2af1f1e3f1), LL(0xbfa2e6cc7373d173), LL(0x905a248212124812),
    LL(0x3a5d807a40401d40), LL(0x4028104808082008), LL(0x56e89b95c3c32bc3), LL(0x337bc5dfecec97ec),
    LL(0x9690ab4ddbdb4bdb), LL(0x611f5fc0a1a1bea1), LL(0x1c8307918d8d0e8d), LL(0xf5c97ac83d3df43d),
    LL(0xccf1335b97976697), LL(0x0000000000000000), LL(0x36d483f9cfcf1bcf), LL(0x4587566e2b2bac2b),
    LL(0x97b3ece17676c576), LL(0x64b019e682823282), LL(0xfea9b128d6d67fd6), LL(0xd87736c31b1b6c1b),
    LL(0xc15b7774b5b5eeb5), LL(0x112943beafaf86af), LL(0x77dfd41d6a6ab56a), LL(0xba0da0ea50505d50),
    LL(0x124c8a5745450945), LL(0xcb18fb38f3f3ebf3), LL(0x9df060ad3030c030), LL(0x2b74c3c4efef9bef),
    LL(0xe5c37eda3f3ffc3f), LL(0x921caac755554955), LL(0x791059dba2a2b2a2), LL(0x0365c9e9eaea8fea),
    LL(0x0fecca6a65658965), LL(0xb9686903babad2ba), LL(0x65935e4a2f2fbc2f), LL(0x4ee79d8ec0c027c0),
    LL(0xbe81a160dede5fde), LL(0xe06c38fc1c1c701c), LL(0xbb2ee746fdfdd3fd), LL(0x52649a1f4d4d294d),
    LL(0xe4e0397692927292), LL(0x8fbceafa7575c975), LL(0x301e0c3606061806), LL(0x249809ae8a8a128a),
    LL(0xf940794bb2b2f2b2), LL(0x6359d185e6e6bfe6), LL(0x70361c7e0e0e380e), LL(0xf8633ee71f1f7c1f),
    LL(0x37f7c45562629562), LL(0xeea3b53ad4d477d4), LL(0x29324d81a8a89aa8), LL(0xc4f4315296966296),
    LL(0x9b3aef62f9f9c3f9), LL(0x66f697a3c5c533c5), LL(0x35b14a1025259425), LL(0xf220b2ab59597959),
    LL(0x54ae15d084842a84), LL(0xb7a7e4c57272d572), LL(0xd5dd72ec3939e439), LL(0x5a6198164c4c2d4c),
    LL(0xca3bbc945e5e655e), LL(0xe785f09f7878fd78), LL(0xddd870e53838e038), LL(0x148605988c8c0a8c),
    LL(0xc6b2bf17d1d163d1), LL(0x410b57e4a5a5aea5), LL(0x434dd9a1e2e2afe2), LL(0x2ff8c24e61619961),
    LL(0xf1457b42b3b3f6b3), LL(0x15a5423421218421), LL(0x94d625089c9c4a9c), LL(0xf0663cee1e1e781e),
    LL(0x2252866143431143), LL(0x76fc93b1c7c73bc7), LL(0xb32be54ffcfcd7fc), LL(0x2014082404041004),
    LL(0xb208a2e351515951), LL(0xbcc72f2599995e99), LL(0x4fc4da226d6da96d), LL(0x68391a650d0d340d),
    LL(0x8335e979fafacffa), LL(0xb684a369dfdf5bdf), LL(0xd79bfca97e7ee57e), LL(0x3db4481924249024),
    LL(0xc5d776fe3b3bec3b), LL(0x313d4b9aabab96ab), LL(0x3ed181f0cece1fce), LL(0x8855229911114411),
    LL(0x0c8903838f8f068f), LL(0x4a6b9c044e4e254e), LL(0xd1517366b7b7e6b7), LL(0x0b60cbe0ebeb8beb),
    LL(0xfdcc78c13c3cf03c), LL(0x7cbf1ffd81813e81), LL(0xd4fe354094946a94), LL(0xeb0cf31cf7f7fbf7),
    LL(0xa1676f18b9b9deb9), LL(0x985f268b13134c13), LL(0x7d9c58512c2cb02c), LL(0xd6b8bb05d3d36bd3),
    LL(0x6b5cd38ce7e7bbe7), LL(0x57cbdc396e6ea56e), LL(0x6ef395aac4c437c4), LL(0x180f061b03030c03),
    LL(0x8a13acdc56564556), LL(0x1a49885e44440d44), LL(0xdf9efea07f7fe17f), LL(0x21374f88a9a99ea9),
    LL(0x4d8254672a2aa82a), LL(0xb16d6b0abbbbd6bb), LL(0x46e29f87c1c123c1), LL(0xa202a6f153535153),
    LL(0xae8ba572dcdc57dc), LL(0x582716530b0b2c0b), LL(0x9cd327019d9d4e9d), LL(0x47c1d82b6c6cad6c),
    LL(0x95f562a43131c431), LL(0x87b9e8f37474cd74), LL(0xe309f115f6f6fff6), LL(0x0a438c4c46460546),
    LL(0x092645a5acac8aac), LL(0x3c970fb589891e89), LL(0xa04428b414145014), LL(0x5b42dfbae1e1a3e1),
    LL(0xb04e2ca616165816), LL(0xcdd274f73a3ae83a), LL(0x6fd0d2066969b969), LL(0x482d124109092409),
    LL(0xa7ade0d77070dd70), LL(0xd954716fb6b6e2b6), LL(0xceb7bd1ed0d067d0), LL(0x3b7ec7d6eded93ed),
    LL(0x2edb85e2cccc17cc), LL(0x2a57846842421542), LL(0xb4c22d2c98985a98), LL(0x490e55eda4a4aaa4),
    LL(0x5d8850752828a028), LL(0xda31b8865c5c6d5c), LL(0x933fed6bf8f8c7f8), LL(0x44a411c286862286),
};

__constant__ static const u64 C5[256] = {
    LL(0x18c07830d8181860), LL(0x2305af462623238c), LL(0xc67ef991b8c6c63f), LL(0xe8136fcdfbe8e887),
    LL(0x874ca113cb878726), LL(0xb8a9626d11b8b8da), LL(0x0108050209010104), LL(0x4f426e9e0d4f4f21),
    LL(0x36adee6c9b3636d8), LL(0xa6590451ffa6a6a2), LL(0xd2debdb90cd2d26f), LL(0xf5fb06f70ef5f5f3),
    LL(0x79ef80f2967979f9), LL(0x6f5fcede306f6fa1), LL(0x91fcef3f6d91917e), LL(0x52aa07a4f8525255),
    LL(0x6027fdc04760609d), LL(0xbc89766535bcbcca), LL(0x9baccd2b379b9b56), LL(0x8e048c018a8e8e02),
    LL(0xa371155bd2a3a3b6), LL(0x0c603c186c0c0c30), LL(0x7bff8af6847b7bf1), LL(0x35b5e16a803535d4),
    LL(0x1de8693af51d1d74), LL(0xe05347ddb3e0e0a7), LL(0xd7f6acb321d7d77b), LL(0xc25eed999cc2c22f),
    LL(0x2e6d965c432e2eb8), LL(0x4b627a96294b4b31), LL(0xfea321e15dfefedf), LL(0x578216aed5575741),
    LL(0x15a8412abd151554), LL(0x779fb6eee87777c1), LL(0x37a5eb6e923737dc), LL(0xe57b56d79ee5e5b3),
    LL(0x9f8cd923139f9f46), LL(0xf0d317fd23f0f0e7), LL(0x4a6a7f94204a4a35), LL(0xda9e95a944dada4f),
    LL(0x58fa25b0a258587d), LL(0xc906ca8fcfc9c903), LL(0x29558d527c2929a4), LL(0x0a5022145a0a0a28),
    LL(0xb1e14f7f50b1b1fe), LL(0xa0691a5dc9a0a0ba), LL(0x6b7fdad6146b6bb1), LL(0x855cab17d985852e),
    LL(0xbd8173673cbdbdce), LL(0x5dd234ba8f5d5d69), LL(0x1080502090101040), LL(0xf4f303f507f4f4f7),
    LL(0xcb16c08bddcbcb0b), LL(0x3eedc67cd33e3ef8), LL(0x0528110a2d050514), LL(0x671fe6ce78676781),
    LL(0xe47353d597e4e4b7), LL(0x2725bb4e0227279c), LL(0x4132588273414119), LL(0x8b2c9d0ba78b8b16),
    LL(0xa7510153f6a7a7a6), LL(0x7dcf94fab27d7de9), LL(0x95dcfb374995956e), LL(0xd88e9fad56d8d847),
    LL(0xfb8b30eb70fbfbcb), LL(0xee2371c1cdeeee9f), LL(0x7cc791f8bb7c7ced), LL(0x6617e3cc71666685),
    LL(0xdda68ea77bdddd53), LL(0x17b84b2eaf17175c), LL(0x4702468e45474701), LL(0x9e84dc211a9e9e42),
    LL(0xca1ec589d4caca0f), LL(0x2d75995a582d2db4), LL(0xbf9179632ebfbfc6), LL(0x07381b0e3f07071c),
    LL(0xad012347acadad8e), LL(0x5aea2fb4b05a5a75), LL(0x836cb51bef838336), LL(0x3385ff66b63333cc),
    LL(0x633ff2c65c636391), LL(0x02100a0412020208), LL(0xaa39384993aaaa92), LL(0x71afa8e2de7171d9),
    LL(0xc80ecf8dc6c8c807), LL(0x19c87d32d1191964), LL(0x497270923b494939), LL(0xd9869aaf5fd9d943),
    LL(0xf2c31df931f2f2ef), LL(0xe34b48dba8e3e3ab), LL(0x5be22ab6b95b5b71), LL(0x8834920dbc88881a),
    LL(0x9aa4c8293e9a9a52), LL(0x262dbe4c0b262698), LL(0x328dfa64bf3232c8), LL(0xb0e94a7d59b0b0fa),
    LL(0xe91b6acff2e9e983), LL(0x0f78331e770f0f3c), LL(0xd5e6a6b733d5d573), LL(0x8074ba1df480803a),
    LL(0xbe997c6127bebec2), LL(0xcd26de87ebcdcd13), LL(0x34bde468893434d0), LL(0x487a75903248483d),
    LL(0xffab24e354ffffdb), LL(0x7af78ff48d7a7af5), LL(0x90f4ea3d6490907a), LL(0x5fc23ebe9d5f5f61),
    LL(0x201da0403d202080), LL(0x6867d5d00f6868bd), LL(0x1ad07234ca1a1a68), LL(0xae192c41b7aeae82),
    LL(0xb4c95e757db4b4ea), LL(0x549a19a8ce54544d), LL(0x93ece53b7f939376), LL(0x220daa442f222288),
    LL(0x6407e9c86364648d), LL(0xf1db12ff2af1f1e3), LL(0x73bfa2e6cc7373d1), LL(0x12905a2482121248),
    LL(0x403a5d807a40401d), LL(0x0840281048080820), LL(0xc356e89b95c3c32b), LL(0xec337bc5dfecec97),
    LL(0xdb9690ab4ddbdb4b), LL(0xa1611f5fc0a1a1be), LL(0x8d1c8307918d8d0e), LL(0x3df5c97ac83d3df4),
    LL(0x97ccf1335b979766), LL(0x0000000000000000), LL(0xcf36d483f9cfcf1b), LL(0x2b4587566e2b2bac),
    LL(0x7697b3ece17676c5), LL(0x8264b019e6828232), LL(0xd6fea9b128d6d67f), LL(0x1bd87736c31b1b6c),
    LL(0xb5c15b7774b5b5ee), LL(0xaf112943beafaf86), LL(0x6a77dfd41d6a6ab5), LL(0x50ba0da0ea50505d),
    LL(0x45124c8a57454509), LL(0xf3cb18fb38f3f3eb), LL(0x309df060ad3030c0), LL(0xef2b74c3c4efef9b),
    LL(0x3fe5c37eda3f3ffc), LL(0x55921caac7555549), LL(0xa2791059dba2a2b2), LL(0xea0365c9e9eaea8f),
    LL(0x650fecca6a656589), LL(0xbab9686903babad2), LL(0x2f65935e4a2f2fbc), LL(0xc04ee79d8ec0c027),
    LL(0xdebe81a160dede5f), LL(0x1ce06c38fc1c1c70), LL(0xfdbb2ee746fdfdd3), LL(0x4d52649a1f4d4d29),
    LL(0x92e4e03976929272), LL(0x758fbceafa7575c9), LL(0x06301e0c36060618), LL(0x8a249809ae8a8a12),
    LL(0xb2f940794bb2b2f2), LL(0xe66359d185e6e6bf), LL(0x0e70361c7e0e0e38), LL(0x1ff8633ee71f1f7c),
    LL(0x6237f7c455626295), LL(0xd4eea3b53ad4d477), LL(0xa829324d81a8a89a), LL(0x96c4f43152969662),
    LL(0xf99b3aef62f9f9c3), LL(0xc566f697a3c5c533), LL(0x2535b14a10252594), LL(0x59f220b2ab595979),
    LL(0x8454ae15d084842a), LL(0x72b7a7e4c57272d5), LL(0x39d5dd72ec3939e4), LL(0x4c5a6198164c4c2d),
    LL(0x5eca3bbc945e5e65), LL(0x78e785f09f7878fd), LL(0x38ddd870e53838e0), LL(0x8c148605988c8c0a),
    LL(0xd1c6b2bf17d1d163), LL(0xa5410b57e4a5a5ae), LL(0xe2434dd9a1e2e2af), LL(0x612ff8c24e616199),
    LL(0xb3f1457b42b3b3f6), LL(0x2115a54234212184), LL(0x9c94d625089c9c4a), LL(0x1ef0663cee1e1e78),
    LL(0x4322528661434311), LL(0xc776fc93b1c7c73b), LL(0xfcb32be54ffcfcd7), LL(0x0420140824040410),
    LL(0x51b208a2e3515159), LL(0x99bcc72f2599995e), LL(0x6d4fc4da226d6da9), LL(0x0d68391a650d0d34),
    LL(0xfa8335e979fafacf), LL(0xdfb684a369dfdf5b), LL(0x7ed79bfca97e7ee5), LL(0x243db44819242490),
    LL(0x3bc5d776fe3b3bec), LL(0xab313d4b9aabab96), LL(0xce3ed181f0cece1f), LL(0x1188552299111144),
    LL(0x8f0c8903838f8f06), LL(0x4e4a6b9c044e4e25), LL(0xb7d1517366b7b7e6), LL(0xeb0b60cbe0ebeb8b),
    LL(0x3cfdcc78c13c3cf0), LL(0x817cbf1ffd81813e), LL(0x94d4fe354094946a), LL(0xf7eb0cf31cf7f7fb),
    LL(0xb9a1676f18b9b9de), LL(0x13985f268b13134c), LL(0x2c7d9c58512c2cb0), LL(0xd3d6b8bb05d3d36b),
    LL(0xe76b5cd38ce7e7bb), LL(0x6e57cbdc396e6ea5), LL(0xc46ef395aac4c437), LL(0x03180f061b03030c),
    LL(0x568a13acdc565645), LL(0x441a49885e44440d), LL(0x7fdf9efea07f7fe1), LL(0xa921374f88a9a99e),
    LL(0x2a4d8254672a2aa8), LL(0xbbb16d6b0abbbbd6), LL(0xc146e29f87c1c123), LL(0x53a202a6f1535351),
    LL(0xdcae8ba572dcdc57), LL(0x0b582716530b0b2c), LL(0x9d9cd327019d9d4e), LL(0x6c47c1d82b6c6cad),
    LL(0x3195f562a43131c4), LL(0x7487b9e8f37474cd), LL(0xf6e309f115f6f6ff), LL(0x460a438c4c464605),
    LL(0xac092645a5acac8a), LL(0x893c970fb589891e), LL(0x14a04428b4141450), LL(0xe15b42dfbae1e1a3),
    LL(0x16b04e2ca6161658), LL(0x3acdd274f73a3ae8), LL(0x696fd0d2066969b9), LL(0x09482d1241090924),
    LL(0x70a7ade0d77070dd), LL(0xb6d954716fb6b6e2), LL(0xd0ceb7bd1ed0d067), LL(0xed3b7ec7d6eded93),
    LL(0xcc2edb85e2cccc17), LL(0x422a578468424215), LL(0x98b4c22d2c98985a), LL(0xa4490e55eda4a4aa),
    LL(0x285d8850752828a0), LL(0x5cda31b8865c5c6d), LL(0xf8933fed6bf8f8c7), LL(0x8644a411c2868622),
};

__constant__ static const u64 C6[256] = {
    LL(0x6018c07830d81818), LL(0x8c2305af46262323), LL(0x3fc67ef991b8c6c6), LL(0x87e8136fcdfbe8e8),
    LL(0x26874ca113cb8787), LL(0xdab8a9626d11b8b8), LL(0x0401080502090101), LL(0x214f426e9e0d4f4f),
    LL(0xd836adee6c9b3636), LL(0xa2a6590451ffa6a6), LL(0x6fd2debdb90cd2d2), LL(0xf3f5fb06f70ef5f5),
    LL(0xf979ef80f2967979), LL(0xa16f5fcede306f6f), LL(0x7e91fcef3f6d9191), LL(0x5552aa07a4f85252),
    LL(0x9d6027fdc0476060), LL(0xcabc89766535bcbc), LL(0x569baccd2b379b9b), LL(0x028e048c018a8e8e),
    LL(0xb6a371155bd2a3a3), LL(0x300c603c186c0c0c), LL(0xf17bff8af6847b7b), LL(0xd435b5e16a803535),
    LL(0x741de8693af51d1d), LL(0xa7e05347ddb3e0e0), LL(0x7bd7f6acb321d7d7), LL(0x2fc25eed999cc2c2),
    LL(0xb82e6d965c432e2e), LL(0x314b627a96294b4b), LL(0xdffea321e15dfefe), LL(0x41578216aed55757),
    LL(0x5415a8412abd1515), LL(0xc1779fb6eee87777), LL(0xdc37a5eb6e923737), LL(0xb3e57b56d79ee5e5),
    LL(0x469f8cd923139f9f), LL(0xe7f0d317fd23f0f0), LL(0x354a6a7f94204a4a), LL(0x4fda9e95a944dada),
    LL(0x7d58fa25b0a25858), LL(0x03c906ca8fcfc9c9), LL(0xa429558d527c2929), LL(0x280a5022145a0a0a),
    LL(0xfeb1e14f7f50b1b1), LL(0xbaa0691a5dc9a0a0), LL(0xb16b7fdad6146b6b), LL(0x2e855cab17d98585),
    LL(0xcebd8173673cbdbd), LL(0x695dd234ba8f5d5d), LL(0x4010805020901010), LL(0xf7f4f303f507f4f4),
    LL(0x0bcb16c08bddcbcb), LL(0xf83eedc67cd33e3e), LL(0x140528110a2d0505), LL(0x81671fe6ce786767),
    LL(0xb7e47353d597e4e4), LL(0x9c2725bb4e022727), LL(0x1941325882734141), LL(0x168b2c9d0ba78b8b),
    LL(0xa6a7510153f6a7a7), LL(0xe97dcf94fab27d7d), LL(0x6e95dcfb37499595), LL(0x47d88e9fad56d8d8),
    LL(0xcbfb8b30eb70fbfb), LL(0x9fee2371c1cdeeee), LL(0xed7cc791f8bb7c7c), LL(0x856617e3cc716666),
    LL(0x53dda68ea77bdddd), LL(0x5c17b84b2eaf1717), LL(0x014702468e454747), LL(0x429e84dc211a9e9e),
    LL(0x0fca1ec589d4caca), LL(0xb42d75995a582d2d), LL(0xc6bf9179632ebfbf), LL(0x1c07381b0e3f0707),
    LL(0x8ead012347acadad), LL(0x755aea2fb4b05a5a), LL(0x36836cb51bef8383), LL(0xcc3385ff66b63333),
    LL(0x91633ff2c65c6363), LL(0x0802100a04120202), LL(0x92aa39384993aaaa), LL(0xd971afa8e2de7171),
    LL(0x07c80ecf8dc6c8c8), LL(0x6419c87d32d11919), LL(0x39497270923b4949), LL(0x43d9869aaf5fd9d9),
    LL(0xeff2c31df931f2f2), LL(0xabe34b48dba8e3e3), LL(0x715be22ab6b95b5b), LL(0x1a8834920dbc8888),
    LL(0x529aa4c8293e9a9a), LL(0x98262dbe4c0b2626), LL(0xc8328dfa64bf3232), LL(0xfab0e94a7d59b0b0),
    LL(0x83e91b6acff2e9e9), LL(0x3c0f78331e770f0f), LL(0x73d5e6a6b733d5d5), LL(0x3a8074ba1df48080),
    LL(0xc2be997c6127bebe), LL(0x13cd26de87ebcdcd), LL(0xd034bde468893434), LL(0x3d487a7590324848),
    LL(0xdbffab24e354ffff), LL(0xf57af78ff48d7a7a), LL(0x7a90f4ea3d649090), LL(0x615fc23ebe9d5f5f),
    LL(0x80201da0403d2020), LL(0xbd6867d5d00f6868), LL(0x681ad07234ca1a1a), LL(0x82ae192c41b7aeae),
    LL(0xeab4c95e757db4b4), LL(0x4d549a19a8ce5454), LL(0x7693ece53b7f9393), LL(0x88220daa442f2222),
    LL(0x8d6407e9c8636464), LL(0xe3f1db12ff2af1f1), LL(0xd173bfa2e6cc7373), LL(0x4812905a24821212),
    LL(0x1d403a5d807a4040), LL(0x2008402810480808), LL(0x2bc356e89b95c3c3), LL(0x97ec337bc5dfecec),
    LL(0x4bdb9690ab4ddbdb), LL(0xbea1611f5fc0a1a1), LL(0x0e8d1c8307918d8d), LL(0xf43df5c97ac83d3d),
    LL(0x6697ccf1335b9797), LL(0x0000000000000000), LL(0x1bcf36d483f9cfcf), LL(0xac2b4587566e2b2b),
    LL(0xc57697b3ece17676), LL(0x328264b019e68282), LL(0x7fd6fea9b128d6d6), LL(0x6c1bd87736c31b1b),
    LL(0xeeb5c15b7774b5b5), LL(0x86af112943beafaf), LL(0xb56a77dfd41d6a6a), LL(0x5d50ba0da0ea5050),
    LL(0x0945124c8a574545), LL(0xebf3cb18fb38f3f3), LL(0xc0309df060ad3030), LL(0x9bef2b74c3c4efef),
    LL(0xfc3fe5c37eda3f3f), LL(0x4955921caac75555), LL(0xb2a2791059dba2a2), LL(0x8fea0365c9e9eaea),
    LL(0x89650fecca6a6565), LL(0xd2bab9686903baba), LL(0xbc2f65935e4a2f2f), LL(0x27c04ee79d8ec0c0),
    LL(0x5fdebe81a160dede), LL(0x701ce06c38fc1c1c), LL(0xd3fdbb2ee746fdfd), LL(0x294d52649a1f4d4d),
    LL(0x7292e4e039769292), LL(0xc9758fbceafa7575), LL(0x1806301e0c360606), LL(0x128a249809ae8a8a),
    LL(0xf2b2f940794bb2b2), LL(0xbfe66359d185e6e6), LL(0x380e70361c7e0e0e), LL(0x7c1ff8633ee71f1f),
    LL(0x956237f7c4556262), LL(0x77d4eea3b53ad4d4), LL(0x9aa829324d81a8a8), LL(0x6296c4f431529696),
    LL(0xc3f99b3aef62f9f9), LL(0x33c566f697a3c5c5), LL(0x942535b14a102525), LL(0x7959f220b2ab5959),
    LL(0x2a8454ae15d08484), LL(0xd572b7a7e4c57272), LL(0xe439d5dd72ec3939), LL(0x2d4c5a6198164c4c),
    LL(0x655eca3bbc945e5e), LL(0xfd78e785f09f7878), LL(0xe038ddd870e53838), LL(0x0a8c148605988c8c),
    LL(0x63d1c6b2bf17d1d1), LL(0xaea5410b57e4a5a5), LL(0xafe2434dd9a1e2e2), LL(0x99612ff8c24e6161),
    LL(0xf6b3f1457b42b3b3), LL(0x842115a542342121), LL(0x4a9c94d625089c9c), LL(0x781ef0663cee1e1e),
    LL(0x1143225286614343), LL(0x3bc776fc93b1c7c7), LL(0xd7fcb32be54ffcfc), LL(0x1004201408240404),
    LL(0x5951b208a2e35151), LL(0x5e99bcc72f259999), LL(0xa96d4fc4da226d6d), LL(0x340d68391a650d0d),
    LL(0xcffa8335e979fafa), LL(0x5bdfb684a369dfdf), LL(0xe57ed79bfca97e7e), LL(0x90243db448192424),
    LL(0xec3bc5d776fe3b3b), LL(0x96ab313d4b9aabab), LL(0x1fce3ed181f0cece), LL(0x4411885522991111),
    LL(0x068f0c8903838f8f), LL(0x254e4a6b9c044e4e), LL(0xe6b7d1517366b7b7), LL(0x8beb0b60cbe0ebeb),
    LL(0xf03cfdcc78c13c3c), LL(0x3e817cbf1ffd8181), LL(0x6a94d4fe35409494), LL(0xfbf7eb0cf31cf7f7),
    LL(0xdeb9a1676f18b9b9), LL(0x4c13985f268b1313), LL(0xb02c7d9c58512c2c), LL(0x6bd3d6b8bb05d3d3),
    LL(0xbbe76b5cd38ce7e7), LL(0xa56e57cbdc396e6e), LL(0x37c46ef395aac4c4), LL(0x0c03180f061b0303),
    LL(0x45568a13acdc5656), LL(0x0d441a49885e4444), LL(0xe17fdf9efea07f7f), LL(0x9ea921374f88a9a9),
    LL(0xa82a4d8254672a2a), LL(0xd6bbb16d6b0abbbb), LL(0x23c146e29f87c1c1), LL(0x5153a202a6f15353),
    LL(0x57dcae8ba572dcdc), LL(0x2c0b582716530b0b), LL(0x4e9d9cd327019d9d), LL(0xad6c47c1d82b6c6c),
    LL(0xc43195f562a43131), LL(0xcd7487b9e8f37474), LL(0xfff6e309f115f6f6), LL(0x05460a438c4c4646),
    LL(0x8aac092645a5acac), LL(0x1e893c970fb58989), LL(0x5014a04428b41414), LL(0xa3e15b42dfbae1e1),
    LL(0x5816b04e2ca61616), LL(0xe83acdd274f73a3a), LL(0xb9696fd0d2066969), LL(0x2409482d12410909),
    LL(0xdd70a7ade0d77070), LL(0xe2b6d954716fb6b6), LL(0x67d0ceb7bd1ed0d0), LL(0x93ed3b7ec7d6eded),
    LL(0x17cc2edb85e2cccc), LL(0x15422a5784684242), LL(0x5a98b4c22d2c9898), LL(0xaaa4490e55eda4a4),
    LL(0xa0285d8850752828), LL(0x6d5cda31b8865c5c), LL(0xc7f8933fed6bf8f8), LL(0x228644a411c28686),
};

__constant__ static const u64 C7[256] = {
    LL(0x186018c07830d818), LL(0x238c2305af462623), LL(0xc63fc67ef991b8c6), LL(0xe887e8136fcdfbe8),
    LL(0x8726874ca113cb87), LL(0xb8dab8a9626d11b8), LL(0x0104010805020901), LL(0x4f214f426e9e0d4f),
    LL(0x36d836adee6c9b36), LL(0xa6a2a6590451ffa6), LL(0xd26fd2debdb90cd2), LL(0xf5f3f5fb06f70ef5),
    LL(0x79f979ef80f29679), LL(0x6fa16f5fcede306f), LL(0x917e91fcef3f6d91), LL(0x525552aa07a4f852),
    LL(0x609d6027fdc04760), LL(0xbccabc89766535bc), LL(0x9b569baccd2b379b), LL(0x8e028e048c018a8e),
    LL(0xa3b6a371155bd2a3), LL(0x0c300c603c186c0c), LL(0x7bf17bff8af6847b), LL(0x35d435b5e16a8035),
    LL(0x1d741de8693af51d), LL(0xe0a7e05347ddb3e0), LL(0xd77bd7f6acb321d7), LL(0xc22fc25eed999cc2),
    LL(0x2eb82e6d965c432e), LL(0x4b314b627a96294b), LL(0xfedffea321e15dfe), LL(0x5741578216aed557),
    LL(0x155415a8412abd15), LL(0x77c1779fb6eee877), LL(0x37dc37a5eb6e9237), LL(0xe5b3e57b56d79ee5),
    LL(0x9f469f8cd923139f), LL(0xf0e7f0d317fd23f0), LL(0x4a354a6a7f94204a), LL(0xda4fda9e95a944da),
    LL(0x587d58fa25b0a258), LL(0xc903c906ca8fcfc9), LL(0x29a429558d527c29), LL(0x0a280a5022145a0a),
    LL(0xb1feb1e14f7f50b1), LL(0xa0baa0691a5dc9a0), LL(0x6bb16b7fdad6146b), LL(0x852e855cab17d985),
    LL(0xbdcebd8173673cbd), LL(0x5d695dd234ba8f5d), LL(0x1040108050209010), LL(0xf4f7f4f303f507f4),
    LL(0xcb0bcb16c08bddcb), LL(0x3ef83eedc67cd33e), LL(0x05140528110a2d05), LL(0x6781671fe6ce7867),
    LL(0xe4b7e47353d597e4), LL(0x279c2725bb4e0227), LL(0x4119413258827341), LL(0x8b168b2c9d0ba78b),
    LL(0xa7a6a7510153f6a7), LL(0x7de97dcf94fab27d), LL(0x956e95dcfb374995), LL(0xd847d88e9fad56d8),
    LL(0xfbcbfb8b30eb70fb), LL(0xee9fee2371c1cdee), LL(0x7ced7cc791f8bb7c), LL(0x66856617e3cc7166),
    LL(0xdd53dda68ea77bdd), LL(0x175c17b84b2eaf17), LL(0x47014702468e4547), LL(0x9e429e84dc211a9e),
    LL(0xca0fca1ec589d4ca), LL(0x2db42d75995a582d), LL(0xbfc6bf9179632ebf), LL(0x071c07381b0e3f07),
    LL(0xad8ead012347acad), LL(0x5a755aea2fb4b05a), LL(0x8336836cb51bef83), LL(0x33cc3385ff66b633),
    LL(0x6391633ff2c65c63), LL(0x020802100a041202), LL(0xaa92aa39384993aa), LL(0x71d971afa8e2de71),
    LL(0xc807c80ecf8dc6c8), LL(0x196419c87d32d119), LL(0x4939497270923b49), LL(0xd943d9869aaf5fd9),
    LL(0xf2eff2c31df931f2), LL(0xe3abe34b48dba8e3), LL(0x5b715be22ab6b95b), LL(0x881a8834920dbc88),
    LL(0x9a529aa4c8293e9a), LL(0x2698262dbe4c0b26), LL(0x32c8328dfa64bf32), LL(0xb0fab0e94a7d59b0),
    LL(0xe983e91b6acff2e9), LL(0x0f3c0f78331e770f), LL(0xd573d5e6a6b733d5), LL(0x803a8074ba1df480),
    LL(0xbec2be997c6127be), LL(0xcd13cd26de87ebcd), LL(0x34d034bde4688934), LL(0x483d487a75903248),
    LL(0xffdbffab24e354ff), LL(0x7af57af78ff48d7a), LL(0x907a90f4ea3d6490), LL(0x5f615fc23ebe9d5f),
    LL(0x2080201da0403d20), LL(0x68bd6867d5d00f68), LL(0x1a681ad07234ca1a), LL(0xae82ae192c41b7ae),
    LL(0xb4eab4c95e757db4), LL(0x544d549a19a8ce54), LL(0x937693ece53b7f93), LL(0x2288220daa442f22),
    LL(0x648d6407e9c86364), LL(0xf1e3f1db12ff2af1), LL(0x73d173bfa2e6cc73), LL(0x124812905a248212),
    LL(0x401d403a5d807a40), LL(0x0820084028104808), LL(0xc32bc356e89b95c3), LL(0xec97ec337bc5dfec),
    LL(0xdb4bdb9690ab4ddb), LL(0xa1bea1611f5fc0a1), LL(0x8d0e8d1c8307918d), LL(0x3df43df5c97ac83d),
    LL(0x976697ccf1335b97), LL(0x0000000000000000), LL(0xcf1bcf36d483f9cf), LL(0x2bac2b4587566e2b),
    LL(0x76c57697b3ece176), LL(0x82328264b019e682), LL(0xd67fd6fea9b128d6), LL(0x1b6c1bd87736c31b),
    LL(0xb5eeb5c15b7774b5), LL(0xaf86af112943beaf), LL(0x6ab56a77dfd41d6a), LL(0x505d50ba0da0ea50),
    LL(0x450945124c8a5745), LL(0xf3ebf3cb18fb38f3), LL(0x30c0309df060ad30), LL(0xef9bef2b74c3c4ef),
    LL(0x3ffc3fe5c37eda3f), LL(0x554955921caac755), LL(0xa2b2a2791059dba2), LL(0xea8fea0365c9e9ea),
    LL(0x6589650fecca6a65), LL(0xbad2bab9686903ba), LL(0x2fbc2f65935e4a2f), LL(0xc027c04ee79d8ec0),
    LL(0xde5fdebe81a160de), LL(0x1c701ce06c38fc1c), LL(0xfdd3fdbb2ee746fd), LL(0x4d294d52649a1f4d),
    LL(0x927292e4e0397692), LL(0x75c9758fbceafa75), LL(0x061806301e0c3606), LL(0x8a128a249809ae8a),
    LL(0xb2f2b2f940794bb2), LL(0xe6bfe66359d185e6), LL(0x0e380e70361c7e0e), LL(0x1f7c1ff8633ee71f),
    LL(0x62956237f7c45562), LL(0xd477d4eea3b53ad4), LL(0xa89aa829324d81a8), LL(0x966296c4f4315296),
    LL(0xf9c3f99b3aef62f9), LL(0xc533c566f697a3c5), LL(0x25942535b14a1025), LL(0x597959f220b2ab59),
    LL(0x842a8454ae15d084), LL(0x72d572b7a7e4c572), LL(0x39e439d5dd72ec39), LL(0x4c2d4c5a6198164c),
    LL(0x5e655eca3bbc945e), LL(0x78fd78e785f09f78), LL(0x38e038ddd870e538), LL(0x8c0a8c148605988c),
    LL(0xd163d1c6b2bf17d1), LL(0xa5aea5410b57e4a5), LL(0xe2afe2434dd9a1e2), LL(0x6199612ff8c24e61),
    LL(0xb3f6b3f1457b42b3), LL(0x21842115a5423421), LL(0x9c4a9c94d625089c), LL(0x1e781ef0663cee1e),
    LL(0x4311432252866143), LL(0xc73bc776fc93b1c7), LL(0xfcd7fcb32be54ffc), LL(0x0410042014082404),
    LL(0x515951b208a2e351), LL(0x995e99bcc72f2599), LL(0x6da96d4fc4da226d), LL(0x0d340d68391a650d),
    LL(0xfacffa8335e979fa), LL(0xdf5bdfb684a369df), LL(0x7ee57ed79bfca97e), LL(0x2490243db4481924),
    LL(0x3bec3bc5d776fe3b), LL(0xab96ab313d4b9aab), LL(0xce1fce3ed181f0ce), LL(0x1144118855229911),
    LL(0x8f068f0c8903838f), LL(0x4e254e4a6b9c044e), LL(0xb7e6b7d1517366b7), LL(0xeb8beb0b60cbe0eb),
    LL(0x3cf03cfdcc78c13c), LL(0x813e817cbf1ffd81), LL(0x946a94d4fe354094), LL(0xf7fbf7eb0cf31cf7),
    LL(0xb9deb9a1676f18b9), LL(0x134c13985f268b13), LL(0x2cb02c7d9c58512c), LL(0xd36bd3d6b8bb05d3),
    LL(0xe7bbe76b5cd38ce7), LL(0x6ea56e57cbdc396e), LL(0xc437c46ef395aac4), LL(0x030c03180f061b03),
    LL(0x5645568a13acdc56), LL(0x440d441a49885e44), LL(0x7fe17fdf9efea07f), LL(0xa99ea921374f88a9),
    LL(0x2aa82a4d8254672a), LL(0xbbd6bbb16d6b0abb), LL(0xc123c146e29f87c1), LL(0x535153a202a6f153),
    LL(0xdc57dcae8ba572dc), LL(0x0b2c0b582716530b), LL(0x9d4e9d9cd327019d), LL(0x6cad6c47c1d82b6c),
    LL(0x31c43195f562a431), LL(0x74cd7487b9e8f374), LL(0xf6fff6e309f115f6), LL(0x4605460a438c4c46),
    LL(0xac8aac092645a5ac), LL(0x891e893c970fb589), LL(0x145014a04428b414), LL(0xe1a3e15b42dfbae1),
    LL(0x165816b04e2ca616), LL(0x3ae83acdd274f73a), LL(0x69b9696fd0d20669), LL(0x092409482d124109),
    LL(0x70dd70a7ade0d770), LL(0xb6e2b6d954716fb6), LL(0xd067d0ceb7bd1ed0), LL(0xed93ed3b7ec7d6ed),
    LL(0xcc17cc2edb85e2cc), LL(0x4215422a57846842), LL(0x985a98b4c22d2c98), LL(0xa4aaa4490e55eda4),
    LL(0x28a0285d88507528), LL(0x5c6d5cda31b8865c), LL(0xf8c7f8933fed6bf8), LL(0x86228644a411c286),
};

__constant__ static const u64 rc[R + 1] = {
    LL(0x0000000000000000),
    LL(0x1823c6e887b8014f),
    LL(0x36a6d2f5796f9152),
    LL(0x60bc9b8ea30c7b35),
    LL(0x1de0d7c22e4bfe57),
    LL(0x157737e59ff04ada),
    LL(0x58c9290ab1a06b85),
    LL(0xbd5d10f4cb3e0567),
    LL(0xe427418ba77d95d8),
    LL(0xfbee7c66dd17479e),
    LL(0xca2dbf07ad5a8333),
};

/**
 * The core Whirlpool transform.
 */
__device__ static void processBuffer(struct NESSIEstruct * const structpointer) {
    int i, r;
    u64 K[8];        /* the round key */
    u64 block[8];    /* mu(buffer) */
    u64 state[8];    /* the cipher state */
    u64 L[8];
    u8 *buffer = structpointer->buffer;
    /*
     * map the buffer to a block:
     */
    for (i = 0; i < 8; i++, buffer += 8) {
        block[i] =
            (((u64)buffer[0]        ) << 56) ^
            (((u64)buffer[1] & 0xffL) << 48) ^
            (((u64)buffer[2] & 0xffL) << 40) ^
            (((u64)buffer[3] & 0xffL) << 32) ^
            (((u64)buffer[4] & 0xffL) << 24) ^
            (((u64)buffer[5] & 0xffL) << 16) ^
            (((u64)buffer[6] & 0xffL) <<  8) ^
            (((u64)buffer[7] & 0xffL)      );
    }
    /*
     * compute and apply K^0 to the cipher state:
     */
    state[0] = block[0] ^ (K[0] = structpointer->hash[0]);
    state[1] = block[1] ^ (K[1] = structpointer->hash[1]);
    state[2] = block[2] ^ (K[2] = structpointer->hash[2]);
    state[3] = block[3] ^ (K[3] = structpointer->hash[3]);
    state[4] = block[4] ^ (K[4] = structpointer->hash[4]);
    state[5] = block[5] ^ (K[5] = structpointer->hash[5]);
    state[6] = block[6] ^ (K[6] = structpointer->hash[6]);
    state[7] = block[7] ^ (K[7] = structpointer->hash[7]);
    /*
     * iterate over all rounds:
     */
    for (r = 1; r <= R; r++) {
        /*
         * compute K^r from K^{r-1}:
         */
        L[0] =
            C0[(int)(K[0] >> 56)       ] ^
            C1[(int)(K[7] >> 48) & 0xff] ^
            C2[(int)(K[6] >> 40) & 0xff] ^
            C3[(int)(K[5] >> 32) & 0xff] ^
            C4[(int)(K[4] >> 24) & 0xff] ^
            C5[(int)(K[3] >> 16) & 0xff] ^
            C6[(int)(K[2] >>  8) & 0xff] ^
            C7[(int)(K[1]      ) & 0xff] ^
            rc[r];
        L[1] =
            C0[(int)(K[1] >> 56)       ] ^
            C1[(int)(K[0] >> 48) & 0xff] ^
            C2[(int)(K[7] >> 40) & 0xff] ^
            C3[(int)(K[6] >> 32) & 0xff] ^
            C4[(int)(K[5] >> 24) & 0xff] ^
            C5[(int)(K[4] >> 16) & 0xff] ^
            C6[(int)(K[3] >>  8) & 0xff] ^
            C7[(int)(K[2]      ) & 0xff];
        L[2] =
            C0[(int)(K[2] >> 56)       ] ^
            C1[(int)(K[1] >> 48) & 0xff] ^
            C2[(int)(K[0] >> 40) & 0xff] ^
            C3[(int)(K[7] >> 32) & 0xff] ^
            C4[(int)(K[6] >> 24) & 0xff] ^
            C5[(int)(K[5] >> 16) & 0xff] ^
            C6[(int)(K[4] >>  8) & 0xff] ^
            C7[(int)(K[3]      ) & 0xff];
        L[3] =
            C0[(int)(K[3] >> 56)       ] ^
            C1[(int)(K[2] >> 48) & 0xff] ^
            C2[(int)(K[1] >> 40) & 0xff] ^
            C3[(int)(K[0] >> 32) & 0xff] ^
            C4[(int)(K[7] >> 24) & 0xff] ^
            C5[(int)(K[6] >> 16) & 0xff] ^
            C6[(int)(K[5] >>  8) & 0xff] ^
            C7[(int)(K[4]      ) & 0xff];
        L[4] =
            C0[(int)(K[4] >> 56)       ] ^
            C1[(int)(K[3] >> 48) & 0xff] ^
            C2[(int)(K[2] >> 40) & 0xff] ^
            C3[(int)(K[1] >> 32) & 0xff] ^
            C4[(int)(K[0] >> 24) & 0xff] ^
            C5[(int)(K[7] >> 16) & 0xff] ^
            C6[(int)(K[6] >>  8) & 0xff] ^
            C7[(int)(K[5]      ) & 0xff];
        L[5] =
            C0[(int)(K[5] >> 56)       ] ^
            C1[(int)(K[4] >> 48) & 0xff] ^
            C2[(int)(K[3] >> 40) & 0xff] ^
            C3[(int)(K[2] >> 32) & 0xff] ^
            C4[(int)(K[1] >> 24) & 0xff] ^
            C5[(int)(K[0] >> 16) & 0xff] ^
            C6[(int)(K[7] >>  8) & 0xff] ^
            C7[(int)(K[6]      ) & 0xff];
        L[6] =
            C0[(int)(K[6] >> 56)       ] ^
            C1[(int)(K[5] >> 48) & 0xff] ^
            C2[(int)(K[4] >> 40) & 0xff] ^
            C3[(int)(K[3] >> 32) & 0xff] ^
            C4[(int)(K[2] >> 24) & 0xff] ^
            C5[(int)(K[1] >> 16) & 0xff] ^
            C6[(int)(K[0] >>  8) & 0xff] ^
            C7[(int)(K[7]      ) & 0xff];
        L[7] =
            C0[(int)(K[7] >> 56)       ] ^
            C1[(int)(K[6] >> 48) & 0xff] ^
            C2[(int)(K[5] >> 40) & 0xff] ^
            C3[(int)(K[4] >> 32) & 0xff] ^
            C4[(int)(K[3] >> 24) & 0xff] ^
            C5[(int)(K[2] >> 16) & 0xff] ^
            C6[(int)(K[1] >>  8) & 0xff] ^
            C7[(int)(K[0]      ) & 0xff];
        K[0] = L[0];
        K[1] = L[1];
        K[2] = L[2];
        K[3] = L[3];
        K[4] = L[4];
        K[5] = L[5];
        K[6] = L[6];
        K[7] = L[7];
        /*
         * apply the r-th round transformation:
         */
        L[0] =
            C0[(int)(state[0] >> 56)       ] ^
            C1[(int)(state[7] >> 48) & 0xff] ^
            C2[(int)(state[6] >> 40) & 0xff] ^
            C3[(int)(state[5] >> 32) & 0xff] ^
            C4[(int)(state[4] >> 24) & 0xff] ^
            C5[(int)(state[3] >> 16) & 0xff] ^
            C6[(int)(state[2] >>  8) & 0xff] ^
            C7[(int)(state[1]      ) & 0xff] ^
            K[0];
        L[1] =
            C0[(int)(state[1] >> 56)       ] ^
            C1[(int)(state[0] >> 48) & 0xff] ^
            C2[(int)(state[7] >> 40) & 0xff] ^
            C3[(int)(state[6] >> 32) & 0xff] ^
            C4[(int)(state[5] >> 24) & 0xff] ^
            C5[(int)(state[4] >> 16) & 0xff] ^
            C6[(int)(state[3] >>  8) & 0xff] ^
            C7[(int)(state[2]      ) & 0xff] ^
            K[1];
        L[2] =
            C0[(int)(state[2] >> 56)       ] ^
            C1[(int)(state[1] >> 48) & 0xff] ^
            C2[(int)(state[0] >> 40) & 0xff] ^
            C3[(int)(state[7] >> 32) & 0xff] ^
            C4[(int)(state[6] >> 24) & 0xff] ^
            C5[(int)(state[5] >> 16) & 0xff] ^
            C6[(int)(state[4] >>  8) & 0xff] ^
            C7[(int)(state[3]      ) & 0xff] ^
            K[2];
        L[3] =
            C0[(int)(state[3] >> 56)       ] ^
            C1[(int)(state[2] >> 48) & 0xff] ^
            C2[(int)(state[1] >> 40) & 0xff] ^
            C3[(int)(state[0] >> 32) & 0xff] ^
            C4[(int)(state[7] >> 24) & 0xff] ^
            C5[(int)(state[6] >> 16) & 0xff] ^
            C6[(int)(state[5] >>  8) & 0xff] ^
            C7[(int)(state[4]      ) & 0xff] ^
            K[3];
        L[4] =
            C0[(int)(state[4] >> 56)       ] ^
            C1[(int)(state[3] >> 48) & 0xff] ^
            C2[(int)(state[2] >> 40) & 0xff] ^
            C3[(int)(state[1] >> 32) & 0xff] ^
            C4[(int)(state[0] >> 24) & 0xff] ^
            C5[(int)(state[7] >> 16) & 0xff] ^
            C6[(int)(state[6] >>  8) & 0xff] ^
            C7[(int)(state[5]      ) & 0xff] ^
            K[4];
        L[5] =
            C0[(int)(state[5] >> 56)       ] ^
            C1[(int)(state[4] >> 48) & 0xff] ^
            C2[(int)(state[3] >> 40) & 0xff] ^
            C3[(int)(state[2] >> 32) & 0xff] ^
            C4[(int)(state[1] >> 24) & 0xff] ^
            C5[(int)(state[0] >> 16) & 0xff] ^
            C6[(int)(state[7] >>  8) & 0xff] ^
            C7[(int)(state[6]      ) & 0xff] ^
            K[5];
        L[6] =
            C0[(int)(state[6] >> 56)       ] ^
            C1[(int)(state[5] >> 48) & 0xff] ^
            C2[(int)(state[4] >> 40) & 0xff] ^
            C3[(int)(state[3] >> 32) & 0xff] ^
            C4[(int)(state[2] >> 24) & 0xff] ^
            C5[(int)(state[1] >> 16) & 0xff] ^
            C6[(int)(state[0] >>  8) & 0xff] ^
            C7[(int)(state[7]      ) & 0xff] ^
            K[6];
        L[7] =
            C0[(int)(state[7] >> 56)       ] ^
            C1[(int)(state[6] >> 48) & 0xff] ^
            C2[(int)(state[5] >> 40) & 0xff] ^
            C3[(int)(state[4] >> 32) & 0xff] ^
            C4[(int)(state[3] >> 24) & 0xff] ^
            C5[(int)(state[2] >> 16) & 0xff] ^
            C6[(int)(state[1] >>  8) & 0xff] ^
            C7[(int)(state[0]      ) & 0xff] ^
            K[7];
        state[0] = L[0];
        state[1] = L[1];
        state[2] = L[2];
        state[3] = L[3];
        state[4] = L[4];
        state[5] = L[5];
        state[6] = L[6];
        state[7] = L[7];
    }
    /*
     * apply the Miyaguchi-Preneel compression function:
     */
    structpointer->hash[0] ^= state[0] ^ block[0];
    structpointer->hash[1] ^= state[1] ^ block[1];
    structpointer->hash[2] ^= state[2] ^ block[2];
    structpointer->hash[3] ^= state[3] ^ block[3];
    structpointer->hash[4] ^= state[4] ^ block[4];
    structpointer->hash[5] ^= state[5] ^ block[5];
    structpointer->hash[6] ^= state[6] ^ block[6];
    structpointer->hash[7] ^= state[7] ^ block[7];
}

/**
 * Initialize the hashing state.
 */
__device__ void WHIRLPOOL_init(struct NESSIEstruct * const structpointer) {
    int i;

    memset(structpointer->bitLength, 0, 32);
    structpointer->bufferBits = structpointer->bufferPos = 0;
    structpointer->buffer[0] = 0; /* it's only necessary to cleanup buffer[bufferPos] */
    for (i = 0; i < 8; i++) {
        structpointer->hash[i] = 0L; /* initial value */
    }
}

/**
 * Delivers input data to the hashing algorithm.
 *
 * @param    source        plaintext data to hash.
 * @param    sourceBits    how many bits of plaintext to process.
 *
 * This method maintains the invariant: bufferBits < DIGESTBITS
 */
__device__ void WHIRLPOOL_add(const unsigned char * const source,
               unsigned __int32 sourceBits,
               struct NESSIEstruct * const structpointer) {
    /*
                       sourcePos
                       |
                       +-------+-------+-------
                          ||||||||||||||||||||| source
                       +-------+-------+-------
    +-------+-------+-------+-------+-------+-------
    ||||||||||||||||||||||                           buffer
    +-------+-------+-------+-------+-------+-------
                    |
                    bufferPos
    */
    int sourcePos    = 0; /* index of leftmost source u8 containing data (1 to 8 bits). */
    int sourceGap    = (8 - ((int)sourceBits & 7)) & 7; /* space on source[sourcePos]. */
    int bufferRem    = structpointer->bufferBits & 7; /* occupied bits on buffer[bufferPos]. */
    int i;
    u32 b, carry;
    u8 *buffer       = structpointer->buffer;
    u8 *bitLength    = structpointer->bitLength;
    int bufferBits   = structpointer->bufferBits;
    int bufferPos    = structpointer->bufferPos;

    /*
     * tally the length of the added data:
     */
    u64 value = sourceBits;
    for (i = 31, carry = 0; i >= 0 && (carry != 0 || value != LL(0)); i--) {
        carry += bitLength[i] + ((u32)value & 0xff);
        bitLength[i] = (u8)carry;
        carry >>= 8;
        value >>= 8;
    }
    /*
     * process data in chunks of 8 bits (a more efficient approach would be to take whole-word chunks):
     */
    while (sourceBits > 8) {
        /* N.B. at least source[sourcePos] and source[sourcePos+1] contain data. */
        /*
         * take a byte from the source:
         */
        b = ((source[sourcePos] << sourceGap) & 0xff) |
            ((source[sourcePos + 1] & 0xff) >> (8 - sourceGap));
        /*
         * process this byte:
         */
        buffer[bufferPos++] |= (u8)(b >> bufferRem);
        bufferBits += 8 - bufferRem; /* bufferBits = 8*bufferPos; */
        if (bufferBits == DIGESTBITS) {
            /*
             * process data block:
             */
            processBuffer(structpointer);
            /*
             * reset buffer:
             */
            bufferBits = bufferPos = 0;
        }
        buffer[bufferPos] = (u8) (b << (8 - bufferRem));
        bufferBits += bufferRem;
        /*
         * proceed to remaining data:
         */
        sourceBits -= 8;
        sourcePos++;
    }
    /* now 0 <= sourceBits <= 8;
     * furthermore, all data (if any is left) is in source[sourcePos].
     */
    if (sourceBits > 0) {
        b = (source[sourcePos] << sourceGap) & 0xff; /* bits are left-justified on b. */
        /*
         * process the remaining bits:
         */
        buffer[bufferPos] |= b >> bufferRem;
    } else {
        b = 0;
    }
    if (bufferRem + sourceBits < 8) {
        /*
         * all remaining data fits on buffer[bufferPos],
         * and there still remains some space.
         */
        bufferBits += sourceBits;
    } else {
        /*
         * buffer[bufferPos] is full:
         */
        bufferPos++;
        bufferBits += 8 - bufferRem; /* bufferBits = 8*bufferPos; */
        sourceBits -= 8 - bufferRem;
        /* now 0 <= sourceBits < 8;
         * furthermore, all data (if any is left) is in source[sourcePos].
         */
        if (bufferBits == DIGESTBITS) {
            /*
             * process data block:
             */
            processBuffer(structpointer);
            /*
             * reset buffer:
             */
            bufferBits = bufferPos = 0;
        }
        buffer[bufferPos] = (u8) (b << (8 - bufferRem));
        bufferBits += (int)sourceBits;
    }
    structpointer->bufferBits   = bufferBits;
    structpointer->bufferPos    = bufferPos;
}

/**
 * Get the hash value from the hashing state.
 * 
 * This method uses the invariant: bufferBits < DIGESTBITS
 */
__device__ void WHIRLPOOL_finalize(struct NESSIEstruct * const structpointer,
                    unsigned char * const result) {
    int i;
    u8 *buffer      = structpointer->buffer;
    u8 *bitLength   = structpointer->bitLength;
    int bufferBits  = structpointer->bufferBits;
    int bufferPos   = structpointer->bufferPos;
    u8 *digest      = result;

    /*
     * append a '1'-bit:
     */
    buffer[bufferPos] |= 0x80U >> (bufferBits & 7);
    bufferPos++; /* all remaining bits on the current u8 are set to zero. */
    /*
     * pad with zero bits to complete (N*WBLOCKBITS - LENGTHBITS) bits:
     */
    if (bufferPos > WBLOCKBYTES - LENGTHBYTES) {
        if (bufferPos < WBLOCKBYTES) {
            memset(&buffer[bufferPos], 0, WBLOCKBYTES - bufferPos);
        }
        /*
         * process data block:
         */
        processBuffer(structpointer);
        /*
         * reset buffer:
         */
        bufferPos = 0;
    }
    if (bufferPos < WBLOCKBYTES - LENGTHBYTES) {
        memset(&buffer[bufferPos], 0, (WBLOCKBYTES - LENGTHBYTES) - bufferPos);
    }
    bufferPos = WBLOCKBYTES - LENGTHBYTES;
    /*
     * append bit length of hashed data:
     */
    memcpy(&buffer[WBLOCKBYTES - LENGTHBYTES], bitLength, LENGTHBYTES);
    /*
     * process data block:
     */
    processBuffer(structpointer);
    /*
     * return the completed message digest:
     */
    for (i = 0; i < DIGESTBYTES/8; i++) {
        digest[0] = (u8)(structpointer->hash[i] >> 56);
        digest[1] = (u8)(structpointer->hash[i] >> 48);
        digest[2] = (u8)(structpointer->hash[i] >> 40);
        digest[3] = (u8)(structpointer->hash[i] >> 32);
        digest[4] = (u8)(structpointer->hash[i] >> 24);
        digest[5] = (u8)(structpointer->hash[i] >> 16);
        digest[6] = (u8)(structpointer->hash[i] >>  8);
        digest[7] = (u8)(structpointer->hash[i]      );
        digest += 8;
    }
    structpointer->bufferBits   = bufferBits;
    structpointer->bufferPos    = bufferPos;
}

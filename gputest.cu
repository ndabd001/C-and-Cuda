#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>

int N = 64000000;                                                                                                                                                                                         
int doPrint = 0; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER CODE TO INITIALIZE, PRINT AND TIME
struct timeval start, end;
void initialize(float *a, int N) {
  int i;
  for (i = 0; i < N; ++i) { 
    a[i] = pow(rand() % 10, 2); 
  }                                                                                                                                                                                       
}

void print(float* a, int N) {
   if (doPrint) {
   int i;
   for (i = 0; i < N; ++i)
      printf("%f ", a[i]);
   printf("\n");
   }
}  

void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

void init(float* a, int N, const char* c) {
  printf("***************** %s **********************\n", c);
  initialize(a, N); 
  print(a, N);
  starttime();
}

void finish(float* a, int N, const char* c) {
  endtime(c);
  print(a, N);
  printf("***************************************************\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////



// Normal C function to square root values
void normal(float* a, int N)                                                                                                                                                                                     
{
  int i;                                                                                                                                                                                                                
  for (i = 0; i < N; ++i)                                                                                                                                                                                    
    a[i] = sqrt(a[i]);                                                                                                                                                                                           
}                 

// GPU function to square root values
__global__ void gpu_sqrt(float* a, int N) {
   int element = blockIdx.x*blockDim.x + threadIdx.x;
   if (element < N) a[element] = sqrt(a[element]);
}

void gpu(float* a, int N) {
   int numThreads = 1024; // This can vary, up to 1024
   int numCores = N / 1024 + 1;

   float* gpuA;
   cudaMalloc(&gpuA, N*sizeof(float)); // Allocate enough memory on the GPU
   cudaMemcpy(gpuA, a, N*sizeof(float), cudaMemcpyHostToDevice); // Copy array from CPU to GPU
   gpu_sqrt<<<numCores, numThreads>>>(gpuA, N);  // Call GPU Sqrt
   cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy array from GPU to CPU
   cudaFree(&gpuA); // Free the memory on the GPU
}
                                                                                                                                                                                               
 

int main()                                                                                                                                                                                  
{                                                                                                                                                                                                                
  //////////////////////////////////////////////////////////////////////////
  // Necessary if you are doing SSE.  Align on a 128-bit boundary (16 bytes)
  float* a;                                                                                                                                                                                                      
  posix_memalign((void**)&a, 16,  N * sizeof(float));                                                                                                                                                            
  /////////////////////////////////////////////////////////////////////////

  // Test 1: Sequential For Loop
  init(a, N, "Normal");
  normal(a, N); 
  finish(a, N, "Normal"); 

  // Test 2: Vectorization
  init(a, N, "GPU");
  gpu(a, N);  
  finish(a, N, "GPU");

  return 0;
}


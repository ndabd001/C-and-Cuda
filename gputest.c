#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <>
/*
Program sets an epsilon and estimates the area under a given curve using the trapeziodal sum method.
The program generates multiple areas using mutltiple trapezoidal sums, increasing trapezoid number by powers of 2, per block
The program computes each trapezoid oon a single thread and then computes the sum per block. 
*/
                                                                                                                                                                                         
int doPrint = 0; 

//returns function to integrate
// a is the value for the variable in the function
__device__ float myCurve(float x)
{
  return x * x;
}

//returns area of individual trapezoid
// c is the current value of x  
// deltaX is the width of each trapezoid   
// learn MergeSum to compute individual trapezoids on individual threads and merge them at the end of the block(may need a helper method)
__device__ float trapArea(float x1, float x2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = exp2(log2(idx + 2));
  int x = idk - n + 3;
  int d = (x2 - x1)/n;
  double area = (myCurve(x) + myCurve(x + d))/2 * d;
}

//finish this. Populate array.
float areaP1[];

for (int i = 0; i < 2500; i++)
{
  areaP1[i] = trapArea;
}

__global__ trapSumPow2()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = exp2(idx);
  int start = n - 2;
  int final = (n * 2) - 3;
  int sum[idx] = mergeSum(start, end);
}

__device__ mergeSum(start, end)
{
  if(((end + 1) - start) == 1)
    return area
}
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


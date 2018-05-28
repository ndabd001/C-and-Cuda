#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
//#include <>

int N = 64000000;
int nTrapsPow2 = 2046;
int nSumsPow2 = 10;
int nTraps;
int nSums;                                                                                                                                         
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

void init(const char* c) {
  printf("***************** %s **********************\n", c);
  //initialize(a, N); 
  //print(a, N);
  starttime();
}

void finish(const char* c) {
  endtime(c);
  //print(a, N);
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

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__device__ double myCurve(double x)
{
	return x * x;
}

__device__ double Sum(int trapStart, int trapEnd, double* area)
{	
	int i;
	double total = 0;
        if(((trapEnd+1) - trapStart) <= 0)
                return -1;
	else
	{
		for(i = trapStart; i <= trapEnd; i++)
		{
			total += area[i];
		}
	}
	return total;
}

                                                   
__global__ void trapAreaPow2(int x1, int x2, double* area, int numTraps)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numTraps)
	{
		int n, t;
		double d, x;
		n = index + 2;
		n = log2((double) n);
		if(index == 6 || index == 62)
		{
//			printf("%d) log2(%f) = %d\n", index,(double) (index+2), n);
			n++;
		}
		n = exp2((double) n);
		d = (x2 - x1) / (double)n;
		t = index - n + 3;
		x = ((t - 1) * d) + x1;
		area[index] = ((myCurve(x) + myCurve(x+d)) / 2) * d;
		if(index >= 1022 && index <= 2045)
		{
			//printf("index = %d:\nn = %d\nd = %f\nt = %d\nx = %f\n", index, n, d, t, x);
			//printf("area[%d] = %f\n", index, area[index]);
			//printf("index = %d, n = %d\n", index, n);
		}
	}
}
__global__ void trapSumPow2(double* sum, int numSums, double* area)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numSums)
	{
		int n, trapStart, trapEnd;	
		n = exp2((double) (index+1));
		trapStart = n - 2;
		trapEnd = (n * 2) - 3;
		sum[index] = Sum(trapStart, trapEnd, area);
	}
}


__global__ void trapArea(int x1, int x2, int S, double* area, int numTraps)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numTraps)
	{
		int a, b, t, n, i=0;
		double d, c;
		t = index;
		while(t >= 0)
		{
			t = t - S - i;
			i++;
		}
		a = i - 1;
		n = S + a;
		d = (x2 - x1) / (double) n;
		b = ((a * a) - a) / 2;
		c = index - ((a * S) + b);
		c = x1 + (c * d);
		area[index] =  ((myCurve(c) + myCurve(c+d)) / 2) * d;
	//	printf("index: %d\na: %d\nn: %d\nd: %f\nc: %f\narea[%d] = %f\n", index, a, n, d, c, index, area[index]);
	}
}

__global__ void trapSum(int S, double* sum, int numSums, double* area)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numSums)
	{
		int trapStart, trapEnd;
		trapStart = (index * S) + (((index*index) - index) / 2);
		trapEnd = trapStart + S + index - 1;
		sum[index] = Sum(trapStart, trapEnd, area);
	}
}


int main()
{
	init("Trapezoidal Sum");
	
	int S = 0, x1 = 1, x2 = 4;
	int i, x = 0;
	double epsilon = 0.000001;
	double answer = 21;
 	double* areaPow2;
	double* sumPow2;
	double* answersPow2;
	double* areaLinear;
	double* sumLinear;
	int numThreads = 1024;
	int numCores = nTrapsPow2 / numThreads + 1;
	double* answersLinear;
	
//	printf("no fault here\n");

	answersPow2 = (double*)malloc(nSumsPow2*sizeof(double));
	cudaMalloc(&areaPow2, nTrapsPow2*sizeof(double));
	cudaMalloc(&sumPow2, nSumsPow2*sizeof(double));
	
//	printf("so far so good\n");

	trapAreaPow2<<<numCores, numThreads>>>(x1, x2, areaPow2, nTrapsPow2);

//	printf("got past trapAreaPow2\n");
		
	numCores = nSumsPow2 / numThreads + 1;
	trapSumPow2<<<numCores, numThreads>>>(sumPow2, nSumsPow2, areaPow2);

//	printf("made it past trapSumPow2\n");
	
	cudaMemcpy(answersPow2, sumPow2, nSumsPow2*sizeof(double), cudaMemcpyDeviceToHost);
	
//	printf("cudaMemcpy was sucessful\n");

	
//	printf("nSumsPow2 = %d\n", nSumsPow2); 
	for(i = nSumsPow2 - 1; i >= 0; i--)
	{
		printf("answersPow2[%d] = %f\n", i, answersPow2[i]);
		if((answer - answersPow2[i]) >= -epsilon && (answer - answersPow2[i]) <= epsilon)
		{
			x = i + 1;
		}
	}
	printf("loop sucessful\n");	
	S = exp2((double) x) / 2;
	printf("x = %d and S = %d\n", x, S);
	printf("S = %d, the sum with %d trapazoids was %f\n", S, S*2, answersPow2[x-1]);
	
	if(S == 1)
		printf("done\n");
	else if(S == 0)
		printf("1024 trapazoids is too few to get an answer within epsilon.\n");
	else
	{
		nTraps = ((S + 1) * S) + (((S * S) + S) / 2);
		nSums = S + 1;
		printf("nSums = %d\n", nSums);
		cudaMalloc(&sumLinear, nSums * sizeof(double));
		cudaMalloc(&areaLinear, nTraps * sizeof(double));
		numCores = (nTraps/numThreads) + 1;
		trapArea<<<numCores, numThreads>>>(x1, x2, S, areaLinear, nTraps);
		printf("this works\n");
		numCores = (nSums/numThreads) + 1;
		trapSum<<<numCores, numThreads>>>(S, sumLinear, nSums, areaLinear);
		answersLinear = (double*) malloc(nSums * sizeof(double));
		cudaMemcpy(answersLinear, sumLinear, nSums*sizeof(double), cudaMemcpyDeviceToHost);
		for(i=nSums-1; i >= 0; i--)
		{
			printf("answersLinear[%d] = %f\n", i, answersLinear[i]);
			if((answer - answersLinear[i]) >= -epsilon && (answer - answersLinear[i]) <= epsilon)
				x = i;
		}
		x = S + x;
		printf("This is the minimunm number of trapezoids needed to compute a trapezoidal sum that is within our epsilon of the actual answer: %d\n", x);
	}

	finish("Trapezoidal Sum");
	
  //////////////////////////////////////////////////////////////////////////
  // Necessary if you are doing SSE.  Align on a 128-bit boundary (16 bytes)
  float* a;                                                                                                                                                                                                      
  posix_memalign((void**)&a, 16,  N * sizeof(float));                                                                                                                                                            
  /////////////////////////////////////////////////////////////////////////

  // Test 1: Sequential For Loop
//  init(a, N, "Normal");
//  normal(a, N); 
//  finish(a, N, "Normal"); 

  // Test 2: Vectorization
//  init(a, N, "GPU");
//  gpu(a, N);  
//  finish(a, N, "GPU");

  return 0;
}



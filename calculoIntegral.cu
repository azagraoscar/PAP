#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 80000
#define NUM_THREADS 1024
#define BLOCKS 32

__device__ double valorDeLaFuncion(double valor)
{
	return (valor / ((valor * valor) + 4)) * (sin(1 / valor));
}

__global__ void calcularResultadosParciales(double* a, double* b, double result[BLOCKS], int num_threads, int blocks)
{
	double a_value = *a;
	double b_value = *b;

	__shared__ double resultados[NUM_THREADS]; //array para almacenar los resultados calculados por los hilos de un bloque

	int k = blockIdx.x * blockDim.x + threadIdx.x;	//iteracion correspondiente al hilo
	double suma = 0;
	
	while (k < N)	//el sumatorio de la regla del trapecio compuesto se ejecuta hasta N-1
	{
		if (k!=0)
		{
			double valor = a_value + (k * ((b_value - a_value) / N));
			suma += valorDeLaFuncion(valor);
		}

		k += (num_threads *blocks);
	}

	resultados[threadIdx.x] = suma;
	__syncthreads();

	//reduccion
	int limite = num_threads / 2;

	while (limite > 0)
	{
		if (threadIdx.x < limite)
		{
			resultados[threadIdx.x] += resultados[threadIdx.x + limite];
		}
		__syncthreads();
		limite /= 2;
	}
	
	if (threadIdx.x == 0)	//el resultado de sumar los valores calculados por los hilos 
	{						//se almacenaria en la primera posicion del array despues de la reduccion

		result[blockIdx.x] = resultados[0];	//almacenamos en el array de resultados el resultado parcial 
											//calculado por este bloque. 
											//Al final tendremos un array de length=numeroBloques 
											//y habrá que sumar todos los valores para hallar el valor final
	}
}

int main(void)
{
	//get gpu characteristics
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int maxBlocksSM = prop.maxBlocksPerMultiProcessor;
	int maxThreadsBlock = prop.maxThreadsPerBlock;
	int blocksPerGrid = 0;

	if (maxBlocksSM < ((N + maxThreadsBlock - 1) / maxThreadsBlock))
	{
		blocksPerGrid = maxBlocksSM;
	}
	else
	{
		blocksPerGrid = ((N + maxThreadsBlock - 1) / maxThreadsBlock);
	}

	//Declare all variables
	double a = 1;
	double b = 3;
	double* a_h;
	double* b_h;
	a_h = &a;
	b_h = &b;
	double result_h[BLOCKS];
	double* a_d;
	double* b_d;
	double* result_d;

	int size = sizeof(double);

	// Dynamically allocate device memory for GPU results.
	cudaMalloc((double**)&a_d, size);
	cudaMalloc((double**)&b_d, size);
	cudaMalloc((void**)&result_d, sizeof(result_h));


	// Copy host memory to device memory
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

	//call kernel function
	calcularResultadosParciales << < maxThreadsBlock, maxThreadsBlock >> > (a_d, b_d, result_d, maxThreadsBlock, blocksPerGrid);

	// Write GPU results in device memory back to host memory
	cudaMemcpy(result_h, result_d, sizeof(result_h), cudaMemcpyDeviceToHost);

	//calcular el valor de la integral
	double resultado = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		//printf("%f", result_h[i]);
		resultado += result_h[i];
	}
	double f_a = (a / (a * a + 4)) * (sin(1 / a));
	double f_b = (b / (b * b + 4)) * (sin(1 / b));
	double resultadoIntegral = ((b - a) / N) * (((f_a + f_b) / 2) + resultado);
	//print result
	printf("El resultado de la integral es %f", resultadoIntegral);

	// Free dynamically−allocated device memory
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(result_d);

	return 0;
}
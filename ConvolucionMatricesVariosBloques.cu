
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

const int WIDTH = 16;  //tamaño de la matriz A
const int WIDTH_B = 3; // tamaño de la matriz B
__global__ void convolucion(int* r, int* a, int* b)
{
	int pos_x = blockDim.x * blockIdx.x + threadIdx.x;  //Calculamos el valor de x
	int pos_y = blockDim.y * blockIdx.y + threadIdx.y;  //Calculamos el valor de y
	int pos = pos_x + pos_y * (WIDTH + (WIDTH_B - 1));  //Calculamos la posición en la matriz R
	if (pos_x < (WIDTH + (WIDTH_B - 1)) && pos_y < (WIDTH + (WIDTH_B - 1))) //Ya que tenemos hilos que no hacen nada porque tenemos bloques de hilos parciales, 
	{																		//tenemos que hacer la comprobación de que los hilos estén dentro de la matriz
		int sum = 0;
		int pos_i, pos_j; //posiciones de la matriz A
		for (int i = 0; i < WIDTH_B; i++)		//recorremos las filas de la matriz B
		{
			for (int j = 0; j < WIDTH_B; j++)	//recorremos las columnas de la matriz B
			{
				pos_i = pos_y + (i - (WIDTH_B - 1)); //calculamos la posición de la i restándole a la posición del hilo el desplazamiento de B 
				pos_j = pos_x + (j - (WIDTH_B - 1)); //calculamos la posición de la j restándole a la posición del hilo el desplazamiento de B
				if (!(pos_i < 0 || pos_j < 0 || pos_i >= WIDTH || pos_j >= WIDTH))
				{
					sum += a[pos_j + pos_i * WIDTH] * b[j + i * WIDTH_B]; //Multiplicamos el valor del valor que está en A con su correspondiente en B
				}
			}
		}
		r[pos] = sum;
	}
}
__global__ void invertir_matriz(int* r, int* a, int width)
{
	int pos = blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y * blockDim.x; //obtenemos la posición del hilo con el que estamos trabajando
	int inv_pos = -pos + (width * width - 1); //Le restamos a la posición máxima la posición actual
	r[pos] = a[inv_pos]; //cambiamos los valores
}

void imprimir_matriz(int* m, int width) //función auxiliar para imprimir matrices por pantalla
{
	for (int j = 0; j < width; j++)
	{
		for (int i = 0; i < width; i++)
		{
			printf("%6d  | ", m[i + j * width]);
		}
		printf("\n");
	}
}
int main()
{
	//Declaración de variables
	//Variables del host
	int* h_a;
	int* h_b;
	int* h_bi;
	int* h_res;

	//Variables del device
	int* d_a;
	int* d_b;
	int* d_bi;
	int* d_res;

	int width_res = (WIDTH + (WIDTH_B - 1)); //Calculamos el tamaño de la matriz resultante
	srand(time(NULL)); //Inicializamos srand para obtener valores aleatorios

	//calculamos los tamaños de las matrices
	int size_matriz_a = WIDTH * WIDTH * sizeof(int);
	int size_matriz_b = WIDTH_B * WIDTH_B * sizeof(int);
	int size_matriz_res = width_res * width_res * sizeof(int);

	//Usamos un tamaño de bloques de 16 porque es el óptimo. El número de bloques entonces, al trabajar con una matriz de tamaño igual a la matriz A + 2, necesitaremos 2x2 bloques
	dim3 blockSize(16, 16);
	dim3 grid(2, 2);


	//Gestión de memoria
	h_a = (int*)malloc(size_matriz_a);
	h_b = (int*)malloc(size_matriz_b);
	h_bi = (int*)malloc(size_matriz_b);
	h_res = (int*)malloc(size_matriz_res);

	cudaMalloc((void**)&d_a, size_matriz_a);
	cudaMalloc((void**)&d_b, size_matriz_b);
	cudaMalloc((void**)&d_bi, size_matriz_b);
	cudaMalloc((void**)&d_res, size_matriz_res);

	//Generamos las matrices con valores aleatorios
	for (int i = 1; i <= WIDTH * WIDTH; i++)
	{
		h_a[i - 1] = rand() % 255;
	}

	for (int i = 1; i <= WIDTH_B * WIDTH_B; i++)
	{
		h_b[i - 1] = rand() % 9;
	}

	//Imprimimos las matrices
	printf("MATRIZ A:\n");
	imprimir_matriz(h_a, WIDTH);

	printf("MATRIZ B:\n");
	imprimir_matriz(h_b, WIDTH_B);

	//Movemos las variables al dispositivo
	cudaMemcpy(d_a, h_a, size_matriz_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size_matriz_b, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bi, h_bi, size_matriz_b, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, h_res, size_matriz_res, cudaMemcpyHostToDevice);

	//liberamos memoria
	free(h_a);
	free(h_b);
	free(h_res);

	//operamos en el dispositivo
	invertir_matriz << <1, dim3(WIDTH_B, WIDTH_B) >> > (d_bi, d_b, WIDTH_B);
	cudaMemcpy(h_bi, d_bi, size_matriz_b, cudaMemcpyDeviceToHost);

	cudaFree(d_b); //ya no usaremos d_b, así que podemos liberar memoria

	//Imprimimos la matriz invertida
	printf("MATRIZ B INVERTIDA:\n");
	imprimir_matriz(h_bi, WIDTH_B);

	//Realizamos las operaciones
	convolucion << <grid, blockSize >> > (d_res, d_a, d_bi);
	cudaMemcpy(h_res, d_res, size_matriz_res, cudaMemcpyDeviceToHost);

	//liberamos memoria
	cudaFree(d_a);
	cudaFree(d_bi);
	cudaFree(d_res);

	//Imprimimos el resultado por pantalla
	printf("RESULTADO: \n");
	imprimir_matriz(h_res, width_res);
}

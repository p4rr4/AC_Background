/*
* Ejemplo en CUDA de la implementaci�n background subtraction
* aplicado a dos imágenes
*
* Adaptacion de un codigo de calculo de mediana
* desarrollado por Sergio Orts-Escolano
* Copyright Universidad de Alicante, 2012
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if _WIN32
#include <Windows.h>
#else
#include <sys/types.h>
#include <sys/time.h>
#endif

// Ficheros de inclusión para que funcione el intellisense en Visual Studio
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "EasyBMP.h"

#if _WIN32
typedef LARGE_INTEGER timeStamp;
void getCurrentTimeStamp(timeStamp& _time);
timeStamp getCurrentTimeStamp();
double getTimeMili(const timeStamp& start, const timeStamp& end);
double getTimeSecs(const timeStamp& start, const timeStamp& end);
#endif

// Dimensiones de la imagen a procesar
int WIDTH;
int HEIGHT;

// valor del umbral 
int Threshold;


// Funciones auxiliares
double get_current_time();
void checkCUDAError(const char*);


// Tamaño de grid y bloque CUDA
#define GRID_W  16
#define GRID_H  16
#define BLOCK_W 32
#define BLOCK_H 32

// Buffers de imagenes
unsigned char* input_imageb;
unsigned char* input_imagef;
unsigned char* gpu_output;

// CPU background
void CPUBackground(unsigned char* outputimage, unsigned char* inputb, unsigned char* inputf)
{
	for (int y = 1; y < HEIGHT - 1; y++) {
		for (int x = 1; x < WIDTH - 1; x++) {

			int mean = 0;
			for (int yWindow = -1; yWindow < 2; yWindow++) {
				int y2 = y + yWindow;
				for (int xWindow = -1; xWindow < 2; xWindow++) {
					int x2 = x + xWindow;
					mean += inputf[y2 * WIDTH + x2];
				}
			}
			mean = mean / 9;
			int temp = abs((mean - inputb[y * WIDTH + x]));

			if (temp > Threshold)
				outputimage[y * WIDTH + x] = 255;
			else
				outputimage[y * WIDTH + x] = 0;
		}
	}
}


// CUDA kernel background
__global__ void GPUBackground(unsigned char* d_output, unsigned char* d_inputb, unsigned char* d_inputf, int width, int threshold)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * width + col;

	if (row < width && col < width) {
		int mean = 0;
		for (int yWindow = -1; yWindow < 2; yWindow++) {
			int y2 = row + yWindow;
			for (int xWindow = -1; xWindow < 2; xWindow++) {
				int x2 = col + xWindow;
				mean += d_inputf[y2 * width + x2];
			}
		}
		mean = mean / 9;
		int temp = abs((mean - d_inputb[index]));
		if (temp > threshold)
			d_output[index] = 255;
		else
			d_output[index] = 0;
	}
}

/***********************************************************************************/

// El main puede tener como argumentos: nombres de los fichero de las imagenes (tiene que ser BMP) y el umbral
int main(int argc, char* argv[])
{
	double start_time_inc_data, end_time_inc_data;
	double cpu_start_time, cpu_end_time;

	unsigned char* d_inputb, * d_inputf, * d_output;

	unsigned char* output_image;

	/* Numero de argumentos */
	if (argc != 4)
	{
		fprintf(stderr, "Número de parametros incorecto\n");
		fprintf(stderr, "Uso: %s fondo.bmp imagen.bmp valorumbral\n", argv[0]);
		return -1;
	}

	// Leemos las im�genes 
	BMP Fondo, Image;
	Fondo.ReadFromFile(argv[1]);
	Image.ReadFromFile(argv[2]);

	// Leemos el valor del umbral por ejemplo 150 es un valor típico
	Threshold = atoi(argv[3]);

	BMP ResultadoGPU;
	BMP ResultadoCPU;

	// Calculo del tamaño de la imagen
	WIDTH = Fondo.TellWidth();
	HEIGHT = Fondo.TellHeight();

	// Establecemos el tamaño de la imagen de salida
	ResultadoGPU.SetSize(Fondo.TellWidth(), Fondo.TellHeight());
	ResultadoGPU.SetBitDepth(1);
	ResultadoCPU.SetSize(Fondo.TellWidth(), Fondo.TellHeight());
	ResultadoCPU.SetBitDepth(1);

	// Reserva memoria en el host para alojar la imagen
	input_imageb = (unsigned char*)calloc(HEIGHT * WIDTH, sizeof(unsigned char));
	input_imagef = (unsigned char*)calloc(HEIGHT * WIDTH, sizeof(unsigned char));
	gpu_output = (unsigned char*)calloc(HEIGHT * WIDTH, sizeof(unsigned char));
	output_image = (unsigned char*)calloc(HEIGHT * WIDTH, sizeof(unsigned char));


	for (int i = 0; i < WIDTH; i++)
	{
		for (int j = 0; j < HEIGHT; j++)
		{
			input_imageb[i * HEIGHT + j] = Fondo(i, j)->Red;
			input_imagef[i * HEIGHT + j] = Image(i, j)->Red;
		}
	}


	cudaSetDevice(0);

	// Calculamos memoria necesaria para alojar las imagenes 
	size_t memSize = WIDTH * HEIGHT * sizeof(unsigned char);

	/* Reservamos memoria en la GPU */
	cudaMalloc(&d_inputb, memSize);
	cudaMalloc(&d_inputf, memSize);
	cudaMalloc(&d_output, memSize);


	start_time_inc_data = get_current_time();

	/*
	* Copiamos todos los arrays a la memoria de la GPU
	*/
	cudaMemcpy(d_inputb, input_imageb, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputf, input_imagef, memSize, cudaMemcpyHostToDevice);

	/***********************************************************/
	// Ejecutar background en la GPU
	/* Ejecución kernel  */
	
	// Calculamos el tamaño de grid y del bloque
	dim3 threadsPerBlock(GRID_H, GRID_W, 1);
	dim3 numBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y, 1);

	printf("Grid size: %dx%d\n", GRID_W, GRID_H);
	printf("Block size: %dx%d\n", WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y, 1);

	GPUBackground <<<numBlocks, threadsPerBlock>>> (d_output, d_inputb, d_inputf, WIDTH, Threshold);

	// Copiamos de la memoria de la GPU 
	cudaMemcpy(gpu_output, d_output, memSize, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	end_time_inc_data = get_current_time();

	checkCUDAError("Background CUDA: ");

	/*************************************************************/

	/****************************
	* Versión CPU background
	*****************************/
	cpu_start_time = get_current_time();

	CPUBackground(output_image, input_imageb, input_imagef);

	cpu_end_time = get_current_time();

	/* Comprobamos que los resultados de la GPU coinciden con los calculados en la CPU */

	int errors = 0;
	for (int y = 1; y < HEIGHT - 1; y++) {
		for (int x = 1; x < WIDTH - 1; x++) {
			if (output_image[y * WIDTH + x] != gpu_output[y * WIDTH + x]) {
				errors++;
				printf("Error en %d,%d (CPU=%i, GPU=%i)\n", x, y, output_image[y *WIDTH + x], gpu_output[y*WIDTH + x]);
			}
		}
	}
	printf("Errores %d\n", errors);

	if (errors == 0) printf("\n\n ***TEST CORRECTO*** \n\n\n");

	cudaFree(d_inputb);
	cudaFree(d_inputf);
	cudaFree(d_output);

	printf("Tiempo ejecución GPU (Incluyendo transferencia de datos): %fs\n", \
		end_time_inc_data - start_time_inc_data);
	printf("Tiempo de ejecución en la CPU                          : %fs\n", \
		cpu_end_time - cpu_start_time);

	// Copiamos el resultado al formato de la libreria y guardamos el fichero BMP procesado
	for (int i = 0; i < WIDTH; i++)
	{
		for (int j = 0; j < HEIGHT; j++)
		{
			ResultadoGPU(i, j)->Red = gpu_output[i * HEIGHT + j];
			ResultadoGPU(i, j)->Green = gpu_output[i * HEIGHT + j];
			ResultadoGPU(i, j)->Blue = gpu_output[i * HEIGHT + j];
			ResultadoCPU(i, j)->Red = output_image[i * HEIGHT + j];
			ResultadoCPU(i, j)->Green = output_image[i * HEIGHT + j];
			ResultadoCPU(i, j)->Blue = output_image[i * HEIGHT + j];
		}
	}

	// Guardamos el resultado de aplicar el filtro en un nuevo fichero
	ResultadoGPU.WriteToFile("resultado_backgroundGPU.bmp");
	ResultadoCPU.WriteToFile("resultado_backgroundCPU.bmp");

	getchar();
	return 0;
}


/* Funciones auxiliares */

#if _WIN32
void getCurrentTimeStamp(timeStamp& _time)
{
	QueryPerformanceCounter(&_time);
}

timeStamp getCurrentTimeStamp()
{
	timeStamp tmp;
	QueryPerformanceCounter(&tmp);
	return tmp;
}

double getTimeMili()
{
	timeStamp start;
	timeStamp dwFreq;
	QueryPerformanceFrequency(&dwFreq);
	QueryPerformanceCounter(&start);
	return double(start.QuadPart) / double(dwFreq.QuadPart);
}
#endif 

double get_current_time()
{
#if _WIN32 
	return getTimeMili();
#else
	static int start = 0, startu = 0;
	struct timeval tval;
	double result;

	if (gettimeofday(&tval, NULL) == -1)
		result = -1.0;
	else if (!start) {
		start = tval.tv_sec;
		startu = tval.tv_usec;
		result = 0.0;
	}
	else
		result = (double)(tval.tv_sec - start) + 1.0e-6 * (tval.tv_usec - startu);
	return result;
#endif
}

/* Funci�n para comprobar errores CUDA */
void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


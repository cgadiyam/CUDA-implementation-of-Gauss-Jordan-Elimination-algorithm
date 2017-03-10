#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

const int TILE_SIZE = 16;

__global__ void ScaleRowKernel(float* matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outputMatrix, int current_column)
{
	int tx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int ty = (blockIdx.y*blockDim.y) + threadIdx.y;
	int tID = (ty*numberOfColumns)+tx;
	if(current_column == ty && ty < numberOfRows)
	{
		if(tx < numberOfColumns)
		{
			outputMatrix[tID] = matrix[tID]/matrix[(current_column*numberOfColumns)+current_column];
		}
	}
}

__global__ void SubtractRowKernel(float* matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outputMatrix, int current_column)
{
	int tx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int ty = (blockIdx.y*blockDim.y) + threadIdx.y;
	int tID = (ty*numberOfColumns)+tx;
	if(current_column != ty && ty < numberOfRows)
	{
		if(tx < numberOfColumns)
		{
			outputMatrix[tID] = matrix[tID] - (matrix[(current_column*numberOfColumns)+tx] * matrix[(ty*numberOfColumns)+current_column]);
		}
	}
}


bool GaussianEliminationGPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the matrix.
	int bytes = numberOfRows * numberOfColumns * sizeof(float);
	float *Md, *Pd;

	float *M = new float[bytes];
	float *P = new float[bytes];
	int count = 0;
	int rowID = 0;
	for(int i=0;i<numberOfRows;i++)
	{
		for(int j=0;j<numberOfColumns;j++)
		{
			M[count] = matrix[i][j];
			count++;
		}
	}
	// Allocate memory on the device to store each matrix
	cudaMalloc((void**) &Md, bytes);
	cudaMalloc((void**) &Pd, bytes);
	// Copy the host input data to the device
	cudaMemcpy(Md, M, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(Pd, Md, bytes, cudaMemcpyDeviceToDevice);
	// Specify the size of the grid and the size of the block
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)numberOfColumns / (float)TILE_SIZE),(int)ceil((float)numberOfRows / (float)TILE_SIZE));

	//std::cout << "\nnumber of rows: "<<numberOfRows;
	for(int i=0;i<numberOfRows;i++)
	{
		//cudaMemcpy(Md, M, bytes, cudaMemcpyHostToDevice);

		ScaleRowKernel<<<dimGrid, dimBlock>>>(Md, numberOfRows, numberOfColumns, Pd, i);
		cudaThreadSynchronize();
		// Check for errors
		status = cudaGetLastError();
		if (status != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
			cudaFree(Md);
			cudaFree(Pd);
			return false;
		}
		cudaMemcpy(Md, Pd, bytes, cudaMemcpyDeviceToDevice);
		//std::cout<<"\nscale row...";
		/*for(int j =0;j<(numberOfRows*numberOfColumns);j++)
		{
			rowID = j/numberOfColumns;
			if(rowID == i)
			{
				M[j] = P[j];
				//std::cout<<"\n"<<j<<"th element: "<<P[j];
			}
		}
		cudaMemcpy(Md, M, bytes, cudaMemcpyHostToDevice);*/

		SubtractRowKernel<<<dimGrid, dimBlock>>>(Md, numberOfRows, numberOfColumns, Pd, i);
		cudaThreadSynchronize();
		// Check for errors
		status = cudaGetLastError();
		if (status != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
			cudaFree(Md);
			cudaFree(Pd);
			return false;
		}
		cudaMemcpy(Md, Pd, bytes, cudaMemcpyDeviceToDevice);
		//std::cout<<"\nsubtract row...";
		/*for(int j =0;j<(numberOfRows*numberOfColumns);j++)
		{
			rowID = j/numberOfColumns;
			if(rowID != i)
			{
				M[j] = P[j];
				//std::cout<<"\n"<<j<<"th element: "<<P[j];
			}
		}*/
	}
	// Retrieve the result matrix
	cudaMemcpy(P, Md, bytes, cudaMemcpyDeviceToHost);
	//std::cout<<"\noutput matrix GPU: ";
	/*for(int i =0;i<(numberOfColumns*numberOfRows);i++)
	{
		std::cout<<M[i]<<"\t";
	}*/
	count = 0;
	for(int i=0;i<numberOfRows;i++)
	{
		for(int j=0;j<numberOfColumns;j++)
		{
			outputMatrix[i][j] = P[count];
			count++;
		}
	}
	// Free device memory
	cudaFree(Md);
	cudaFree(Pd);
	delete[] P;
	delete[] M;
	// Success
	return true;
}
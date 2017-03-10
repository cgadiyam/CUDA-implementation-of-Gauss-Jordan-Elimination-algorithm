#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <fstream>
#include <ctime>
#include <cuda_runtime_api.h>
#include "common.h"

const int ROWS = 174;
const int COLUMNS = 175;
const bool PARTIAL_PIVOT = false;
const int ITERS = 1;
std::ofstream MyFile;

void DisplayMatrix(float **Matrix)
{
	for(int i =0;i<ROWS;i++)
	{
		for(int j =0;j<COLUMNS;j++)
		{
			std::cout<<Matrix[i][j]<<"\t";
		}
		std::cout<<std::endl;
	}
}

float ComputeL2Norm(float** outputMatrixCPU, float** outputMatrixGPU)
{
	float sum = 0, delta = 0;
	for(int i =0;i<ROWS;i++)
	{
		for(int j =0;j<COLUMNS;j++)
		{
			delta += (outputMatrixCPU[i][j] - outputMatrixGPU[i][j]) * (outputMatrixCPU[i][j] - outputMatrixGPU[i][j]);
			sum += (outputMatrixCPU[i][j] * outputMatrixGPU[i][j]);
		}
	}
	float L2norm = sqrt(delta / sum);
	return L2norm;
}

int main()
{
	MyFile.open ("Matrix.txt", std::ofstream::out | std::ofstream::ate | std::ofstream::app | std::ofstream::binary) ;
	// Timing data
	float tcpu, tgpu;
	clock_t start, end;

	float sum = 0, delta = 0;
	bool status;
	float L2norm;

	float** InputMatrix = new float*[ROWS];
	for(int i =0;i<ROWS;i++)
		InputMatrix[i] = new float[COLUMNS];
	float** OutputMatrixCPU = new float*[ROWS];
	for(int i =0;i<ROWS;i++)
		OutputMatrixCPU[i] = new float[COLUMNS];
	float** OutputMatrixGPU = new float*[ROWS];
	for(int i =0;i<ROWS;i++)
		OutputMatrixGPU[i] = new float[COLUMNS];

	for(int i =0;i<ROWS;i++)
	{
		for(int j =0;j<COLUMNS;j++)
			InputMatrix[i][j] = (float)(rand())/(float)(RAND_MAX);
	}

	std::cout<<"\noperating on a "<<ROWS<<" x "<<COLUMNS<<" matrix"<<std::endl;

	//MyFile<<"\nInput Matrix:"<<std::endl;
	//DisplayMatrix(InputMatrix);

	// Gauss Jordan Elimination on CPU
	start = clock();
	for (int i = 0; i < ITERS; i++) 
	{
		GaussianEliminationCPU(InputMatrix, ROWS, COLUMNS, OutputMatrixCPU, PARTIAL_PIVOT);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "\nHost Result (direct) took " << tcpu << " ms" << std::endl;
	//MyFile<<"\nOutput Matrix CPU:"<<std::endl;
	//DisplayMatrix(OutputMatrixCPU);

	bool success = GaussianEliminationGPU(InputMatrix, ROWS, COLUMNS, OutputMatrixGPU, PARTIAL_PIVOT);
	if (!success)
	{
		MyFile << "\n * Device error! * \n" << std::endl;
		while(true);
		return 1;
	}
	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) 
	{
		GaussianEliminationGPU(InputMatrix, ROWS, COLUMNS, OutputMatrixGPU, PARTIAL_PIVOT);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "\nDevice Computation took " << tgpu << " ms" << std::endl;
	//MyFile<<"\nOutput Matrix GPU:"<<std::endl;
	//DisplayMatrix(OutputMatrixGPU);

	float L2Norm = ComputeL2Norm(OutputMatrixCPU, OutputMatrixGPU);
	std::cout<<"\nRelative Error: "<<L2Norm<<std::endl<<std::endl;

	MyFile.close();

	for (int i=0; i<ROWS; i++)
		delete [] InputMatrix[i];
	delete [] InputMatrix;

	for (int i=0; i<ROWS; i++)
		delete [] OutputMatrixCPU[i];
	delete [] OutputMatrixCPU;

	for (int i=0; i<ROWS; i++)
		delete [] OutputMatrixGPU[i];
	delete [] OutputMatrixGPU;

	//delete[] InputMatrix;
	//delete[] OutputMatrix;

	std::cout<<"\nBoom.......";

	while(true);

	return 0;
}
#include <cstdlib> // malloc(), free()
#include <iostream>
#include <stdio.h>
#include "common.h"

void GaussianEliminationCPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot )
{
	int column_in_operation = 0;
	bool exit = false;
	for(int i=0;i<numberOfRows;i++)
	{
		for(int j=0;j<numberOfColumns;j++)
		{
			outputMatrix[i][j] = matrix[i][j];
		}
	}
	int count = 0;
	float temp=0;
	while(!exit)
	{
		temp = outputMatrix[column_in_operation][column_in_operation];
		for(int j=0;j<numberOfColumns;j++)
		{
			if(outputMatrix[column_in_operation][j] != 0)
			{
				outputMatrix[column_in_operation][j] = (outputMatrix[column_in_operation][j])/temp;
			}
			//std::cout<<"\n"<<column_in_operation<<" row\t"<<j<<" column:\t"<<outputMatrix[column_in_operation][j]<<std::endl;
			count++;
		}
		//std::cout<<"\ncount: "<<count;

		for(int i=0;i<numberOfRows;i++)
		{
			temp = outputMatrix[i][column_in_operation];
			for(int j=0;j<numberOfColumns;j++)
			{
				if(i != column_in_operation)
				{
					outputMatrix[i][j] = outputMatrix[i][j] -  (temp * outputMatrix[column_in_operation][j]);
					//std::cout<<"\n"<<i<<" row\t"<<j<<" column:\t"<<outputMatrix[i][j]<<std::endl;
				}
			}
		}
		column_in_operation++;
		if(column_in_operation == numberOfRows)
		{
			exit = true;
		}
	}
}
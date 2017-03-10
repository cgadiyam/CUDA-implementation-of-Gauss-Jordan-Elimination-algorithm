/************************************************************************/
// Author: Jason Lowden
// Date: March 30, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: Gaussian.h
// The purpose of this program is to compare Gaussian Elimination on the
// GPU and CPU.
/************************************************************************/

#ifndef __GAUSSIAN_H__
#define __GAUSSIAN_H__

/**
 * Computes the CPU Gaussian algorithm
 * matrix - Input matrix to reduce
 * numberOfRows - number of rows
 * numberOfColumns - number of columns
 * outputMatrix - output matrix where the result is stored
 * partialPivot - flag to perform partial pivoting
 */
void Train_ELM(float** InputMatrix, unsigned int NumberOfSamples, unsigned int NumberOfFeatures, float* OutputMatrix, unsigned int NumberOfHiddenNeurons);

/**
 * Computes the GPU Gaussian algorithm
 * matrix - Input matrix to reduce
 * numberOfRows - number of rows
 * numberOfColumns - number of columns
 * outputMatrix - output matrix where the result is stored
 * partialPivot - flag to perform partial pivoting
 */
void Predict_ELM(float* InputTestSample, unsigned int NumberOfFeatures, float* PredictedOutput);

#endif
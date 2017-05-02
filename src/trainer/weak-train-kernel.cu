#ifndef _WEAK_TRAIN_H_
#define _WEAK_TRAIN_H_

#include <numeric>
#define FNUM 882

__const__ bool V[FNUM] = {false};
__const__ double W[FNUM] = {0.0};

__global__ void KernelWeakTrain(int *index, int testNum, double validweight, int* indexR, bool* goodR, double* errorR) {
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ int index[FNUM];
    __shared__ bool good[FNUM];
    __shared__ double error[FNUM];
    __shared__ double positive_error[FNUM] = {validweight};
    __shared__ double negative_error[FNUM] = {validweight};


    index[id] = 0;
    good[id] = true;
    error[id] = std::numeric_limits<double>::max();

    int pos = id*FNUM;
    for(int i=0; i<testNum; ++i){
        if (V[index[pos]]){
            positive_error[pos] -= W[index[pos]];

            if (positive_error[pos] < error[pos]){
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ + 1, 1, positive_error);
            }
        }
        else{
            positive_error[pos] += W[index[pos]];
            negative_error[pos]= 1.0 - positive_error[pos];

            if (negative_error[pos] < error[pos]){
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ - 1, -1, negative_error);
            }
        }
    }
    indexR[id] = index[id];
    goodR[id] = good[id]
    errorR[id] = errorR[id];
}

void select_best_gpu(int featureNum, int testNum, bool * valids, double * weights, double validweight, int* featureIndex,
    int & index, bool & good, double & error){

    cudaMemcpyToSymbol(V, valids, FNUM *sizeof(bool));
    cudaMemcpyToSymbol(W, weights, FNUM *sizeof(double));

    int * d_f_i;
    size_t bytes = featureNum  * testNum * sizeof( int );
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_f_i, bytes);   
 
    // Copy host vectors to device
    cudaMemcpy(d_f_i, featureIndex, bytes, cudaMemcpyHostToDevice);

    // Launch the device computation threads!
    int * d_i;
    bool * d_g;
    double * d_e;
    cudaMalloc(&d_i, featureNum*sizeof(int));
    cudaMalloc(&d_g, featureNum*sizeof(bool));
    cudaMalloc(&d_e, featureNum*sizeof(double));
    KernelWeakTrain<<<1, FNUM>>>(d_f_i, testNum, validweight, d_i, d_g, d_e);

    // Copy array back to host
    int* r_i = (int*)malloc(featureNum*sizeof(int));
    bool* r_g = (bool*)malloc(featureNum*sizeof(bool));
    double* r_e = (double*)malloc(featureNum*sizeof(double));
    cudaMemcpy(r_i, d_i, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(r_g, d_g, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(r_e, d_e, bytes, cudaMemcpyDeviceToHost); 

    index = 1;
    good = true;
    error = validweight;

    // Free device matrices
    cudaFree(d_f_i);
}

#endif // #ifndef _WEAK_TRAIN_H_
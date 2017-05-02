#ifndef _WEAK_TRAIN_H_
#define _WEAK_TRAIN_H_

#define FNUM 882

__const__ bool V[FNUM] = {false};
__const__ double W[FNUM] = {0.0};

__global__ void KernelWeakTrain(int *index, int testNum) {
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;


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
    KernelWeakTrain<<<1, FNUM>>>(d_f_i, testNum);

    index = 1;
    good = true;
    error = validweight;

    // Free device matrices
    cudaFree(d_f_i);
}

#endif // #ifndef _WEAK_TRAIN_H_
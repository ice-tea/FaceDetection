#ifndef _WEAK_TRAIN_H_
#define _WEAK_TRAIN_H_

__global__ void KernelWeakTrain(long *pic, int width, int height) {
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
}

void select_best_gpu(bool * valids, double * weights, double validweight, int featureNum, int* deatureIndex){

    // Allocate memory for each vector on GPU
    /*
    int featureNum = features_values.size();

    



    cudaMalloc(&d_pic, bytes);   
 
    // Copy host vectors to device
    cudaMemcpy(d_pic, h_i, bytes, cudaMemcpyHostToDevice);

    // Launch the device computation threads!
    IntegralKernelWidth<<<1, width>>>(d_pic, width, height);
    IntegralKernelHeight<<<1, height>>>(d_pic, width, height);

    // Copy array back to host
    cudaMemcpy(h_i, d_pic, bytes, cudaMemcpyDeviceToHost); 

    // Free device matrices
    cudaFree(d_pic);
    */
}

#endif // #ifndef _WEAK_TRAIN_H_
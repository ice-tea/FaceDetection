#ifndef _WEAK_TRAIN_H_
#define _WEAK_TRAIN_H_

#define TNUM 50
#define FNUM 882

//__const__ bool V[TNUM] = {false};
//__const__ double W[TNUM] = {0.0};

__global__ void KernelWeakTrain(int featureNum, int testNum, int *tindex, 
    double validweight, int* indexR, bool* goodR, double* errorR,
    bool * V, double * W) {
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    indexR[id] = 0;
    goodR[id] = true;
    errorR[id] = 2e20;


    double positive_error = validweight;
    double negative_error = validweight;

    int pos = id*TNUM;
    for(int i=0; i<testNum; ++i){
        if (V[tindex[pos]]){
            positive_error -= W[tindex[pos]];

            if (positive_error < error[id]){
                errorR[id] = positive_error[id];
                goodR[id] = true;
                indexR[id] = i;
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ + 1, 1, positive_error);
            }
        }
        else{
            positive_error += W[tindex[pos]];
            negative_error = 1.0 - positive_error;

            if (negative_error < error[id]){
                errorR[id] = negative_error;
                goodR[id] = false;
                indexR[id] = i;
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ - 1, -1, negative_error);
            }
        }
        pos++;
    }
    //indexR[id] = index[id];
    //goodR[id] = good[id];
    //errorR[id] = error[id];
}
void select_best_gpu(int featureNum, int testNum, bool * valids, double * weights, double validweight, int* featureIndex,
    int * indexResult, bool * goodResult, double * errorResult){

    //cudaMemcpyToSymbol(V, valids, TNUM *sizeof(bool));
    //cudaMemcpyToSymbol(W, weights, TNUM *sizeof(double));

    int * d_f_i;
    size_t bytes = featureNum  * testNum * sizeof( int );
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_f_i, bytes);   
 
    // Copy host vectors to device
    cudaMemcpy(d_f_i, featureIndex, bytes, cudaMemcpyHostToDevice);

    //constant
    bool * V;
    cudaMalloc(&V, testNum *sizeof(bool));
    cudaMemcpy(V, valids, testNum *sizeof(bool), cudaMemcpyHostToDevice);
    double * W;
    cudaMalloc(&W, testNum *sizeof(double));
    cudaMemcpy(W, weights, testNum *sizeof(double), cudaMemcpyHostToDevice);


    // Launch the device computation threads!
    int * d_i;
    bool * d_g;
    double * d_e;
    cudaMalloc(&d_i, featureNum *sizeof(int));
    cudaMalloc(&d_g, featureNum *sizeof(bool));
    cudaMalloc(&d_e, featureNum *sizeof(double));
    KernelWeakTrain<<<1, featureNum>>> (featureNum, testNum, d_f_i, validweight, d_i, d_g, d_e, V, W);

    // Copy array back to host
    cudaMemcpy(indexResult, d_i, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(goodResult, d_g, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(errorResult, d_e, bytes, cudaMemcpyDeviceToHost);

    // Free device matrices
    cudaFree(V);
    cudaFree(W);
    cudaFree(d_f_i);
    cudaFree(d_i);
    cudaFree(d_g);
    cudaFree(d_e);
}
#endif // #ifndef _WEAK_TRAIN_H_
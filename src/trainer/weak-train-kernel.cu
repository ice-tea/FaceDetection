#ifndef _WEAK_TRAIN_H_
#define _WEAK_TRAIN_H_

#define TNUM 6987
#define FNUM 882
#define TNUM 96

__constant__ bool V[TNUM];
__constant__ double W[TNUM];

__global__ void KernelWeakTrain(int featureNum, int testNum, int *tindex, 
    double validweight, int* indexR, bool* goodR, double* errorR
    /*, bool * V, double * W*/) {
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    indexR[id] = 0;
    goodR[id] = true;
    errorR[id] = 2e20;

    double positive_error = validweight;
    double negative_error = validweight;

    double local_best = validweight;
    bool loca_good = true;
    int local_index = 0;

    int pos = id*testNum;
    
    for(int i=0; i<testNum; ++i){
        if (V[tindex[pos]]){
            positive_error -= W[tindex[pos]];

            if (positive_error < local_best){
                local_best = positive_error;
                loca_good = true;
                local_index = i;
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ + 1, 1, positive_error);
            }
        }
        else{
            positive_error += W[tindex[pos]];
            negative_error = 1.0 - positive_error;

            if (negative_error < local_best){
                //errorR[id] = negative_error;
                //goodR[id] = false;
                //indexR[id] = i;
                local_best = negative_error;
                loca_good = false;
                local_index = i;
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ - 1, -1, negative_error);
            }
        }
        pos++;
    }
    
    indexR[id] = local_index;
    goodR[id] = loca_good;
    errorR[id] = local_best;
}
void select_best_gpu(int featureNum, int testNum, bool * valids, double * weights, double validweight, int* featureIndex,
    int * indexResult, bool * goodResult, double * errorResult){

    cudaMemcpyToSymbol(V, valids, testNum *sizeof(bool));
    cudaMemcpyToSymbol(W, weights, testNum *sizeof(double));

    int * d_f_i;
    cudaMalloc(&d_f_i, featureNum  * testNum * sizeof( int ));
    cudaMemcpy(d_f_i, featureIndex, featureNum  * testNum * sizeof( int ), cudaMemcpyHostToDevice);

    //constant
    /*
    bool * V;
    cudaMalloc(&V, testNum *sizeof(bool));
    cudaMemcpy(V, valids, testNum *sizeof(bool), cudaMemcpyHostToDevice);
    double * W;
    cudaMalloc(&W, testNum *sizeof(double));
    cudaMemcpy(W, weights, testNum *sizeof(double), cudaMemcpyHostToDevice);
    */

    // Launch the device computation threads!
    int * d_i;
    bool * d_g;
    double * d_e;
    cudaMalloc(&d_i, featureNum *sizeof(int));
    cudaMalloc(&d_g, featureNum *sizeof(bool));
    cudaMalloc(&d_e, featureNum *sizeof(double));

    KernelWeakTrain<<<(featureNum-1)/TNUM + 1, TNUM>>> (featureNum, testNum, d_f_i, validweight, d_i, d_g, d_e /*,V, W*/);

    // Copy array back to host
    cudaMemcpy(indexResult, d_i, featureNum *sizeof(int), cudaMemcpyDeviceToHost); 
    cudaMemcpy(goodResult, d_g, featureNum *sizeof(bool), cudaMemcpyDeviceToHost); 
    cudaMemcpy(errorResult, d_e, featureNum *sizeof(double), cudaMemcpyDeviceToHost);

    // Free device matrices
    //cudaFree(V);
    //cudaFree(W);
    cudaFree(d_f_i);
    cudaFree(d_i);
    cudaFree(d_g);
    cudaFree(d_e);
}
#endif // #ifndef _WEAK_TRAIN_H_
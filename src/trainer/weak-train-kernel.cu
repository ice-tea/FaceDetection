#ifndef _WEAK_TRAIN_H_
#define _WEAK_TRAIN_H_

#define TNUM 6987
#define FNUM 882

//__const__ bool V[TNUM] = {false};
//__const__ double W[TNUM] = {0.0};

__global__ void KernelWeakTrain(int *tindex, int testNum, double validweight, int* indexR, bool* goodR, double* errorR,
        bool * V, double * W) {
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ int index[FNUM];
    __shared__ bool good[FNUM];
    __shared__ double error[FNUM];
    __shared__ double positive_error[FNUM];
    __shared__ double negative_error[FNUM];

    index[id] = 0;
    good[id] = true;
    error[id] = 2e20;
    positive_error[id] = validweight;
    negative_error[id] = validweight;

    int pos = id*TNUM;
    for(int i=0; i<testNum; ++i){
        if (V[tindex[pos]]){
            positive_error[id] -= W[tindex[pos]];

            if (positive_error[id] < error[id]){
                error[id] = positive_error[id];
                good[id] = true;
                index[id] = i;
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ + 1, 1, positive_error);
            }
        }
        else{
            positive_error[id] += W[tindex[pos]];
            negative_error[id]= 1.0 - positive_error[id];

            if (negative_error[id] < error[id]){
                error[id] = negative_error[id];
                good[id] = false;
                index[id] = i;
              //best = TestWeakClassifier(feature, feature.values_[itest].value_ - 1, -1, negative_error);
            }
        }
        pos++;
    }
    indexR[id] = index[id];
    goodR[id] = good[id];
    errorR[id] = error[id];
}
void select_best_gpu(int featureNum, int testNum, bool * valids, double * weights, double validweight, int* featureIndex,
    int & index, bool & good, double & error){

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
    cudaMalloc(&V, TNUM*sizeof(bool));
    cudaMemcpy(V, valids, TNUM*sizeof(bool), cudaMemcpyHostToDevice);
    double * W;
    cudaMalloc(&W, TNUM*sizeof(double));
    cudaMemcpy(W, weights, TNUM*sizeof(double), cudaMemcpyHostToDevice);


    // Launch the device computation threads!
    int * d_i;
    bool * d_g;
    double * d_e;
    cudaMalloc(&d_i, featureNum*sizeof(int));
    cudaMalloc(&d_g, featureNum*sizeof(bool));
    cudaMalloc(&d_e, featureNum*sizeof(double));
    KernelWeakTrain<<<1, FNUM>>>(d_f_i, testNum, validweight, d_i, d_g, d_e, V, W);

    // Copy array back to host
    int* r_i = (int*)malloc(featureNum*sizeof(int));
    bool* r_g = (bool*)malloc(featureNum*sizeof(bool));
    double* r_e = (double*)malloc(featureNum*sizeof(double));
    cudaMemcpy(r_i, d_i, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(r_g, d_g, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(r_e, d_e, bytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<FNUM; ++i){
        if(r_e[i] < error){
            error = r_e[i];
            index = r_i[i];
            good = r_g[i];
        }
    }

    // Free device matrices
    cudaFree(V);
    cudaFree(W);
    cudaFree(d_f_i);
    cudaFree(d_i);
    cudaFree(d_g);
    cudaFree(d_e);
}
#endif // #ifndef _WEAK_TRAIN_H_
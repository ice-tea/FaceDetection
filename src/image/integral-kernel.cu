#ifndef _INTEGRAL_KERNEL_H_
#define _INTEGRAL_KERNEL_H_

__global__ void IntegralKernelWidth(long *pic, int width, int height) {
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    int startPos = id * width;
    if (startPos <= width * (height-1) ){
        for(int k=1; k<width; ++k){
            pic[startPos+1] += pic[startPos]; 
            startPos++;
        }
    }
}

__global__ void IntegralKernelHeight(long *pic, int width, int height) {
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    int startPos = id * height;
    if (startPos <= height * (width-1) ){
        for(int k=1; k<height; ++k){
            pic[startPos+ width] += pic[startPos];
            startPos += width;
        }
    }
}

void integral_kernel(long * h_i, int width, int height){
	long * d_pic;
	size_t bytes = width * height * sizeof( long );
	// Allocate memory for each vector on GPU
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
}

#endif // #ifndef _INTEGRAL_KERNEL_H_
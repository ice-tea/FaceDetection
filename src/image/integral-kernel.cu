#ifndef _INTEGRAL_KERNEL_H_
#define _INTEGRAL_KERNEL_H_

__global__ void IntegralKernel(long *pic, int width, int height) {
	
}

void integral_kernel(long * h_i, int width, int height){
	long * d_pic;
	size_t bytes = width * height * sizeof( long );
	// Allocate memory for each vector on GPU
    cudaMalloc(&d_pic, bytes);   
 
    // Copy host vectors to device
    cudaMemcpy(d_pic, h_i, bytes, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((width-1)/TILE_WIDTH + 1, (height-1)/TILE_WIDTH + 1, 1);

    // Launch the device computation threads!
    IntegralKernel<<<dimGrid, dimBlock>>>(d_pic, width, height);

    // Copy array back to host
    cudaMemcpy(h_i, d_pic, bytes, cudaMemcpyDeviceToHost); 

    // Free device matrices
    cudaFree(d_pic);
}

#endif // #ifndef _INTEGRAL_KERNEL_H_
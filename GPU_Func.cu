#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. �Լ��� Colab ȯ�濡�� �����ؾ� �մϴ�.
// 2. �����Ӱ� �����ϼŵ� ������ ��� �Լ����� GPU�� Ȱ���ؾ� �մϴ�.
// 3. CPU_Func.cu�� �ִ� Image_Check�Լ����� True�� Return�Ǿ�� �ϸ�, CPU�ڵ忡 ���� �ӵ��� ����� �մϴ�.
/////////////////////////////////////////////////////////////////////////

// Colab Infomation
// 40 MP, 32 Warp, Max 1024 thread per block
// Shared memory per block = 2^14 * 3, Register per block = 2^16

//Block's Dimension
#define TILE_WIDTH 32
#define TILE_SIZE 1024

//2 Padding Tile Dimension
#define G_SHARED_WIDTH 36
#define G_SHARED_SIZE 1296
//Gaussian FIlter Dimension
#define G_FILTER_WIDTH 5
#define G_FILTER_SIZE 25

//1 Padding Tile Dimension
#define S_SHARED_WIDTH 34
#define S_SHARED_SIZE 1156
//Sobel FIlter Dimension
#define S_FILTER_WIDTH 3
#define S_FILTER_SIZE 9

//Sobel Filter(Constant memory);
__constant__ char sobel_filter_x[S_FILTER_SIZE];
__constant__ char sobel_filter_y[S_FILTER_SIZE];

//GrayScale CUDA Func.
__global__ void Cuda_Grayscale(unsigned char *output, unsigned char *input, const int len){
    // 0 <= x < 1024
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    //convert(x <= len)
    if(x <= len) {
        unsigned char temp = input[x] * 0.114f + input[x + 1] * 0.587f + input[x + 2] * 0.299f;
        output[x] = temp;
        output[x+1] = temp;
        output[x+2] = temp;
    }

    return;
}

//Noise Reduction CUDA Func.
__global__ void Cuda_Noise_Reduction(unsigned char *output, const unsigned char *input, const int width, const int height){
    __shared__ unsigned char shared_mem[G_SHARED_SIZE];     //2 padding shared memory
    __shared__ float gaussian_filter[G_FILTER_SIZE];        //gaussian filter
    
    //current index info
    int dx = threadIdx.x, dy = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + dx, idy = blockIdx.y * blockDim.y + dy;
    
    //Reindexing(x = 0, 2, ... , 34 / y = 0, 1, ...) 
    int shared_x = ((dy * blockDim.y + dx) * 2) % G_SHARED_WIDTH;
    int shared_y = ((dy * blockDim.y + dx) * 2) / G_SHARED_WIDTH;

    //y < 36 case : 2 input -> shared memeory
    if(shared_y < G_SHARED_WIDTH){
        //padding index info
        int padding_x = blockIdx.x * blockDim.x - 2 + shared_x;
        int padding_y = blockIdx.y * blockDim.y - 2 + shared_y;
        int padding_idx = padding_y * width + padding_x;
        
        //when padding index exists, copy
        if(padding_idx >= 0 && padding_idx < width * height)
            shared_mem[shared_y * G_SHARED_WIDTH + shared_x] = input[padding_idx * 3];
        if(padding_idx >= -1 && padding_idx + 1 < width * height)
            shared_mem[shared_y * G_SHARED_WIDTH + shared_x + 1] = input[(padding_idx + 1) * 3];
    }   //y >= 36 case : calculate gaussian filter
    else{
        float sigma = 1.0;
        //reindexing for gaussian filter
        int filter_idx = dy * blockDim.y + dx - (G_SHARED_SIZE / 2);
        int i = filter_idx / G_FILTER_WIDTH - 2;
        int j = filter_idx % G_FILTER_WIDTH - 2;
        if(i < 3)   //calculate gaussian filter(using CUDA Fast Math Func.)
            gaussian_filter[(i + 2) * 5 + (j + 2)] = (1 / (2 * 3.14f * sigma * sigma)) * __expf(-(i * i + j * j) / (2 * sigma * sigma));
    }

    __syncthreads();

    //turn off threads(out of boundary)
    if(idy >= height || idx >= width) return;

    //5 * 5 Matrix Conv.(Gaussian filter)
    float temp = 0.f;
    for(int i = 0; i < G_FILTER_WIDTH && idy - 2 + i < height; i++){
        if(idy - 2 + i < 0) continue;           //zero padding area 
        for(int j = 0; j < G_FILTER_WIDTH && idx - 2 + j < width; j++){
            if(idx - 2 + j < 0) continue;       //zero padding area
            temp += shared_mem[(dy + i) * G_SHARED_WIDTH + (dx + j)] * gaussian_filter[i * G_FILTER_WIDTH + j];
        }
    }

    //Write output
    int cur_idx = (idy * width + idx) * 3;
    output[cur_idx] = (unsigned char)temp;
    output[cur_idx + 1] = (unsigned char)temp;
    output[cur_idx + 2] = (unsigned char)temp;

    return;
}

//Intensity Gradient CUDA Func.
__global__ void Cuda_Intensity_Gradient(unsigned char *sobel, unsigned char *angle, const unsigned char *input
                                        , const int width, const int height){
    __shared__ unsigned char shared_mem[S_SHARED_SIZE];     //1 padding shared memory

    //Current index info
    int dx = threadIdx.x, dy = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + dx, idy = blockIdx.y * blockDim.y + dy;
    
    //Reindexing(x = 0, 2, ... , 32 / y = 0, 1, ... )
    int shared_x = ((dy * blockDim.y + dx) * 2) % S_SHARED_WIDTH;
    int shared_y = ((dy * blockDim.y + dx) * 2) / S_SHARED_WIDTH;

    //y < 34 case : 2 input -> shared memory
    if(shared_y < S_SHARED_WIDTH){
        //padding index info
        int padding_x = blockIdx.x * blockDim.x - 1 + shared_x;
        int padding_y = blockIdx.y * blockDim.y - 1 + shared_y;
        int padding_idx = padding_y * width + padding_x;
        
        //when padding index exists, copy
        if(padding_idx >= 0 && padding_idx < width * height)
            shared_mem[shared_y * S_SHARED_WIDTH + shared_x] = input[padding_idx * 3];
        if(padding_idx >= -1 && padding_idx + 1 < width * height)
            shared_mem[shared_y * S_SHARED_WIDTH + shared_x + 1] = input[(padding_idx + 1) * 3];
    }

    __syncthreads();

    //turn off threads(out of boundary)
    if(idy >= height || idx >= width) return;

    //3 * 3 Matrix Conv.(Sobel filter)
    int gx = 0, gy = 0;
    for(int i = 0; i < S_FILTER_WIDTH && idy - 1 + i < height; i++){
        if(idy - 1 + i < 0) continue;       //zero padding area
        for(int j = 0; j < S_FILTER_WIDTH && idx - 1 + j < width; j++){
            if(idx - 1 + j < 0) continue;   //zero padding area

            gx += (int)shared_mem[(dy + i) * S_SHARED_WIDTH + (dx + j)] * sobel_filter_x[i * S_FILTER_WIDTH + j];   //x_filter Conv.
            gy += (int)shared_mem[(dy + i) * S_SHARED_WIDTH + (dx + j)] * sobel_filter_y[i * S_FILTER_WIDTH + j];   //y_filter Conv.
        }
    }

    //(gx, gy)'s length(using CUDA Fast Math Func.)
    int t = __fsqrt_rn(gx * gx + gy * gy);
    unsigned char v = 0;
    if(t > 255) v = 255;
    else    v = (unsigned char)t;

    //input sobel result;
    int cur_idx = idy * width + idx;
    sobel[cur_idx * 3] = v;
    sobel[cur_idx * 3 + 1] = v;
    sobel[cur_idx * 3 + 2] = v;

    //Calculate angle
	float t_angle = 0;
	if(gy != 0 || gx != 0) 
		t_angle= (float)atan2f(gy, gx) * 180.0f / 3.14f;      //calculate angle(using CUDA Math Func. -> __atan2f() doesn't exist)
	if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))     //angle 0 or 180 case
		angle[cur_idx] = 0;
	else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))    //angle 45 or 225 case
		angle[cur_idx] = 45;
	else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))    //angle 90 or 270 case
		angle[cur_idx] = 90;
	else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))    //angle 135 or 315 case
		angle[cur_idx] = 135;

    return;
}

//Non Matximum Suppression CUDA Func.
__global__ void Cuda_Non_maximum_Suppression(unsigned char *output, unsigned char *minmax, const unsigned char *sobel
                                            , const unsigned char *angle, const int width, const int height){
    __shared__ unsigned char shared_mem[TILE_SIZE];     //no padding shared memory
    
    //index info
    int dx = threadIdx.x, dy = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + dx, idy = blockIdx.y * blockDim.y + dy;
    int cur_idx = (idy * width + idx) * 3;
    
    //shared memory's index
    int shared_idx = dy * TILE_WIDTH + dx;

    //1 input -> shared memory(in boundary)
    if(idx < width && idy < height)
        shared_mem[shared_idx] = sobel[cur_idx];
    
    __syncthreads();

    //Suppression
    if(idy != 0 && idy < height - 1 && idx != 0 && idx < width - 1){
        //read angle
        unsigned char cur_angle = angle[idy * width + idx];
	    uint8_t p1 = 0;
	    uint8_t p2 = 0;
	    
        //set p1, p2(if out of tile, read from sobel, else read from shared memory)
        if (cur_angle == 0) {
	    	p1 = (dy == TILE_WIDTH - 1) ? sobel[cur_idx + width * 3] : shared_mem[shared_idx + TILE_WIDTH];
	    	p2 = (dy == 0) ? sobel[cur_idx - width * 3] : shared_mem[shared_idx - TILE_WIDTH];
	    }
	    else if (cur_angle == 45) {
	        p1 = (dy == TILE_WIDTH - 1 || dx == 0) ? sobel[cur_idx + width * 3 - 3] : shared_mem[shared_idx + TILE_WIDTH - 1];
    	    p2 = (dy == 0 || dx == TILE_WIDTH - 1) ? sobel[cur_idx - width * 3 + 3] : shared_mem[shared_idx - TILE_WIDTH + 1];
	    }
	    else if (cur_angle == 90) {
            p1 = (dx == TILE_WIDTH - 1) ? sobel[cur_idx + 3] : shared_mem[shared_idx + 1];
	    	p2 = (dx == 0) ? sobel[cur_idx - 3] : shared_mem[shared_idx - 1];
	    }
	    else {
            p1 = (dy == TILE_WIDTH - 1 || dx == TILE_WIDTH - 1) ? sobel[cur_idx + width * 3 + 3] : shared_mem[shared_idx + TILE_WIDTH + 1];
    	    p2 = (dy == 0 || dx == 0) ? sobel[cur_idx - width * 3 - 3] : shared_mem[shared_idx - TILE_WIDTH - 1];
	    }

        //check current vertex, p1, p2's value -> write output
	    uint8_t v = shared_mem[shared_idx];
	    if ((v >= p1) && (v >= p2)) {
	    	output[cur_idx] = v;
	    	output[cur_idx + 1] = v;
	    	output[cur_idx + 2] = v;
	    }
	    else {
	    	output[cur_idx] = 0;
	    	output[cur_idx + 1] = 0;
	    	output[cur_idx + 2] = 0;
	    }
    }

    //turn off threads(out of boundary)
    if(idy >= height || idx >= width)   return;

    //color check shared momory
    __shared__ unsigned char check[256];
    check[shared_mem[shared_idx]] = 1;      //turn on each color

    __syncthreads();

    //write local min in minmax
    if(shared_idx == 0)
        for(int i = 0; i < 256; i++)    if(check[i] == 1){  minmax[i] = 1;  return; }
    //write local max in minmax
    if(shared_idx == 1)
        for(int i = 255; i >= 0; i--)   if(check[i] == 1){  minmax[i] = 1;  return; }
    
    return;
}

//Hysteresis Thresholding CUDA Func.
__global__ void Cuda_Hysteresis_Thresholding(unsigned char *output, const unsigned char *input, const int width, const int height
                                            , unsigned char min, unsigned char max){
    __shared__ unsigned char shared_mem[S_SHARED_SIZE];     //1 padding shared memory
    
    //current index info.
    int dx = threadIdx.x, dy = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + dx, idy = blockIdx.y * blockDim.y + dy;
    
    //Reindexing(x = 0, 2, ... , 32 / y = 0, 1, ... )
    int shared_x = ((dy * blockDim.y + dx) * 2) % S_SHARED_WIDTH;
    int shared_y = ((dy * blockDim.y + dx) * 2) / S_SHARED_WIDTH;
    
    //Calculate 2 Threshold
    unsigned char diff = max - min;
	unsigned char low_t = min + diff * 0.01f;
	unsigned char high_t = min + diff * 0.2f;
    
    unsigned char temp;
    //2 input -> shared memory(with thresholding)
    if(shared_y < S_SHARED_WIDTH){
        //padding index info
        int padding_x = blockIdx.x * blockDim.x - 1 + shared_x;
        int padding_y = blockIdx.y * blockDim.y - 1 + shared_y;
        int padding_idx = padding_y * width + padding_x;
        
        //when padding index exists, thresholding
        if(padding_idx >= 0 && padding_idx < width * height){
            temp = input[padding_idx * 3];
            shared_mem[shared_y * S_SHARED_WIDTH + shared_x] = temp < low_t ? 0 : (temp < high_t ? 123 : 255);
        }
        if(padding_idx >= -1 && padding_idx + 1 < width * height){
            temp = input[(padding_idx + 1) * 3];
            shared_mem[shared_y * S_SHARED_WIDTH + shared_x + 1]  = temp < low_t ? 0 : (temp < high_t ? 123 : 255);
        }
    }

    __syncthreads();

    //turn off threads(out of boundary)
    if(idy >= height || idx >= width)   return;

    //Hysteresis
    int shared_idx = (dy + 1) * S_SHARED_WIDTH + dx + 1;
    temp = shared_mem[shared_idx];
    //Do only week pixel
    for(int i = -1; i < 2 && temp == 123 && idy + i < height; i++){
        if(idy + i < 0)     continue;       //zero padding area
        for(int j = -1; j < 2 && idx + j < width; j++){
            if(idx + j < 0) continue;       //zero padding area
            //strong pixel exists
            if(shared_mem[shared_idx + i * S_SHARED_WIDTH + j] == 255){
                temp = 255;
                break;
            }
        }
    }

    //if temp is 123 : make 0
    temp = (temp == 255) ? 255 : 0;

    //write output
    int cur_idx = (idy * width + idx) * 3;
    output[cur_idx] = temp;
    output[cur_idx + 1] = temp;
    output[cur_idx + 2] = temp;    

    return;
}

//GPU Grayscale Host Func.
void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {
    uint8_t *dev_in, *dev_out;      //device pointer
    
    //Set block, grid dimension : block = 1D 1024 threads, grid = 1D
    int grid_width = (int)ceilf((float)((len - start_add) / 3) / TILE_SIZE);
    dim3 blockDim(TILE_SIZE);
    dim3 gridDim(grid_width);

    //Device Memory Allocation
    cudaMalloc(&dev_in, sizeof(uint8_t) * (len + 2 - start_add));
    cudaMalloc(&dev_out, sizeof(uint8_t) * (len + 2 - start_add));

    //Copy(Host -> Device)
    cudaMemcpy(dev_in, buf + start_add, sizeof(uint8_t) * (len + 2 - start_add), cudaMemcpyHostToDevice);
    
    //Call Cuda_Grayscale
    Cuda_Grayscale<<<gridDim, blockDim>>> ((unsigned char *)dev_out, (unsigned char *)dev_in, len - start_add);
    
    //Copy(Device -> Host)
    cudaMemcpy(gray + start_add, dev_out, sizeof(uint8_t) * (len + 2 - start_add), cudaMemcpyDeviceToHost);

    //Device Memory Deallocation
    cudaFree(dev_in);
    cudaFree(dev_out);
}

//GPU Noise Reduction Host Func.
void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian) {
    uint8_t *dev_in, *dev_out;      //device pointer

    //Set block, grid dimension : block = 2D 32 * 32 threads, grid = 2D
    int grid_width = (int)ceilf((double)width / TILE_WIDTH);
    int grid_height = (int)ceilf((double)height / TILE_WIDTH);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(grid_width, grid_height);

    //Device Memory Allocation
    cudaMalloc(&dev_in, sizeof(uint8_t) * (width * height * 3));
    cudaMalloc(&dev_out, sizeof(uint8_t) * (width * height * 3));

    //Copy(Host -> Device)
    cudaMemcpy(dev_in, gray, sizeof(uint8_t) * (width * height * 3), cudaMemcpyHostToDevice);
 
    //Call Cuda_Noise_Reduction
    Cuda_Noise_Reduction<<<gridDim, blockDim>>> ((unsigned char *)dev_out, (unsigned char *)dev_in, width, height);

    //Copy(Device -> Host)
    cudaMemcpy(gaussian, dev_out, sizeof(uint8_t) * (width * height * 3), cudaMemcpyDeviceToHost);    

    //Device Memory Deallocation
    cudaFree(dev_in);
    cudaFree(dev_out);
}

//GPU Intensity Gradient Host Func.
void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle){
    uint8_t *dev_in, *dev_sobel, *dev_angle;      //device pointer

    //Set block, grid dimension : block = 2D 32 * 32 threads, grid = 2D
    int grid_width = (int)ceilf((double)width / TILE_WIDTH);
    int grid_height = (int)ceilf((double)height / TILE_WIDTH);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(grid_width, grid_height);

    //Sobel filter
    char filter_x[S_FILTER_SIZE] = { -1, 0, 1
                                    , -2, 0, 2
                                    , -1, 0, 1};
    char filter_y[S_FILTER_SIZE] = { 1, 2, 1
                                    , 0, 0, 0
                                    , -1, -2, -1};
    
    //Sobel filter -> constant memory
    cudaMemcpyToSymbol(sobel_filter_x, &filter_x, sizeof(char) * S_FILTER_SIZE);
    cudaMemcpyToSymbol(sobel_filter_y, &filter_y, sizeof(char) * S_FILTER_SIZE);    

    //Device Memory Allocation  
    cudaMalloc(&dev_in, sizeof(uint8_t) * (width * height * 3));
    cudaMalloc(&dev_sobel, sizeof(uint8_t) * (width * height * 3));
    cudaMalloc(&dev_angle, sizeof(uint8_t) * (width * height));

    //Copy(Host -> Device)
    cudaMemcpy(dev_in, gaussian, sizeof(uint8_t) * (width * height * 3), cudaMemcpyHostToDevice);

    //Call Cuda_Intensity_Gradient
    Cuda_Intensity_Gradient<<<gridDim, blockDim>>> ((unsigned char *)dev_sobel, (unsigned char *)dev_angle
                                                    , (unsigned char *)dev_in, width, height);
    
    //Copy(Device -> Host)
    cudaMemcpy(sobel, dev_sobel, sizeof(uint8_t) * (width * height * 3), cudaMemcpyDeviceToHost);
    cudaMemcpy(angle, dev_angle, sizeof(uint8_t) * (width * height), cudaMemcpyDeviceToHost);    

    //Device Memory Deallocation
    cudaFree(dev_in);
    cudaFree(dev_sobel);
    cudaFree(dev_angle);
}

//GPU Non Maximum Suppression Host Func.
void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle, uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max){
    uint8_t *dev_angle, *dev_sobel, *dev_out, *dev_minmax;      //device pointer

    //Set block, grid dimension : block = 2D 32 * 32 threads, grid = 2D
    int grid_width = (int)ceilf((double)width / TILE_WIDTH);
    int grid_height = (int)ceilf((double)height / TILE_WIDTH);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(grid_width, grid_height);

    //pixel minmax check array
    uint8_t minmax[256];

    //Device Memory Allocation
    cudaMalloc(&dev_angle, sizeof(uint8_t) * (width * height));
    cudaMalloc(&dev_sobel, sizeof(uint8_t) * (width * height * 3));
    cudaMalloc(&dev_out, sizeof(uint8_t) * (width * height * 3));
    cudaMalloc(&dev_minmax, sizeof(uint8_t) * 256);

    //Copy(Host -> Device)
    cudaMemcpy(dev_angle, angle, sizeof(uint8_t) * (width * height), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sobel, sobel, sizeof(uint8_t) * (width * height * 3), cudaMemcpyHostToDevice);

    //Call Cuda_maximum_Suppression
    Cuda_Non_maximum_Suppression<<<gridDim, blockDim>>> ((unsigned char *)dev_out, (unsigned char * )dev_minmax, (unsigned char *)dev_sobel
                                                        , (unsigned char *)dev_angle, width, height);
    
    //Copy(Device -> Host)
    cudaMemcpy(suppression_pixel, dev_out, sizeof(uint8_t) * (width * height * 3), cudaMemcpyDeviceToHost);
    cudaMemcpy(minmax, dev_minmax, sizeof(uint8_t) * 256, cudaMemcpyDeviceToHost);

    //find global min
    for(int i = 0; i < 256; i++)    if(minmax[i] == 1){ min = i;    break;  }
    //find global max
    for(int i = 255; i >= 0; i--)   if(minmax[i] == 1){ max = i;    break;  }

    //Device Memory Deallocation    
    cudaFree(dev_angle);
    cudaFree(dev_sobel);
    cudaFree(dev_out);
    cudaFree(dev_minmax);
}

//GPU Hysteresis Thresholding Host Func.
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max) {
    uint8_t *dev_in, *dev_out;      //device pointer

    //Set block, grid dimension : block = 2D 32 * 32 threads, grid = 2D
    int grid_width = (int)ceilf((double)width / TILE_WIDTH);
    int grid_height = (int)ceilf((double)height / TILE_WIDTH);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(grid_width, grid_height);

    //Device Memory Allocation
    cudaMalloc(&dev_in, sizeof(uint8_t) * (width * height * 3));
    cudaMalloc(&dev_out, sizeof(uint8_t) * (width * height * 3));

    //Copy(Host -> Device)
    cudaMemcpy(dev_in, suppression_pixel, sizeof(uint8_t) * (width * height * 3), cudaMemcpyHostToDevice);

    //Call Cuda_Hysteresis_Thresholding
    Cuda_Hysteresis_Thresholding<<<gridDim, blockDim>>> ((unsigned char *)dev_out, (unsigned char *)dev_in, width, height, min, max);
    
    //Copy(Device -> Host)
    cudaMemcpy(hysteresis, dev_out, sizeof(uint8_t) * (width * height * 3), cudaMemcpyDeviceToHost);    

    //Device Memory Deallocation 
    cudaFree(dev_in);
    cudaFree(dev_out);
}
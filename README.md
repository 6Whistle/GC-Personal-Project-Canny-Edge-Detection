# GPU Computing Project

### 이름 : 이준휘

### 학번 : 2018202046

### 교수 : 공영호 교수님

### 강의 시간 : 월 수


## 1. Introduction

```
해당 과제는 CUDA를 통해 Canny Edge Detection을 구현하는 프로젝트다. Canny Edge
Detection이란 Image의 Edge를 검출하는 대중적인 알고리즘으로 총 4 단계로 나누어져
있다. 첫 번째로 Noise Reduction 단계에서는 이미지의 Noise를 Gaussian Filter로 Blurring
처리를 통해 Edge 검출을 용이하게 한다. 다음 단계로 Intensity Gradient에서는 Image의
Pixel이 Edge인지를 판단하기 위한 Gradient와 방향을 Sobel filter와의 Convolution을 통
해 구한다. Non-maximum Suppression 단계는 Edge가 아닌 Pixel을 제거하기 위해 해당
방향의 픽셀들과 값을 비교하여 pixel을 정리한다. 마지막으로 Hysteresis Thresholding 단
계에서는 Double Thresholding을 통해 Pixel 값을 3 가지로 정리한 후 만약 약한 연결인
경우 주변의 픽셀이 강하지 않으면 해당 픽셀을 정리하는 연산을 통해 최종 Edge를 검출
한다. 위의 코드를 GPU를 통해 구현하여 수행 시간을 CPU 연산에 대비 감소시키는 것을
목표로 한다.
```
## 2. Background

```
GPU의 구조는 CPU와 차이가 있다. 기본적으로 CPU는 하나의 Controller가 하나의
thread를 통제한다면, GPU는 하나의 Controller가 여러 개의 thread를 통제한다. 이는 즉
GPU는 반복적인 연산을 수행할 시 여러 thread를 통해 병렬적으로 빠르게 연산을 수행
할 수 있다는 것이다. 하지만 하나의 Controller만 사용하기 때문에 각각의 Thread가 많이
다른 연산을 수행하진 못한다.
하나의 Controller는 보통 32 개의 thread, 즉 warp 단위로 scheduling을 수행하며, warp는
하나의 block을 쪼개어 수행하는 것이다. 이러한 block은 보통 MP(Multi Processor) 단위
로 수행된다.
GPU의 메모리 구조 또한 일반적인 구조와 다르다. 우선 각 block 단위로 접근할 수 있는
Register가 존재한다. 또한 MP 내에는 Shared Memory라는 영역이 있어, block 단위로 데
이터를 공유하며, 빠른 속도로 데이터에 접근할 수 있다. Constant Memory는 읽기만 가능
한 Memory 영역으로 모든 MP에서 접근할 수 있으며 cached 되어있기에 속도 또한 매
우 빠르다. local Memory는 Device에 있는 VRAM을 의미하며 모든 block에서 접근할 수
있는 장점이 있지만 cached되어있지 않기에 속도가 매우 느리다는 단점이 있다. 마지막
으로 global Memory는 local memory가 부족할 시 Main Memory의 일부를 사용하는 것
으로 이 또한 거리가 멀기 때문에 느리다는 단점이 있다.
```
## 3. Code explanation & Optimization

```
a. Value
```

해당 사진은 전역을 Define된 값을 보인다. 기본적으로 Colab의 환경을 확인하였을 때 1
개의 block 당 1024 개의 Thread를 사용할 수 있다. 이후에 shared memory를 사용할 때
공유할 수 있는 부분을 최대로 하기 위해 TILE_WIDTH를 32 로, TILE_SIZE를 1024 로 설정하
였다. 최대로 설정할 경우 이후 padding된 shared memory를 사용할 때 해당 크기가 전
체적으로 확인하였을 때 줄어들기 때문이다.

G_SHARED_XX 변수는 2 개의 값을 zero padding하는 Noise Reduction에서 사용할
shared memory의 크기로 WIDTH = 36, SIZE = 36 * 36으로 설정하였다. G_FILTER_XX는
Gaussian filter의 크기를 나타낸다.

S_SHARED_XX 변수는 1 개의 값을 zero padding하는 곳에서 사용할 shared memory의
크기로 WIDTH = 34, SIZE = 34 * 34로 설정하였다. S_FILTER_XX는 Sobel Filter의 크기를 나
타낸다.

__constant__를 통해 constant memory에 선언된 char형 sobel_filter_x, sobel_filter_y는 x
축, y축 sobel filter로 추후에 데이터가 입력될 예정이다. 해당 값은 계산이 필요 없기에
constant로 선언하여 보다 빠르게 접근할 수 있도록 하였다.

b. Gray Scale

```
기존 이미지를 Gray scale의 이미지로 변환하는 단계로, RGB의 각 값을 확인하고 이를
하나의 값으로 통일하는 역할을 수행한다.
```

해당 함수는 host에서 call할 device의 Cuda_Grayscale 함수로 output으로 값을 반환하
며, input으로 값을 입력받고, len을 통해 길이를 알 수 있다.

우선 x에 현재 index의 위치를 가져온다. 해당 함수는 단순하게 1D로 수행될 예정이기
에 x축만을 사용하였다. 3 을 곱하는 이유는 RGB 3개의 색상이기에 다음 Pixel을 보기 위해
서는 3 씩 이동해야 하기 때문이다.

해당 동작은 len의 길이 내에 있을 경우에만 동작해야 하기에 조건을 달았으며, 3 번의
읽기와 3 번의 쓰기가 필수적으로 동작하기에 따로 필요한 shared memory가 필요 없다.
각 색상을 읽어 float 연산을 통해 Gray 값으로 변환하며 이를 register temp에 저장한다.
이후 해당 값을 output의 같은 index 위치에 입력한다.

다음 함수는 host에서 수행하는 부분이다. dev_in과 dev_out pointer는 device에서 입력과
출력으로 사용할 메모리의 주소값이다. grid_width는 (len – start_add) / 3 을 통해 Pixel의
수를 계산하고, 이를 1024(TILE_SIZE)로 나눈 값을 ceilf()를 통해 올림하여 모든 pixel이 동
작할 수 있도록 하였다.

cudaMalloc을 통해 dev_in과 dev_out에 Memory를 len+ 2 – start_add(실제 pixel 길이)만
큼 할당한 후 cudaMemcpy를 통해 buf+start_add(실제 Pixel 시작 주소)부터 길이만큼을


dev_out에 복사(host -> device)한다. 이후 Cuda_Grayscale Kernel을 호출하고, 결과가 저장
된 dev_out에서 gray+start_add로 값을 복사(device -> host)한다.

c. Noise Reduction

기존 Grayscale image에서 Gaussian filter를 사용하여 blurring을 수행한다.

해당 함수는 host에서 Call할 Cuda_Noise_Reduction 함수로 output에 결과를 작성하며,
input에 기존 데이터가 존재하고, 사진의 크기가 width, height를 통해 주어진다.


해당 함수는 __shared__를 통해 Tile에서 상하좌우로 2 씩 padding된 크기의 shared_mem
과 gaussian filter를 사용한다. padding된 크기로 shared memory를 사용하는 이유는
Convolution 동작에서 외각 부분을 local memory에서 가져올 경우 최대 16 번(모서리 부
분)의 local memory access를 해야 한다. 이는 매우 큰 성능 저하로 이어지기 때문에
padding된 크기로 shared memory를 사용하는 것이다. gaussian filter의 경우 25 번의 반복
을 각 thread별로 동작 시킬 시 각각 1 번의 동작으로 성능이 향상될 수 있기에 shared
memory로 생성하였다.

shared_x, shared_y는 shared memory에서 해당 thread가 접근할 shared memory가 어디
인지를 나타내며 기존 32 * 32 index를 2 배로 늘린 후 이를 G_SHARED_WIDTH로 나누거
나 몫을 구하여 x, y의 위치를 설정한다. 이렇게 될 경우 각 x의 값은 0 이상
G_SHARED_WIDTH 미만의 짝수 값으로 설정되며, Y의 값은 0 부터 증가하는 정수가 된다.

이후 y가 G_SHARED_WIDTH 미만인 경우에는 shared memory 행렬에 값을 넣는 동작을
수행한다. 현재 shared_x, y가 가리키는 실제 메모리 위치를 확인하고 해당 값이 실제 접
근 가능한 메모리 위치일 경우 해당 위치의 값을 가져와 shared memory에 삽입한다. 이
때 x의 값이 짝수 배이기에 shared_x + 1 의 위치도 수행하여, 하나의 thread에서 최대 2
번의 local memory read 동작을 수행하도록 한다.

shared_x를 짝수 배로 설정하였기에 위의 동작을 수행하지 않는 thread가 여유있게 존재
한다. 해당 thread들은 다시 gaussian filter에 따라 reindexing을 수행하여 25 개의 Thread
가 gaussian filter를 계산할 수 있도록 한다. gaussian filter를 계산할 때 CUDA의 fast math
func.인 __expf()을 사용하여 빠르게 float 연산을 할 수 있도록 한다.

위의 과정을 마치면 모든 thread를 __syncthreads() 함수를 통해 synchronization을 수행
한다. 이후의 동작은 실제 Pixel이 존재하는 구간에서만 수행하기 때문에 이외의 범위의
thread는 종료시킨다.

이후 동작은 Gaussian filter와 Shared memory 간의 Matrix Convolution이다. 반복문을 통
해 만약 현재 위치에 index가 실제로 존재하는 경우에 register의 temp에 gaussian filter
와 shared memory를 곱한 값을 누적시킨다. 누적시킨 값은 output의 RGB에 해당하는 픽
셀에 쓰여지며, 이후에 함수를 종료한다.

해당 동작을 보았을 때 각 Thread 별로 최대 2 번의 local memory read가 발생하고 3 번
의 local memory write를 수행하는 장점을 지닌 코드다.


해당 함수는 기존에 host에서 수행하는 Noise Reduction 부분이다. 이전 GPU_Grayscale
과 마찬가지로 device pointer를 생성한다. 하지만 해당 함수는 기존과 다른 점이 있다면
blockDim을 (32, 32) 2D로 설정하기 때문에 grid_width와 height 또한 TILE_WIDTH로 쪼갠
후 이를 올림하는 연산을 취해준다.

이후의 동작 또한 GPU_Grayscale과 유사하다. width * height * 3 크기로 dev_in과
dev_out을 할당해주며 gray의 값을 dev_in으로 복사(host -> device)한다. 그 후
Cuda_Noise_Reduction 커널을 수행하며 수행한 결과가 저장된 dev_out을 gaussian으로
복사(device -> host)한다. 그 후 사용한 dev_in과 dev_out의 메모리를 할당 해제하고 함수
를 마친다.

d. Intensity Gradient

기존 Gaussian blur를 적용한 image에서 gradient(sobel)와 angle을 찾는 연산을 수행한
다.


해당 함수는 __shared__를 통해 Tile에서 상하좌우로 1 씩 padding된 크기의 shared_mem
를 사용한다. padding된 크기로 shared memory를 사용하는 이유는 Convolution 동작에서
외각 부분을 local memory에서 가져올 경우 최대 5 번(모서리 부분)의 local memory
access를 해야 한다. 이는 매우 큰 성능 저하로 이어지기 때문에 padding된 크기로
shared memory를 사용하는 것이다.


shared_x, shared_y는 shared memory에서 해당 thread가 접근할 shared memory가 어디
인지를 나타내며 기존 32 * 32 index를 2 배로 늘린 후 이를 S_SHARED_WIDTH(34)로 나누
거나 몫을 구하여 x, y의 위치를 설정한다. 이렇게 될 경우 각 x의 값은 0 이상
S_SHARED_WIDTH 미만의 짝수 값으로 설정되며, Y의 값은 0 부터 증가하는 정수가 된다.

이후 y가 S_SHARED_WIDTH 미만인 경우에는 shared memory 행렬에 값을 넣는 동작을
수행한다. 현재 shared_x, y가 가리키는 실제 메모리 위치를 확인하고 해당 값이 실제 접
근 가능한 메모리 위치일 경우 해당 위치의 값을 가져와 shared memory에 삽입한다. 이
때 x의 값이 짝수 배이기에 shared_x + 1 의 위치도 수행하여, 하나의 thread에서 최대 2
번의 local memory read 동작을 수행하도록 한다.

위의 과정을 마치면 모든 thread를 __syncthreads() 함수를 통해 synchronization을 수행
한다. 이후의 동작은 실제 Pixel이 존재하는 구간에서만 수행하기 때문에 이외의 범위의
thread는 종료시킨다.

```
이후 동작은 Sobel filter와 Shared memory 간의 Matrix Convolution이다. 반복문을 통해
만약 현재 위치에 index가 실제로 존재하는 경우에 register의 gx, gy에 sobel filter와
shared memory를 곱한 값을 누적시킨다. 이 때 sobel filter가 저장된 위치는 constant
Memory며 곱하는 값으로 shared memory를 사용하기 때문에 메모리 접근하는데 걸리는
시간이 적다.
누적된 값을 통해 gradient를 구하기 위해 CUDA의 fast math 함수인 __fsqrt_rn()함수를
사용하여 gx, gy의 길이를 구한다. 해당 값이 255 를 넘어갈 경우 255 로 값을 정리해 v에
결과를 저장하고 이를 sobel의 해당 Pixel 위치에 삽입한다.
각도의 경우 arctan와 관련한 함수는 fast math 함수가 없기 때문에 일반 math 함수인
atan2f를 사용하여 각도를 구한다. 해당 각도의 범위를 총 4 방향(0, 4 5 9 0 13 5 )으로 분류
하여 값을 angle에 저장한다.
해당 코드를 보면 한 thread 당 최대 2 번의 local memory read와 4 번의 local memory
write 동작을 취하는 것을 볼 수 있어 이에 따른 성능이 향상된다.
```

해당 함수는 기존에 host에서 수행하는 Intensity_Gradient 부분이다. 이전
GPU_Noise_Reduction과 마찬가지로 device pointer를 생성한다. 하지만 해당 함수는 기존
과 다른 점이 있다면 angle을 저장할 곳이 필요하기 때문에 추가로 생성해준다.

이후 Dimension과 관련 부분은 기존 함수와 동일하게 설정되어 있으며, 새롭게 추가되
는 부분은 constant memory에 데이터를 삽입하기 위한 작업이 추가되었다. filter_x, filter_y
배열에 sobel filter의 값을 넣어둔 채 initialize를 수행한다. 그 후 filter의 값을
__constant__ sobel_filter_x, y에 넣기 위해 cudaMemcpyToSymbol() 함수를 사용한다.

이후의 동작 또한 GPU_Noiste과 유사하다. width * height * 3 크기로 dev_in과 dev_sobel
을 할당해주며, dev_angle의 경우 RGB값이 필요 없기에 width * height 크기로 할당한다.
gaussian의 값을 dev_in으로 복사(host -> device)한다. 그 후 Cuda_Intensity_Gradient 커
널을 수행하며 수행한 결과가 저장된 dev_sobel을 sobel으로, dev_angle을 angle로 복사
(device -> host)한다. 그 후 사용한 dev_in과 dev_sobel, dev_angle의 메모리를 할당 해제
하고 함수를 마친다.

e. Non-maximum Suppression


기존 Intensity Gradient의 결과로 나온 gradient와 angle을 바탕으로 주요 선 이외의 선은
없애고, minmax 값을 찾는다.


해당 함수는 __shared__를 통해 Tile 크기의 shared_mem를 사용한다. 기존과 다르게
padding된 memory를 사용하지 않는 이유는 이후 Suppression 동작에서 외각 부분이라
할 지라도 1 번의 추가 local memory을 접근하기 때문에 기존처럼 각 thread를 2 번 씩 읽
어 사용하는 것 보다. 한 번씩 읽고 필요할 경우 추가적으로 읽는 방향으로 접근한다. 이
는 또한 Memory 공간의 낭비를 최소화 할 수 있다.

현재 thread가 가리키는 위치를 idx와 idy에 저장한 후, 해당 값이 pixel 범위를 내인 경
우에만 shared memory에 값을 쓴다. synchronization을 위해 __syncthreads() 함수를 수행
한다.

Suppression 동작은 Pixel의 최외각 부분은 동작하지 않음으로 if문을 통해 이를 판별한
다. 해당 Pixel의 angle을 local memory에서 가져온 후 register p1, p2를 선언해 둔다. 그
후 angle에 따라 해당 방향에서 양 측의 pixel을 가져온다. 가져올 pixel의 위치가 Tile 밖
일 경우 sobel에서 직접 읽어오며 아닐 경우에는 shared memory에서 읽어온다.

그 후 현 위치의 pixel을 가져오며, 해당 위치의 pixel이 양 측의 pixel보다 굵은 pixel인
경우에 v값을 output RGB에 입력하며, 약한 경우에는 0 을 입력한다.

이후의 동작은 local min과 max를 찾는 과정이다. 현 thread 위치에 pixel이 없을 경우에
는 종료시키며 shared memory인 check[ 256 ] 배열을 생성한다. 해당 배열은 각 픽셀이 사
용되었는지 체크하기위한 배열로 색상 2 ^8 크기만큼 생성된 것이다. 그리고 각 thread
별로 현 pixel에 해당하는 색만 1 로 설정한다. 해당 부분에서 thread 간 동시 접근이 발생
한다 하더라도, 값을 1 로 On하는 작업만 수행하기에 결과가 달라지지 않는다.

이후 check의 값을 보장하기 위해 __synthreads()함수를 사용하며 첫 번째 thread와 두
번째 thread에서 해당 Tile 내의 local min, max를 각각 찾아서 이를 local memory
minmax의 pixel 밝기 위치를 On한다. 해당 작업 또한 단순히 값을 1 로 만드는 작업이기
때문에 다른 block 간의 동시 접근이 발생한다 하더라도 값이 바뀌지 않는다.

위의 작업은 최대 3 번의 local memory read가 발생하고 4 번의 local memory write가 발
생할 수 있다.


해당 함수는 기존에 host에서 수행하는 Non_maximum_Suppression 부분이다. 이전
GPU_Intensity_Gradient과 마찬가지로 device pointer를 생성한다. 기존과 다른 점이라면
local minmax값을 확인하기 위해 이를 저장할 dev_minmax와 minmax[ 256 ] 배열이 추가
되어 있다. Dimension과 관련 부분 또한 기존 함수와 동일하게 설정되어 있다.

이후의 동작 또한 GPU_Intensity_Gradient과 유사하다. width * height * 3 크기로
dev_sobel과 dev_out을 할당해주며, dev_angle의 경우 RGB값이 필요 없기에 width *
height 크기로 할당한다. dev_minmax 값은 밝기 범위인 256 으로 설정하였다.

sobel과 angle의 값을 dev_sobel과 angle로 복사(host -> device)한다. 그 후
Cuda_Non_maximum_Suppression 커널을 수행하며 수행한 결과가 저장된 dev_out과
dev_minmax 각각 suppression_pixel과 minmax로 복사(device -> host)한다. 그 후 각
pixel 색상을 확인하여 가장 크고 작은 pixel을 max, min값으로 설정한다. 그 후 사용한
dev_out과 dev_minmax, dev_sobel, dev_angle의 메모리를 할당 해제하고 함수를 마친다.

f. Hysteresis Thresholding

2 Thresholding을 통해 값을 3 개로 분류하며 해당 선분이 약한 선분일 때, 주변에 강한
선분이 없을 경우 해당 선분을 끄고, 있을 경우 강한 선분으로 만드는 Hyteresis 작업을


#### 수행한다.

해당 함수는 __shared__를 통해 Tile에서 상하좌우로 1 씩 padding된 크기의 shared_mem
를 사용한다. padding된 크기로 shared memory를 사용하는 이유는 Hysteresis 동작에서
외각 부분을 local memory에서 가져올 경우 최대 5 번(모서리 부분)의 local memory
access를 해야 한다. 이는 매우 큰 성능 저하로 이어지기 때문에 padding된 크기로
shared memory를 사용하는 것이다.

shared_x, shared_y는 shared memory에서 해당 thread가 접근할 shared memory가 어디
인지를 나타내며 기존 32 * 32 index를 2 배로 늘린 후 이를 S_SHARED_WIDTH(34)로 나누


거나 몫을 구하여 x, y의 위치를 설정한다. 이렇게 될 경우 각 x의 값은 0 이상
S_SHARED_WIDTH 미만의 짝수 값으로 설정되며, Y의 값은 0 부터 증가하는 정수가 된다.

shared memory에 값을 삽입하는 과정에서 동시에 thresholding을 수행하기 위해 min
max 사이의 차이를 이용하여 low_t와 high_t 구간을 생성한다.

이후 y가 S_SHARED_WIDTH 미만인 경우에는 shared memory 행렬에 값을 넣는 동작을
수행한다. 현재 shared_x, y가 가리키는 실제 메모리 위치를 확인하고 해당 값이 실제 접
근 가능한 메모리 위치일 경우 해당 위치의 값을 가져와 shared memory에 삽입한다. 삽
입 과정에서 동시에 thresholding를 수행하기에 low_t보다 작을 경우 0, high_t보다 작을
경우 123, 이보다 클 경우 255 로 설정한다. shared_x의 값이 짝수 배이기에 shared_x + 1
의 위치도 수행하여, 하나의 thread에서 최대 2 번의 local memory read 동작을 수행하도
록 한다.

위의 과정을 마치면 모든 thread를 __syncthreads() 함수를 통해 synchronization을 수행
한다. 이후의 동작은 실제 Pixel이 존재하는 구간에서만 수행하기 때문에 이외의 범위의
thread는 종료시킨다.

이후 Hysteresis 동작을 수행한다. 해당 동작은 현 pixel인 temp가 123 일 때 주변 인접
pixel과 값을 비교한다. 이 때 확인하는 pixel이 실제 가리키는 위치가 존재하는 pixel인지
확인하는 과정이 추가된다. 만약 주변에 강한 pixel이 존재하는 경우 현 temp 또한 255 로
변경한다.

output RGB에 삽입하는 과정에서는 temp가 123 인 경우에는 0 으로 처리하여 값을 입력
한다.

위의 과정을 통해 최대 2 번의 local memory read와 3 번의 local memory write가 발생한
다.


```
해당 함수는 기존에 host에서 수행하는 Hysteresis Thresholding 부분이다. 이전
GPU_Noise_Reduction과 마찬가지로 device pointer를 생성한다. 또한 기존과 동일하게
Dimension을 설정한다.
이후의 동작 또한 GPU_Noise_Reduction과 유사하다. width * height * 3 크기로 dev_in과
dev_out을 할당해주며 suppression_pixel의 값을 dev_in으로 복사(host -> device)한다. 그
후 Cuda_Hysteresis_Thresholding 커널을 수행하며 수행한 결과가 저장된 dev_out을
hysteresis으로 복사(device -> host)한다. 그 후 사용한 dev_in과 dev_out의 메모리를 할당
해제하고 함수를 마친다.
```
## 4. Result

```
해당 파일을 compile 하였을 때 다음과 같이 기본적으로 Canny.cu에서 발생한 warning
만 발생하기 때문에 정상적으로 컴파일이 되었음을 확인할 수 있다.
```

해당 a.out을 실행하였을 때에는 다음과 같은 결과를 볼 수 있다.

Gray Scale의 경우 기존 0.006 596 초에서 0.001 827 초로 약 3.6배의 성능이 향상되었다.
Noise Reduction의 경우 0.0 88040 초에서 0.00206 5 초로 무려 약 42.6배의 성능이 차이가
난다. Intensity Gradient의 경우도 0.0 95820 초에서 0.002 952 초로 약 32.4배의 높은 성능
차이가 보인다. Non-Maximum Suppression 연산의 경우에는 0.01 8858 초에서 0.0024 87 초
로 이전보다는 낮지만 그래도 여전히 높은 약 7. 5 배의 성능 차이를 보였다. 마지막으로
Hysteresis Thresholding 연산은 0.031 231 초에서 0.001 991 초로 약 15. 7 배의 성능이 향상
되었다. 이에 총 수행시간은 0.011322초로 기존의 Gray_Scale ~ Noise Reduction까지의
시간보다 빠르게 전체 연산이 수행되었다.

이러한 결과가 나온 이유는 기본적으로 GPU가 병렬 연산에 특화되어 있기 때문이다. 기
본적으로 image 처리는 각 pixel 별로 수행해야 하기 때문에 많은 양의 병렬 연산이 필
요하다. 특히 각 pixel 별로 Matrix Convolution 연산의 경우에는 이러한 속도가 더 차이
나게 된다. 이는 CPU에서 Matrix Convolution을 쓴 결과인 Noise Reduction과 Intensity
Gradient 연산의 속도가 위와 같이 큰 이유이기도 하다. CPU에서는 모든 부분을 반복문
으로 돌기 때문에 메모리에 접근하는 시간도 많아질 뿐더러 반복문 자체도 Convolution
의 경우 4 중이기 때문이다. 이에 비해 GPU는 Matrix Convolution 연산만 반복문을 사용
해 2 중만 사용하여도 구할 수 있다. 추가적으로 모든 과정에서 Memory를 최소한으로 접
근하도록 하였기에 더욱 빠르게 수행된 것으로 예상된다.


## 5. Consideration

```
해당 과제를 통해 우선 GPU의 shared memory를 최대한 활용하는 방법을 고민하는 시간
을 가졌다. 특히 Padding 과정에서 shared memory를 padding된 크기로 가져오는 방법
을 고민하는 데에 많은 시간을 소비하였다. 다음으로 많은 생각을 하게 된 것을 min,
max값을 찾는 과정이다. 반복문의 경우 전체를 돌며 손쉽게 찾을 수 있었지만, 쪼개진
Tile에서 이러한 minmax값을 찾는 것은 쉽지 않았다. 이에 Tile 간의 local min, max를 중
다시 min, max를 찾아보는 연산도 구현해보았지만 비효율적으로 보였고, pixel의 색 범위
가 256 이라는 것에 착안하여, 위와 같은 방법을 통해 새로운 접근을 할 수 있었다. 또한
해당 과제에서 __constant__ memory를 삽입할 때 cudaMemcpyToSymbol() 함수를 사용
하면 된다는 것을 과제를 하며 공부할 수 있었다.
```
## 6. Reference

#### 강의자료만을 참고



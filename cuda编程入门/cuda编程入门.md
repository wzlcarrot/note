# 第一章

## 一、GPU硬件

### 1、什么是GPU

​		GPU 意为图形处理器，也常被称为显卡，GPU最早主要是进行图形处理的。GPU拥有更多的运算核心，其特别适合数据并行的计算密集型任务，如大型矩阵运算，与GPU对应的一个概念是CPU，但CPU的运算核心较少，但是其可以实现复杂的逻辑运算。

![image-20240807003911944](cuda编程入门.assets/image-20240807003911944.png)



### 2、GPU性能

主要GPU性能指标：

1. 核心数量：为GPU提供计算能力的硬件单元，核心数量越多，可并行运算的线程越多，计算的峰值越高；
2. GPU显存容量：显存容量决定着显存临时存储数据的多少，大显存能减少读取数据的次数，降低延迟，可类比CPU的内存；
3. GPU计算峰值：每块显卡都会给出显卡的GPU计算峰值，这是一个理论值，代表GPU的最大计算能力，一般实际运行是达不到这个数值的；
4. 显存带宽：GPU包含运算单元和显存，显存带宽就是运算单元和显存之间的通信速率，显存带宽越大，代表数据交换的速度越快，性能越高。



## 二、CPU+GPU架构

GPU不能单独进行工作，GPU相当于CPU的协处理器（意思就是cpu是一把手，gpu是二把手）。CPU比较擅长逻辑运算，但是不擅长数据运算。GPU比较擅长数据运算，但是不擅长逻辑运算。

![image-20240807004820283](cuda编程入门.assets/image-20240807004820283.png)

由这个图可以看出，一个GPU具有较多的算术逻辑单元，CPU具有较少的算术逻辑单元。所以GPU可以进行大量的数据运算。

CPU和GPU都有自己的DRAM（dynamic random-access memory，动态随机存取内存），它们之间一般由PCIe总线（peripheral component interconnect express bus）连接。

通常将起控制作用的 CPU 称为主机（host），将起加速作用的 GPU 称为设备（device）。**所以在今后，说到主机就是指CPU，说到设备就是指GPU。**



##　三、CUDA介绍

### 1、什么是CUDA

CUDA是建立在NVIDIA的GPU上的一个通用并行计算平台和编程模型。换句话来说，就是运行在GPU上的应用程序。



### 2、CUDA编程语言

官方语言采用了c++编程语言。



### 3、CUDA运行时API

CUDA提供两层API接口，CUDA驱动(driver)API和CUDA运行时(runtime)API。

![image-20240807010451838](cuda编程入门.assets/image-20240807010451838.png)



## 四、第一个CUDA程序

### 1、第一个CUDA程序

```c++
#include <stdio.h>

//CPU调用核函数，核函数在GPU上执行。
__global__ void hello(){
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello world from block %d, thread %d ,global id %d\n", blockIdx.x, threadIdx.x,Idx);
}

int main(){
  hello<<<2, 4>>>();   //1表示的是线程块，5表示的是线程块里里的线程数
  cudaDeviceSynchronize(); //起到一个同步的作用，确保GPU上的计算完成，然后再返回到CPU上。
  return 0;
}
```

### 2、编译和运行

```
nvcc helloworld.cu -o test   //把helloworld.cu编译成test.exe
./test //运行test.exe
```

### 3、运行结果

![image-20240807011143741](cuda编程入门.assets/image-20240807011143741.png)



## 五、nvidia-smi命令

### 1、命令

```
nvidia-smi //通过该命令可以查看gpu版本和cuda版本和正在GPU上运行的CUDA编程。
```

![image-20240807011553541](cuda编程入门.assets/image-20240807011553541.png)





# 第二章

## 一、c++中的helloworld

编译：

```
g++ hello.cpp -o hello
```

运行：

```
./hello.exe
```



## 二、cuda中的helloworld

编译：

```
nvcc hello.cu -o hello
```

运行：

```
./hello.exe
```



## 三、核函数

1. 核函数在GPU上进行并行执行
2. 限定词__global__ 修饰
3. 返回值必须是void

## 四、CUDA程序举例

![image-20240807222235783](cuda编程入门.assets/image-20240807222235783.png)

## 五、cuda线程模型结构

![image-20240807223111988](cuda编程入门.assets/image-20240807223111988.png)

![image-20240807223208880](cuda编程入门.assets/image-20240807223208880.png)

![image-20240807223250161](cuda编程入门.assets/image-20240807223250161.png)



## 六、线程全局索引计算

### 1、全局索引计算

![image-20240807224303130](cuda编程入门.assets/image-20240807224303130.png)

![image-20240807224314057](cuda编程入门.assets/image-20240807224314057.png)

![image-20240807224327418](cuda编程入门.assets/image-20240807224327418.png)



### 2、不同组合方式

![image-20240807224456453](cuda编程入门.assets/image-20240807224456453.png)

![image-20240807224515599](cuda编程入门.assets/image-20240807224515599.png)

![image-20240807224525696](cuda编程入门.assets/image-20240807224525696.png)

## 七、nvcc编译流程

nvcc先将设备代码编译为PTX（Parallel Thread Execution）伪汇编代码，再将PTX代码编译为 

二进制的cubin目标代码。

![image-20240807225419285](cuda编程入门.assets/image-20240807225419285.png)



## 八、GPU架构与计算能力

![image-20240807225615302](cuda编程入门.assets/image-20240807225615302.png)

**补充：并非GPU 的计算能力越高，性能就越高。**



## 九、指定虚拟架构计算能力

![image-20240807230002246](cuda编程入门.assets/image-20240807230002246.png)

**命令行**

```
nvcc helloworld.cu -o helloworld -arch=compute_61
```



## 十、指定真实架构计算能力

![image-20240807230135186](cuda编程入门.assets/image-20240807230135186.png)

**命令行**

```
nvcc helloworld.cu -o helloworld -arch=compute_61 -code=sm_62
```



## 十一、指定多个GPU版本编译

![image-20240807230336999](cuda编程入门.assets/image-20240807230336999.png)

**命令行**

```
nvcc helloworld.cu -o helloworld -gencode=arch=compute_50,code=sm_50
```



## 十二、nvcc即时编译

补充：NVCC（NVIDIA CUDA 编译器）的即时编译（Just-In-Time Compilation，JIT）是一种编译策略，它允许CUDA程序在运行时进行编译。这意味着，当你运行一个CUDA程序时，NVCC会即时将源代码编译成GPU可以执行的机器码。



## 十三、nvcc编译默认计算能力

![image-20240807231535992](cuda编程入门.assets/image-20240807231535992.png)

补充：默认计算能力也就是真实架构的计算能力。

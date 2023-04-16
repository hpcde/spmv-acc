# NVIDIA 平台的测试流程
机器：NVIDIA V100

## 准备测试代码（本地）
```bash
git clone -b benchmark-in-v100 https://git.hpcer.dev/PRA/spmv/spmv-acc.git 
cd spmv-acc
git pull origin benchmark-in-v100
git lfs pull
```

## 下载依赖（本地）
```bash
#1. 切换到 spmv-acc 代码目录
cd ${path-of-spmv-acc}
#2. 下载 hola 代码
cd third-party && ./get_hola.sh && cd ..
#3. 下载 cub 代码
wget --output-document cub-1.12.1.tar.gz \
https://codeload.github.com/NVIDIA/cub/tar.gz/refs/tags/1.12.1
tar -zxvf cub-1.12.1.tar.gz
```

## 修改 hola 代码
由于 hola 使用的 warp 原语操作已经废弃，无法在高版本的 cuda 环境中使用，需要将其进行替换。
修改 `${path-of-spmv-acc}/third-party/holaspmv/include/common.cuh` 文件第 9 行插入如下代码：
```cpp
#define __shfl_down(X, Y) __shfl_down_sync(0xFFFFFFFF, X, Y)
#define __shfl_up(X, Y) __shfl_up_sync(0xFFFFFFFF, X, Y)
#define __shfl(X, Y) __shfl_sync(0xFFFFFFFF, X, Y)
#define __ballot(X) __ballot_sync(0xFFFFFFFF, X)
```

## 打包数据并传输到远程（本地/远程）
将本地代码以及依赖打包传输到远程服务器。
```bash
# 1. 本地打包
cd ${path-of-spmv-acc}/..
tar -czf spmv-acc.tar.gz ${path-of-spmv-acc}
# 2. 远程解压
tar -zxvf spmv-acc.tar.gz
```

## 配置环境变量（远程）
将 cub 目录加入到 CPLUS_INCLUDE_PATH 环境变量中
```bash
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${path-of-cub}/cub-1.12.1/
```

## 编译（远程）
```bash
cd ${path-of-spmv-acc} \
&& source build-for-benchmark-v100.sh \
&& cmake --build ./cmake-build-release -j8
```

## 测试（远程）
```bash
cd ${path-of-spmv-acc}
srun -n 1 --gres=gpu:1 -p lzhgnormal ./cmake-build-release/bin/spmv-gpu-benchmark ./example/data/af23560.csr -f csr
```

## 生成提交脚本（远程）
```bash
cd ${path-of-spmv-acc}
ln -s /public/share/genshen/suitesparse-matrices/ suitesparse-matrices
# 解压提前生成的脚本
tar -zxvf suitesparse-scripts.tar.gz
```

## 提交脚本（远程）
```bash
sbatch --exclusive ${path-of-spmv-acc}/suitesparse-scripts/spmv_batch_10k.sh
... 100k ...
... 1M   ... 
... 10M  ...
... 100M ...
sbatch --exclusive ${path-of-spmv-acc}/suitesparse-scripts/spmv_batch_1G.sh
```

## 数据后处理（本地/远程）
```bash
cat xxx.log  | grep PERFORMANCE >> nv-v100.all.csv
```





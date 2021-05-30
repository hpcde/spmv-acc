# About the Demo code

## Origin README.md in file demo.zip
测试用例说明

加载环境 rocm-3.9.1
```bash
module switch compiler/rocm/2.9 compiler/rocm/3.9.1
```
### 编译方式
分为主机端编译和异构端编译
### 异构端编译
异构端验证采用rocsparse库和自定义函数进行结果比对，编译方式如下  
hipcc -Dgpu -I/public/software/compiler/rocm/rocm-3.9.1/rocsparse/include -I ./  -L/public/software/compiler/rocm/rocm-3.9.1/rocsparse/lib/ -lrocsparse main.cpp -o Csrsparse

### 主机端编译
主机端验证调用下面的函数
```c++
spmv(alpha,beta,value,rowptr,colindex,m,n,a,hX,hhY);
```  
与参赛者实现的接口函数进行结果比对   
如果采用CPU端验证使用下面编译方式即可,通常情况下采用cpu端验证即可  
hipcc main.cpp -I ./ -o Csrsparse

### 运行方式
```shell
./Csrsparse ./af23560.csr
./Csrsparse ./bayer10.csr
./Csrsparse ./bcsstk18.csr
./Csrsparse ./coater2.csr
./Csrsparse ./dw4096.csr
./Csrsparse ./epb1.csr
./Csrsparse ./exdata_1.csr
./Csrsparse ./nemeth03.csr
./Csrsparse ./poli_large.csr
./Csrsparse ./rajat03.csr
```

## Some comments to demo code from authors of this project (spmv-acc)
## 已知问题
1. 针对 demo 代码和本仓库代码，rocm 版本需要 3.x，rocm 2.x 似乎编译无法通过。  
   如果使用 rocm 2.x + hipcc 2.x，编译会有如下错误：
   - 不识别`<<<`;
   - `__float128 is not supported on this target`;

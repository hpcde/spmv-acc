# About the Demo code

## Origin README.md in file demo.zip
### 编译方式

异构端验证
```bash
hipcc -Dgpu -I/public/software/compiler/rocm/rocm-3.9.1/rocsparse/include -I ./  -L/public/software/compiler/rocm/rocm-3.9.1/rocsparse/lib/ -lrocsparse main.cpp -o Csrsparse
```
Cpu端验证  
如果采用CPU端验证使用下面编译方式即可,通常情况下采用cpu端验证即可
```bash
hipcc main.cpp -I ./ -o Csrsparse   
```

### 运行方式:
```bash
./Csrsparse 3000 3000 0.5
```
第一个和第二个参数是矩阵维度(m行n列)，第三个参数是稠密度(稀疏度=1-稠密度)


## Some comments to demo code from authors of this project (spmv-acc)
## 已知问题
1. 针对 demo 代码和本仓库代码，rocm 版本需要 3.x，rocm 2.x 似乎编译无法通过。  
   如果使用 rocm 2.x + hipcc 2.x，编译会有如下错误：
   - 不识别`<<<`;
   - `__float128 is not supported on this target`;

# spmv-acc

HIP acceleration for SpMV solver.

## Demo code
编译方式

异构端验证
```bash
hipcc -Dgpu -I/public/software/compiler/rocm/rocm-3.9.1/rocsparse/include -I ./  -L/public/software/compiler/rocm/rocm-3.9.1/rocsparse/lib/ -lrocsparse main.cpp -o Csrsparse
```
Cpu端验证  
如果采用CPU端验证使用下面编译方式即可,通常情况下采用cpu端验证即可
```bash
hipcc main.cpp -I ./ -o Csrsparse   
```

运行方式:
```bash
./Csrsparse 3000 3000 0.5
```
第一个和第二个参数是矩阵维度(m行n列)，第三个参数是稠密度(稀疏度=1-稠密度)

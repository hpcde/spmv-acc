//设置计算类型为双精度

using namespace std;
typedef double  dtype;

#ifdef gpu
#include <hipsparse.h>         // cusparseSpMV
#include <rocsparse.h>
#endif


//定义矩阵和稠密向量指针
dtype **mat=NULL; //稀疏矩阵
dtype *hX=NULL;   //密度向量
dtype *hhY=NULL;   //主机结果
dtype *hY=NULL;    //设备最终结果
dtype *temphY=NULL;    //Y向量的初始值，hy和hyy拷贝于此
dtype *value=NULL; //csr存储格式稀疏值
int *colindex=NULL; //csr列标识
int *rowptr=NULL;   //csr行标识
int a =0; //稀疏矩阵非零值个数
int n=0; //矩阵维度
int m=0; //矩阵维度
float s=0.5; //稠密度
int A_nnz=0; //同a
int A_num_cols=0; //同m
int A_num_rows=0; //同n
int   *dA_csrOffsets, *dA_columns; //设备
double *dA_values, *dX, *dY;
double     alpha           = 1; //稀疏矩阵的标量系数
double     beta            = 1; //标量系数

int rand_integer(int min,int max)
{
    return int ((rand() % (max-min+1))+ min);
}

double rand_double(double min,double max)
{
    double temp= min + (max - min)* double(rand()%100) / double((101));
    //cout<<temp<<endl;
    return temp ;
}

int matrix_to_csr(int m,int n,dtype **M,dtype* &value,int * & rowptr,int * & colindex){
   int i,j;
   int a=0;
   for(i=0;i<m;i++)
      for(j=0;j<n;j++)
          if(M[i][j]!=0)
              a++;
   value=new dtype[a];
   colindex=new int[a];
   rowptr=new int[n+1];
   int k=0;
   int l=0;
   for(i=0;i<m;i++)
      for(j=0;j<n;j++){
          if(j==0)
              rowptr[l++]=k;
          if(M[i][j]!=0){
              value[k]=M[i][j];
              colindex[k]=j;
              k++;}
   }
   rowptr[l]=a;
   return a;
}

void spmv(int alpha,int beta,dtype *value,int *rowptr,int *colindex,int m,int n,int a,dtype *x,dtype *y){
    //calculate the matrix-vector multiply where matrix is stored in the form of CSR
    for(int i=0;i<m;i++){
        dtype y0=0;
        for(int j=rowptr[i];j<rowptr[i+1];j++)
            y0+=value[j]*x[colindex[j]];
        //printf("%d,%d,%f,%f\n",alpha,beta,y0,y[i]);
        y[i]=alpha*y0+beta*y[i];
    }
    return;
}

void generate_sparse_matrix(dtype** & ptr,int m,int n,double s){
   ptr=new dtype*[m];
   for(int i=0;i<m;i++)
       ptr[i]=new dtype[n];
   for(int i=0;i<m;i++)
      for(int j=0;j<n;j++)
      {
          double x=rand_double(-1,1);
          if(x>10*s)
            ptr[i][j]=0;
          else
            ptr[i][j]=x;
      }
   return;
}

void print_matrix(dtype **ptr,int m,int n){
   for(int i=0;i<m;i++)
       for(int j=0;j<n;j++)
   {
           cout<<ptr[i][j]<<",";
           if(j==n-1)
               cout<<endl;
   }
   return;
}

void generate_vector(int n,dtype* & x){
    x=new dtype[n];
    for(int i=0;i<n;i++)
        x[i]=rand_double(-1,1);
    return;
}

void print_vector(int n,dtype* x){
    for(int i=0;i<n;i++)
        cout<<x[i]<<" ";
	cout<<endl;
    return;
}

void print_vector(int n,int* x){
    for(int i=0;i<n;i++)
        cout<<x[i]<<" ";
	cout<<endl;
    return;
}

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            exit(-1);                                                          \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != rocsparse_status_success)                                         \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            exit(-1);                                                                \
        }                                                                            \
    }




void create_host_data()
{
    generate_sparse_matrix(mat,m,n,s);
    generate_vector(n,hX);
    generate_vector(m,temphY);
    generate_vector(m,hY);
    generate_vector(m,hhY);
    a=matrix_to_csr(m,n,mat,value,rowptr,colindex); //非零元素的个数
    //统一主机端和设备端的向量值
    memcpy(hY,temphY,m*sizeof(double));
    memcpy(hhY,temphY,m*sizeof(double));
    //debug
    // print_matrix(mat,m,n);
    // print_vector(n,hX);
}

void create_deivce_data()
{
    A_nnz=a;
    A_num_cols=n;
    A_num_rows=m;

	// printf("a is %d value rowptr colindex hX hy as flows\n",a);
	// print_vector(a,value);
	// print_vector(n+1,rowptr);
	// print_vector(a,colindex);
    //  print_vector(n,temphY);
	//  print_vector(n,hY);
    ///////////////////////////////////////////


    HIP_CHECK( hipMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    HIP_CHECK( hipMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    HIP_CHECK( hipMalloc((void**) &dA_values,  A_nnz * sizeof(double))      )
    HIP_CHECK( hipMalloc((void**) &dX,         A_num_cols * sizeof(double)) )
    HIP_CHECK( hipMalloc((void**) &dY,         A_num_rows * sizeof(double)) )

    HIP_CHECK( hipMemcpy(dA_csrOffsets, rowptr,
                           (A_num_rows + 1) * sizeof(int),
                           hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(dA_columns, colindex, A_nnz * sizeof(int),
                           hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(dA_values, value, A_nnz * sizeof(double),
                           hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(dX, hX, A_num_cols * sizeof(double),
                           hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(dY, temphY, A_num_rows * sizeof(double),
                           hipMemcpyHostToDevice) )
}


void verify(double* dy,double* hy,int n)
{
    for(int i =0; i<n; i++) {
        if (fabs(dy[i]-hy[i])>=1e-10)
        {
            cout<<"i:"<<i<<" dy[i]:"<<dy[i]<<" hy[i]:"<<hy[i]<<endl;
            cout<<"Failed verification,please check your code\n"<<endl;
            return ;
        }
        cout<<"i:"<<i<<" dy[i]:"<<dy[i]<<" hy[i]:"<<hy[i]<<endl;
    }
    cout<<"Congratulation,pass validation!\n"<<endl;
}




enum sparse_operation operation= operation_none;

#ifdef gpu
void rocsparse()
{
rocsparse_handle     handle = NULL;
rocsparse_spmat_descr matA;
rocsparse_dnvec_descr vecX, vecY;
void*                dBuffer    = NULL;
size_t               bufferSize = 0;
ROCSPARSE_CHECK( rocsparse_create_handle(&handle) );
    // Create sparse matrix A in CSR format
ROCSPARSE_CHECK( rocsparse_create_csr_descr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      rocsparse_indextype_i32, rocsparse_indextype_i32,
                                      rocsparse_index_base_zero, rocsparse_datatype_f64_r) );
ROCSPARSE_CHECK( rocsparse_create_dnvec_descr(&vecX, A_num_cols, dX, rocsparse_datatype_f64_r) );
ROCSPARSE_CHECK( rocsparse_create_dnvec_descr(&vecY, A_num_rows, dY, rocsparse_datatype_f64_r) );
    // allocate an external buffer if needed
ROCSPARSE_CHECK( rocsparse_spmv(
                                 handle, rocsparse_operation_none,
                                 &alpha, matA, vecX, &beta, vecY, rocsparse_datatype_f64_r,
                                 rocsparse_spmv_alg_default, &bufferSize,nullptr) );
HIP_CHECK( hipMalloc(&dBuffer, bufferSize) );
ROCSPARSE_CHECK( rocsparse_spmv(handle, rocsparse_operation_none,
                                 &alpha, matA, vecX, &beta, vecY, rocsparse_datatype_f64_r,
                                 rocsparse_spmv_alg_default, &bufferSize,dBuffer) );

}
#endif 
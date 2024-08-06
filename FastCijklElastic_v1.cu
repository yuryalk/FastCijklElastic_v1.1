// This routine is written on CUDA C for a single GPU.
// 02 Dec 2023
// Copyright (C) 2023  Yury Alkhimenkov
// Massachusetts Institute of Technology

// Output: See matlab
// Input:  Parameters are generated in the Matlab script
// To use: You need a GPU with 8.0 GB of GPU DRAM; Better to use Cuda/12.0 or above
// To run and visualize: Use the Matlab script 
// 
// to compile:  see matlab
// to run:      a.exe
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define GPU_ID 0
// #define USE_SINGLE_PRECISION      /* Comment this line using "!" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define MPI_DAT MPI_REAL
#define PRECIS  4
#else
#define DAT     double
#define MPI_DAT MPI_DOUBLE_PRECISION
#define PRECIS  8
#endif
////////// ========== Simulation Initialisation ========== //////////
#define BLOCK_X    32  
#define BLOCK_Y    2  
#define BLOCK_Z    8  
#define GRID_X      (NBX*1*1) 
#define GRID_Y      (NBY*4*4) 
#define GRID_Z      (NBZ*2*2) 

// maximum overlap in x, y, z direction. x : Vx is nx+1, so it is 1; y: Vy is ny+1, so it is 1; z: Vz is nz+1, so it is 1.
#define MAX_OVERLENGTH_X OVERX //3
#define MAX_OVERLENGTH_Y OVERY //3
#define MAX_OVERLENGTH_Z OVERZ //3

const DAT rad0 = 1.0;
const DAT Lx = 1.0;
const DAT Ly = 1.0;
const DAT Lz = 1.0;
DAT G0 = 1.0;

// nondimentional parameters
const DAT fric = 1.0 * 1.0 / 2.0;     
DAT Str0_1 = 0.0;        
DAT Str0_2 = 0.0;
DAT Str0_3 = 0.0;
DAT dStr0_1 = 0.1 * 1e-4;         
DAT dStr0_2 = -(1e-4) * 0.1; 
DAT dStr0_3 = 0.0;
DAT divV0 = 0.0;

DAT alpha1 = 0.6;
DAT M1 = 0.3;
DAT eta_k1 = 1e10;
DAT chi = 0.5;

DAT dStr_1 = 0.8 * 5e-4;         
DAT dStr_2 = -(5e-4) * 0.8;
DAT dStr_3 = 0.0;
DAT scale = 0;
const DAT coh0 = 1e-2 * G0;                
DAT alphM   = 0.5;
DAT M       = 1.0;
// Numerics
const int nx     = GRID_X*BLOCK_X - MAX_OVERLENGTH_X;        // we want to have some threads available for all cells of any array, also the ones that are bigger than nx.
const int ny     = GRID_Y*BLOCK_Y - MAX_OVERLENGTH_Y;        // we want to have some threads available for all cells of any array, also the ones that are bigger than ny.
const int nz     = GRID_Z*BLOCK_Z - MAX_OVERLENGTH_Z;        // we want to have some threads available for all cells of any array, also the ones that are bigger than nz.
/// Params to be saved for evol plot ///
#define NB_PARAMS       3
#define iter_EVOL       0
//#define Vx_MAX          1
//#define Vy_MAX          2
#define Vydif_MAX       1
#define Vy_MAX          2
////// global variables //////   
#define PI 3.141592653589793
const int niter = (nx * 12);
int iter = 0;
DAT epsi = 9.0*1e-02;
const int nout = 50;
const int nsave = 1;
const DAT CFL = 0.8499 / sqrt(3.0);// (0.57);//1.0/2.0;                         // V Courant Friedrich Levy codition
const DAT damp = 3.37;//5.0/1.0;                         // damping of acoustic waves
const DAT Dnx = BLOCK_X * GRID_X - MAX_OVERLENGTH_X;
const DAT Dny = BLOCK_Y * GRID_Y - MAX_OVERLENGTH_Y;
// Preprocessing
const DAT dx = Lx / ((DAT)nx - (DAT)1.0);
const DAT dy = Ly / ((DAT)ny - (DAT)1.0);
const DAT dz = Lz / ((DAT)nz - (DAT)1.0);
const DAT rho11 = 1 / ((1.0 * 1.0) / (CFL * CFL) / (dx * dx) / (DAT)1.0 * (G0 + (DAT)4.0 / (DAT)3.0 * G0)); // inertial density
DAT rho12 = (DAT)0.15 * rho11;
DAT rho22 = (DAT)0.3 * rho11;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "time.h"
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Definition of basic macros
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NB_THREADS             (BLOCK_X*BLOCK_Y*BLOCK_Z)
#define NB_BLOCKS              (GRID_X*GRID_Y*GRID_Z)
#define def_sizes(A,nx,ny,nz)  const int sizes_##A[] = {nx,ny,nz};                            
#define size(A,dim)            (sizes_##A[dim-1])
#define numel(A)               (size(A,1)*size(A,2)*size(A,3))
#define end(A,dim)             (size(A,dim)-1)
#define zeros_h(A,nx,ny,nz)    def_sizes(A,nx,ny,nz);                                  \
                               DAT *A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)0.0; }
#define zeros(A,nx,ny,nz)      def_sizes(A,nx,ny,nz);                                         \
                               DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)0.0; }        \
                               cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                               cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define ones(A,nx,ny,nz)       def_sizes(A,nx,ny,nz);                                         \
                               DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)1.0; }        \
                               cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                               cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define gather(A)              cudaMemcpy( A##_h,A##_d,numel(A)*sizeof(DAT),cudaMemcpyDeviceToHost);
#define free_all(A)            free(A##_h);cudaFree(A##_d);
#define swap(A,B,tmp)          DAT *tmp; tmp = A##_d; A##_d = B##_d; B##_d = tmp;
#define load3(A,nx,ny,nz,Aname)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*(nz)*sizeof(DAT));  \
                             FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(DAT), (nx)*(ny)*(nz), A##fid); fclose(A##fid); \
                             cudaMalloc(&A##_d,((nx)*(ny)*(nz))*sizeof(DAT)); \
                             cudaMemcpy(A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice); 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables for cuda
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int nprocs=1, me=0;
dim3 grid, block;
int gpu_id=-1;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions (host code)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_up_gpu(){
    block.x = BLOCK_X; block.y = BLOCK_Y; block.z = BLOCK_Z;
    grid.x  = GRID_X;   grid.y  = GRID_Y;   grid.z  = GRID_Z;
    gpu_id  = GPU_ID;
    cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset();                                // Reset the device to avoid problems caused by a crash in a previous run (does still not assure proper working in any case after a crash!).
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
}

void clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}

////////////////////
#define blockId        (blockIdx.x  +  blockIdx.y *gridDim.x +  blockIdx.z*gridDim.x*gridDim.y)
#define threadId       (threadIdx.x +  threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y)
#define isBlockMaster  (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
// maxval //
#define select(A,ix,iy,iz) ( A[ix + iy*size(A,1) + iz*size(A,1)*size(A,2)] )
#define participate_a(A)  (ix< size(A,1)         && iy< size(A,2)         && iz< size(A,3)        )
#define block_max_init() DAT __thread_maxval;
#define __thread_max(A)   __thread_maxval=0;                                                                                \
                            if (participate_a(A)){ __thread_maxval = max(abs(__thread_maxval),abs(select(A,ix ,iy ,iz))); } 


__shared__ volatile DAT __block_maxval;
#define __block_max(A)    __thread_max(A);                                                            \
                          if (isBlockMaster){ __block_maxval=0; }                                     \
                          __syncthreads();                                                            \
                          for (int i=0; i < (NB_THREADS); i++){                                       \
                            if (i==threadId){ __block_maxval = max(__block_maxval,__thread_maxval); } \
                            __syncthreads();                                                          \
                          }

__global__ void __device_max_d(DAT* A, const int nx_A, const int ny_A, const int nz_A, DAT* __device_maxval) {
    // CUDA specific
    def_sizes(A, nx_A, ny_A, nz_A);
    block_max_init();
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
    // find the maxval for each block
    __block_max(A);
    __device_maxval[blockId] = __block_maxval;
}

#define __MPI_max(A)  __device_max_d<<<grid, block>>>(A##_d, size(A,1),size(A,2),size(A,3), __device_maxval_d); \
                      gather(__device_maxval); device_MAX=(DAT)0.0;          \
                      for (int i=0; i < (grid.x*grid.y*grid.z); i++){        \
                         device_MAX = max(device_MAX,__device_maxval_h[i]);  \
                      }   
///////////////////////////////////
// 
// 

#define av4(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-1)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy-1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vx(A)    (( A[ ix+1    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix +1   + (iy-1)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) + (iy-1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vy(A)    (( A[ ix    +  (iy+1)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  (iy+1)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy-0)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vz(A)    (( A[ ix    +  iy   *size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  iy   *size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-1)*size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy-1)*size(A,1) +  (iz+1)  *size(A,1)*size(A,2)] )*(DAT)0.25)

#define av4xz(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vxxz(A)    (( A[ ix+1    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix +1   + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vyxz(A)    (( A[ ix    +  (iy+1)   *size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  (iy+1)   *size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vzxz(A)    (( A[ ix    +  (iy+0)   *size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  (iy+0)   *size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy+0)*size(A,1) +  (iz-0)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy+0)*size(A,1) +  (iz-0)  *size(A,1)*size(A,2)] )*(DAT)0.25)


#define av4yz(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) +  (iy-1)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) + (iy-1)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vxyz(A)    (( A[ ix+1    +  (iy-1)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix +1   + (iy-1)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vyyz(A)    (( A[ (ix+0)    +  (iy+0)   *size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix+0) +  (iy+1)   *size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[ (ix+0)    + (iy+0)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+0) + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4vzyz(A)    (( A[ ix+0    +  (iy+0)   *size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix+0) +  (iy-1)   *size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[ ix+0    + (iy+0)*size(A,1) +  (iz-0)  *size(A,1)*size(A,2)]  \
   +                 A[(ix+0) + (iy-1)*size(A,1) +  (iz-0)  *size(A,1)*size(A,2)] )*(DAT)0.25)


#define av4xzD(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy-0)*size(A,1) +  (iz+1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4yzD(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) +  (iy-1)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  (iz+1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) + (iy-1)*size(A,1) +  (iz+1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4xzD2(A)   (( A[ ix    +  (iy+0)   *(size(A,1)-0) +  iz  *(size(A,1)-0)*size(A,2)]  \
   +                 A[(ix-1) +  (iy+0)   *(size(A,1)-0) +  iz  *(size(A,1)-0)*size(A,2)]  \
   +                 A[ ix    + (iy+0)*(size(A,1)-0) +  (iz-1)  *(size(A,1)-0)*size(A,2)]  \
   +                 A[(ix-1) + (iy+0)*(size(A,1)-0) +  (iz-1)  *(size(A,1)-0)*size(A,2)] )*(DAT)0.25)
#define av4D3(A)    (( A[ ix    +  (iy+0)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) +  (iy+0)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-1) + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4yzDD(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) +  (iy+1)   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix-0) + (iy+1)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4xzDDD(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) + (iy-0)*size(A,1) +  (iz-1)  *size(A,1)*size(A,2)] )*(DAT)0.25)
#define av4DDD(A)    (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy-1)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) + (iy-1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)

#define av_xya(A) (( A[ ix    +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) +  iy   *size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[ ix    + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)]  \
   +                 A[(ix+1) + (iy+1)*size(A,1) +  iz  *size(A,1)*size(A,2)] )*(DAT)0.25)

#define x_s    (Lx - Lx/2.0)
#define y_s    (Ly - Ly/2.0)
#define z_s    (Lz - Lz/2.0)
// Timer 
int saveData = 0;
DAT GPUinfo[3];
cudaEvent_t startD, stopD;
float milliseconds = 0;
void save_info(){
    FILE* fid;
    if (me==0){ fid=fopen("0_infos.inf", "w"); fprintf(fid,"%d %d %d %d %d %d",PRECIS,nx,ny,nz,NB_PARAMS,(int)ceil((DAT)niter/(DAT)nout));  fclose(fid);}
}

#include <stdarg.h>
int vscprintf(const char* format, va_list ap)
{
    va_list ap_copy;
    va_copy(ap_copy, ap);
    int retval = vsnprintf(NULL, 0, format, ap_copy);
    va_end(ap_copy);
    return retval;
}
int vasprintf(char** strp, const char* format, va_list ap)
{
    int len = vscprintf(format, ap);
    if (len == -1)
        return -1;
    char* str = (char*)malloc((size_t)len + 1);
    if (!str)
        return -1;
    int retval = vsnprintf(str, len + 1, format, ap);
    if (retval == -1) {
        free(str);
        return -1;
    }
    *strp = str;
    return retval;
}
int asprintf(char** strp, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    int retval = vasprintf(strp, format, ap);
    va_end(ap);
    return retval;
}

void save_array(DAT* A, size_t nb_elems, const char A_name[], int isave){
    char* fname; FILE* fid;
    asprintf(&fname, "%d_%d_%s.res" ,isave, me, A_name); 
    fid=fopen(fname, "wb"); fwrite(A, PRECIS, nb_elems, fid); fclose(fid); free(fname);
}
#define SaveArray(A,A_name)  gather(A); save_array(A##_h, numel(A), A_name, isave);

void read_data(DAT* A_h, DAT* A_d, int nx,int ny,int nz, const char A_name[], const char B_name[],int isave){
    char* bname; size_t nb_elems = nx*ny*nz; FILE* fid;
    asprintf(&bname, "%d_%d_%s.%s", isave, me, A_name, B_name);
    fid=fopen(bname, "rb"); // Open file
    if (!fid){ fprintf(stderr, "\nUnable to open file %s \n", bname); return; }
    fread(A_h, PRECIS, nb_elems, fid); fclose(fid);
    cudaMemcpy(A_d, A_h, nb_elems*sizeof(DAT), cudaMemcpyHostToDevice);
    if (me==0) printf("Read data: %d files %s.%s loaded (size = %dx%dx%d) \n", nprocs,A_name,B_name,nx,ny,nz); free(bname);
}

void read_data_h(DAT* A_h, int nx, int ny,int nz, const char A_name[], const char B_name[],int isave){
    char* bname; size_t nb_elems = nx*ny*nz; FILE* fid;
    asprintf(&bname, "%d_%d_%s.%s", isave, me, A_name, B_name);
    fid=fopen(bname, "rb"); // Open file
    if (!fid){ fprintf(stderr, "\nUnable to open file %s \n", bname); return; }
    fread(A_h, PRECIS, nb_elems, fid); fclose(fid);
    if (me==0) printf("Read data: %d files %s.%s loaded (size = %dx%dx%d) \n", nprocs,A_name,B_name,nx,ny,nz); free(bname);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define Dx  ( (DAT)1.0/dx )
#define Dy  ( (DAT)1.0/dy )
#define Dz  ( (DAT)1.0/dz )

#define load(A,nx,ny,Aname) DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*sizeof(DAT));  \
                            FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(DAT), (nx)*(ny), A##fid); fclose(A##fid); \
                            cudaMalloc(&A##_d,((nx)*(ny))*sizeof(DAT)); \
                            cudaMemcpy(A##_d,A##_h,((nx)*(ny))*sizeof(DAT),cudaMemcpyHostToDevice);  
#define  swap(A,B,tmp)      DAT *tmp; tmp = A##_d; A##_d = B##_d; B##_d = tmp;

__global__ void compute_StressPrf(DAT* Prf, DAT* sigma_xx, DAT* sigma_yy,DAT* sigma_zz, DAT* sigma_xy, DAT* sigma_xz, DAT* sigma_yz, DAT* Vx, DAT* Vy,DAT* Vz, DAT* Qxft, DAT* Qyft,DAT* Qzft, const DAT dx, const DAT dy,const DAT dz,const DAT dt,const DAT c11u,const DAT c22u,const DAT c33u,const DAT c12u,const DAT c13u,const DAT c23u,const DAT c44u,const DAT c55u,const DAT c66u,const DAT alpha1,const DAT alpha2,const DAT alpha3,const DAT M1, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    #define diffVx  (   (Vx[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
    #define diffVy  (   (Vy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
    #define diffVz  (   (Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )
    #define div_Qf  ( (((DAT)1.0/dx)*(Qxft[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]-Qxft[ix + iy*(nx+1) + iz*(nx+1)*(ny  )])  + ((DAT)1.0/dy)*(Qyft[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]-Qyft[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]) + ((DAT)1.0/dz)*(Qzft[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]-Qzft[ix + iy*(nx  ) + iz*(nx  )*(ny  )]) )    )

    if (iz<nz && iy<ny && ix<nx){
        DAT Vxx = diffVx;
        DAT Vyy = diffVy;
        DAT Vzz = diffVz;
        DAT Qff = div_Qf;
        sigma_xx[ix + iy*nx + iz*nx*ny]     = sigma_xx[ix + iy*nx + iz*nx*ny] + dt*(c11u* Vxx + c12u* Vyy + c13u* Vzz + alpha1*M1*Qff); 
        sigma_yy[ix + iy*nx + iz*nx*ny]     = sigma_yy[ix + iy*nx + iz*nx*ny] + dt*(c12u* Vxx + c22u* Vyy + c23u* Vzz + alpha2*M1*Qff); 
        sigma_zz[ix + iy*nx + iz*nx*ny]     = sigma_zz[ix + iy*nx + iz*nx*ny] + dt*(c13u* Vxx + c23u* Vyy + c33u* Vzz + alpha3*M1*Qff); 
        Prf[ix + iy*nx + iz*nx*ny] = Prf[ix + iy*nx + iz*nx*ny] + dt*(  -alpha1* M1* Vxx  - alpha2* M1* Vyy - alpha3* M1* Vzz - M1* Qff );
    }
    if (iz<nz && iy>0 && iy<ny && ix>0 && ix<nx){ sigma_xy[ix + iy*(nx-1) + iz*(nx-1)*(ny-1)] = sigma_xy[ix + iy*(nx-1) + iz*(nx-1)*(ny-1)] + c66u*dt*( (Vy[ix+0 + iy* nx + iz*nx  *(ny+1)] - Vy[ix-1 + iy* nx + iz*nx*(ny+1)  ])*((DAT)1.0/dx) + (Vx[ix + (iy+0)* (nx+1)+ iz*(nx+1)*(ny-0)] - Vx[ix  +(iy-1)*(nx+1) +  iz   *(nx+1)*ny])*((DAT)1.0/dy) ); }
    if (iz>0 && iz<nz && iy<ny && ix>0 && ix<nx){ sigma_xz[ix + iy*(nx-1) + iz*(nx-1)* ny   ] = sigma_xz[ix + iy*(nx-1) + iz*(nx-1)* ny   ] + c55u*dt*( (Vz[ix + iy* nx + iz*nx  * ny   ] - Vz[ix-1 + iy* nx + iz*nx* ny     ])*((DAT)1.0/dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)* ny   ] - Vx[ix + iy* (nx+1)    + (iz-1)*(nx+1)*ny])*((DAT)1.0/dz) ); }
    if (iz>0 && iz<nz && iy>0 && iy<ny && ix<nx){ sigma_yz[ix + iy* nx    + iz* nx   *(ny-1)] = sigma_yz[ix + iy* nx    + iz*nx*(ny-1)    ] + c44u*dt*( (Vy[ix + iy* nx + iz*nx*  (ny+1)] - Vy[ix + iy* nx + (iz-1)*nx*(ny+1)])*((DAT)1.0/dz) + (Vz[ix + iy* nx    + iz*nx*ny        ] - Vz[ix + (iy-1)* nx    +  iz   * nx   *ny])*((DAT)1.0/dy) ); }

    #undef diffVx
    #undef diffVy
    #undef diffVz
    #undef div_Qf
}

__global__ void update_Qxft(DAT* Prf, DAT* sigma_xx, DAT* sigma_yy,DAT* sigma_zz, DAT* sigma_xy, DAT* sigma_xz, DAT* sigma_yz, DAT* Vx, DAT* Vy,DAT* Vz, DAT* Qxft, DAT* Qyft,DAT* Qzft,DAT* Qxold,DAT* Qyold,DAT* Qzold, const DAT dx, const DAT dy,const DAT dz,const DAT dt,const DAT eta_k1,const DAT eta_k2,const DAT eta_k3,const DAT mm1,const DAT mm2,const DAT mm3,const DAT delta1,const DAT delta2,const DAT delta3,const DAT rho_fluid1,const DAT rho_fluid2,const DAT rho_fluid3,const DAT rho1,const DAT rho2,const DAT rho3, const int nx, const int ny, const int nz, const DAT chi){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    #define div_Sigmax  ( ( sigma_xx[ix+0 + iy*nx + iz*nx*ny]-sigma_xx[(ix-1) + (iy  )*nx + (iz  )*nx*ny] )*((DAT)1.0/dx) + ( sigma_xy[ix   + (iy+1)*(nx-1) + iz*(nx-1)*(ny-1)] - sigma_xy[ix + (iy-0)*(nx-1) + iz*(nx-1)*(ny-1)])*((DAT)1.0/dy) + (sigma_xz[ix + (iy+0)*(nx-1) + (iz+1)* (nx-1)*(ny+0)] - sigma_xz[ix + iy*(nx-1) + iz*(nx-1)*(ny+0)])*((DAT)1.0/dz) )
    #define div_Sigmay  ( ( sigma_yy[ix + iy*nx + iz*nx*ny]-sigma_yy[(ix  ) + (iy-1)*nx + (iz  )*nx*ny] )*((DAT)1.0/dy) + ( sigma_xy[ix+1 + (iy  )*(nx-1) + iz*(nx-1)*(ny-1)] - sigma_xy[ix + iy*(nx-1) + iz*(nx-1)*(ny-1)])*((DAT)1.0/dx) + (sigma_yz[ix + (iy  )*(nx+0) + (iz+1)* (nx+0)*(ny-1)] - sigma_yz[ix + iy*(nx+0) + iz*(nx+0)*(ny-1)])*((DAT)1.0/dz) )
    #define div_Sigmaz  ( ( sigma_zz[ix + iy*nx + iz*nx*ny]-sigma_zz[(ix  ) + (iy  )*nx + (iz-1)*nx*ny] )*((DAT)1.0/dz) + ( sigma_xz[ix+1 + (iy  )*(nx-1) + iz*(nx-1)*(ny+0)] - sigma_xz[ix + iy*(nx-1) + iz*(nx-1)*(ny+0)])*((DAT)1.0/dx) + (sigma_yz[ix + (iy+1)*(nx+0) +  iz   * (nx+0)*(ny-1)] - sigma_yz[ix + iy*(nx+0) + iz*(nx+0)*(ny-1)])*((DAT)1.0/dy) )
    #define Q_gradPrfx  ( ( Prf[ix + iy*nx + iz*nx*ny]  -Prf[(ix-1) + (iy  )*nx + (iz  )*nx*ny] )*((DAT)1.0/dx) + ((DAT)1.0-chi )*( Qxold[ix + iy*(nx+1) +  iz*  (nx+1)*ny   ] )*eta_k1)
    #define Q_gradPrfy  ( ( Prf[ix + iy*nx + iz*nx*ny]  -Prf[(ix  ) + (iy-1)*nx + (iz  )*nx*ny] )*((DAT)1.0/dy) + ((DAT)1.0-chi )*( Qyold[ix + iy*(nx  ) + (iz  )*nx*  (ny+1)] )*eta_k2)
    #define Q_gradPrfz  ( ( Prf[ix + iy*nx + iz*nx*ny]  -Prf[(ix  ) + (iy  )*nx + (iz-1)*nx*ny] )*((DAT)1.0/dz) + ((DAT)1.0-chi )*( Qzold[ix + iy*(nx  ) +  iz*   nx*   ny   ] )*eta_k3)
    
    if (iz > 0 && iz<nz-1 && iy>0 && iy<ny-1 && ix>0 && ix<nx){
        DAT QPx = Q_gradPrfx;
        DAT dSx = div_Sigmax;
        Qxft[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] =  ( Qxold[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]*((DAT)1.0/dt) - rho1* delta1* QPx - rho_fluid1* delta1* dSx )* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*rho1* delta1*eta_k1 ));
        Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   = ( Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]*((DAT)1.0/dt) + mm1* delta1* dSx + rho_fluid1* delta1* ( QPx + chi*eta_k1*Qxft[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] )  )*dt;
    }   
    if (iz > 0 && iz < nz-1 && iy>0 && iy < ny && ix>0 && ix < nx-1){
        DAT QPy = Q_gradPrfy;
        DAT dSy = div_Sigmay;
        Qyft[ix + iy*(nx  ) + iz*(nx  )*(ny+1)] = ( Qyold[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]*((DAT)1.0/dt) - rho2* delta2* QPy - rho_fluid2* delta2* dSy )* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*rho2* delta2*eta_k2 )  );
        Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   = ( Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]*((DAT)1.0/dt) + mm2* delta2* dSy +  rho_fluid2* delta2* ( QPy +  chi*eta_k2*Qyft[ix + iy*(nx  )   + iz*(nx  )*(ny+1)] )  )*dt;
    }
    if (iz > 0 && iz < nz && iy>0 && iy < ny-1 && ix>0 && ix < nx-1){
        DAT QPz = Q_gradPrfz;
        DAT dSz = div_Sigmaz;
        Qzft[ix + iy*(nx  ) + iz*(nx  )*(ny  )] =  ( Qzold[ix + iy*(nx  ) + iz*(nx  )*(ny  )]*((DAT)1.0/dt) - rho3* delta3* QPz - rho_fluid3* delta3* dSz )* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*rho3* delta3*eta_k3 )  );
        Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   =  ( Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]*((DAT)1.0/dt) + mm3* delta3* dSz +  rho_fluid3* delta3* ( QPz +  chi*eta_k3*Qzft[ix + iy*(nx  )   + iz*(nx  )*(ny  )] )  )*dt;
    }
    #undef div_Sigmax
    #undef div_Sigmay
    #undef div_Sigmaz
    #undef Q_gradPrfx
    #undef Q_gradPrfy
    #undef Q_gradPrfz
    #undef Q_gradPrfx1
    #undef Q_gradPrfy1
    #undef Q_gradPrfz1
}

__global__ void compute_old(DAT* QxoldINCR, DAT* QyoldINCR, DAT* QzoldINCR, DAT* Qxft, DAT* Qyft, DAT* Qzft, DAT* Vfx, DAT* Vfy, DAT* Vfz, DAT* x, DAT* y, DAT* rad, DAT* radc, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Prf, DAT* Prf_old, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txyc_old, DAT* Txzc, DAT* Txzc_old, DAT* Tyzc, DAT* Tyzc_old, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Exyc, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, DAT divV0, DAT eta_ve, DAT eta, DAT G, DAT dt, DAT dt_rho, DAT Vpdt, DAT Lx, DAT Re, const DAT K, const DAT G0, const DAT rho11, const DAT dx, const DAT dy, DAT phi, const int nx, const int ny, const int nz, const DAT rad0) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    //def_sizes(x, nx, ny, nz);
    //def_sizes(y, nx, ny, nz);
    if (iy < ny && ix < nx && iz < nz) {
        Pt_old[ix + iy * nx + iz * nx * ny] = Pt[ix + iy * nx + iz * nx * ny];
        Prf_old[ix + iy * nx + iz * nx * ny] = Prf[ix + iy * nx + iz * nx * ny];
        Txx_old[ix + iy * nx + iz * nx * ny] = Txx[ix + iy * nx + iz * nx * ny];
        Tyy_old[ix + iy * nx + iz * nx * ny] = Tyy[ix + iy * nx + iz * nx * ny];
        Tzz_old[ix + iy * nx + iz * nx * ny] = Tzz[ix + iy * nx + iz * nx * ny];
        //lama[ix + iy * nx + iz * nx * ny] = lama[ix + iy * nx + iz * nx * ny] * (DAT)0.0;
    }
    if (iz < nz && iy>0 && iy < ny && ix>0 && ix < nx) {
      //if (iz < nz &&  iy < ny+1 &&  ix < nx+1) {
        Txyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] ;
    }
    if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) {
      //  if ( iz < nz+1 && iy < ny &&  ix < nx+1) {
        Txzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] ;
    }
    if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
      //  if ( iz < nz+1 &&  iy < ny+1 && ix < nx) {
        Tyzc_old[ix + iy * nx + iz * nx * (ny + 1)] = Tyzc[ix + iy * nx + iz * nx * (ny + 1)] ;
    }
    if (iz > 0 && iz < (nz - 1) && iy>0 && iy < (ny - 1) && ix>0 && ix < nx) {
        Ux[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Ux[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] * dt;
        Vfx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] / phi + Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)];
        QxoldINCR[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)];
    }
    if (iz > 0 && iz < (nz - 1) && iy>0 && iy < ny && ix>0 && ix < (nx - 1)) {
        Uy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Uy[ix + iy * (nx)+iz * (nx) * (ny + 1)] + Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] * dt;
        Vfy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] / phi + Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)];
        QyoldINCR[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)];
    }
    if (iz > 0 && iz < nz && iy>0 && iy < (ny - 1) && ix>0 && ix < (nx - 1)) {
        Uz[ix + iy * (nx)+iz * (nx) * (ny)] = Uz[ix + iy * (nx)+iz * (nx) * (ny)] + Vz[ix + iy * (nx)+iz * (nx) * (ny)] * dt;
        Vfz[ix + iy * (nx)+iz * (nx) * (ny)] = Qzft[ix + iy * (nx)+iz * (nx) * (ny)] / phi + Vz[ix + iy * (nx)+iz * (nx) * (ny)];
        QzoldINCR[ix + iy * (nx)+iz * (nx) * (ny)] = Qzft[ix + iy * (nx)+iz * (nx) * (ny)];
    }
}

__global__ void compute_old2(DAT* x, DAT* y, DAT* rad, DAT* radc, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Prf, DAT* Prf_old, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txyc_old, DAT* Txzc, DAT* Txzc_old, DAT* Tyzc, DAT* Tyzc_old, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Exyc, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, DAT divV0, DAT eta_ve, DAT eta, DAT G, DAT dt, DAT dt_rho, DAT Vpdt, DAT Lx, DAT Re, const DAT K, const DAT G0, const DAT rho11, const DAT dx, const DAT dy, const int nx, const int ny, const int nz, const DAT rad0) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    //def_sizes(x, nx, ny, nz);
    //def_sizes(y, nx, ny, nz);
    if (iy < ny && ix < nx && iz < nz) {
        Pt[ix + iy * nx + iz * nx * ny] = Pt_old[ix + iy * nx + iz * nx * ny];
        Prf[ix + iy * nx + iz * nx * ny] = Prf_old[ix + iy * nx + iz * nx * ny];
        Txx[ix + iy * nx + iz * nx * ny] = Txx_old[ix + iy * nx + iz * nx * ny];
        Tyy[ix + iy * nx + iz * nx * ny] = Tyy_old[ix + iy * nx + iz * nx * ny];
        Tzz[ix + iy * nx + iz * nx * ny] = Tzz_old[ix + iy * nx + iz * nx * ny];
        //lama[ix + iy * nx + iz * nx * ny] = lama[ix + iy * nx + iz * nx * ny] * (DAT)0.0;
    }
    if (iz < nz && iy>0 && iy < ny && ix>0 && ix < nx) {
        //if (iz < nz &&  iy < ny+1 &&  ix < nx+1) {
        Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = Txyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)];
    }
    if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) {
        //  if ( iz < nz+1 && iy < ny &&  ix < nx+1) {
        Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Txzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
    }
    if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
        //  if ( iz < nz+1 &&  iy < ny+1 && ix < nx) {
        Tyzc[ix + iy * nx + iz * nx * (ny + 1)] = Tyzc_old[ix + iy * nx + iz * nx * (ny + 1)];
    }
    //if (iz > 0 && iz < (nz - 1) && iy>0 && iy < (ny - 1) && ix>0 && ix < nx) {
    //    Ux[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Ux[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] * dt;
    //}
    //if (iz > 0 && iz < (nz - 1) && iy>0 && iy < ny && ix>0 && ix < (nx - 1)) {
    //    Uy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Uy[ix + iy * (nx)+iz * (nx) * (ny + 1)] + Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] * dt;
    //}
    //if (iz > 0 && iz < nz && iy>0 && iy < (ny - 1) && ix>0 && ix < (nx - 1)) {
    //    Uz[ix + iy * (nx)+iz * (nx) * (ny)] = Uz[ix + iy * (nx)+iz * (nx) * (ny)] + Vz[ix + iy * (nx)+iz * (nx) * (ny)] * dt;
    //}
}


__global__ void jaumannD(DAT* etam, DAT* Gdtm_etam, DAT* eta_vem, DAT* dt_rhom, DAT* Gdtm, DAT* Kdtm, DAT* Km, DAT* Krm, DAT* Gm, DAT* Grm, DAT* divWxy, DAT* divWxz, DAT* divWyz, DAT* Tdiffxy, DAT* Tdiffxz, DAT* Tdiffyz, DAT* advPt_old, DAT* advTxx_old, DAT* advTyy_old, DAT* advTzz_old, DAT* advTxyc_old, DAT* advTxzc_old, DAT* advTyzc_old, DAT* Txye, DAT* x, DAT* y, DAT* rad, DAT* radc, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Txy, DAT* Txz, DAT* Tyz, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txzc_old, DAT* Tyzc_old, DAT* Exyc, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, DAT divV0, DAT eta_ve, DAT eta, DAT G, DAT dt, DAT dt_rho, DAT Vpdt, DAT Lx, DAT Re, const DAT K, const DAT G0, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT rad0, DAT Gdt, DAT Kdt, DAT K_G) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y 
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
#define Exx2    (  (Vx[(ix+1) + iy *(nx+1)+ iz * (nx+1) * ny]   -  Vx[ix + iy*(nx+1)+ iz * (nx+1) * ny]   )*((DAT)1.0/dx)   )
#define Eyy2    (  (Vy[ ix    + (iy+1)* nx   + iz * nx * (ny+1)]-  Vy[ix + iy* nx   + iz * nx * (ny+1)]   )*((DAT)1.0/dy)   )
#define Ezz2    (  (Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )

    if (iz < nz  && iy < ny && ix < nx) {
        dt_rhom[ix + iy * nx + iz * nx * ny] =  Vpdt * Lx * ((DAT)1.0 / (Re * eta_vem[ix + iy * nx + iz * nx * ny]));
        Gdtm[ix + iy * nx + iz * nx * ny] =  Vpdt * Vpdt * ((DAT)1.0 / (dt_rhom[ix + iy * nx + iz * nx * ny] * (K_G + (DAT)4.0 / (DAT)3.0)));

        Grm[ix + iy * nx + iz * nx * ny] = Gdtm[ix + iy * nx + iz * nx * ny] * ((DAT)1.0 / (Gm[ix + iy * nx + iz * nx * ny] * dt)); //Kr = Kdt / (K * dt);
        Kdtm[ix + iy * nx + iz * nx * ny] = K_G * Gdtm[ix + iy * nx + iz * nx * ny];
        Krm[ix + iy * nx + iz * nx * ny] = Kdtm[ix + iy * nx + iz * nx * ny] * ((DAT)1.0 / (Km[ix + iy * nx + iz * nx * ny] * dt));

        Gdtm_etam[ix + iy * nx + iz * nx * ny] = Gdtm[ix + iy * nx + iz * nx * ny] / etam[ix + iy * nx + iz * nx * ny];
    }
    if (iz < nz && iy < ny && ix < nx) {
        //Txy[ix + (iy) * (nx + 0) + iz * nx * ny] = ((DAT)0.25 * Txyc[ix + (iy + 0) * (nx + 1) + iz * (nx + 1) * (ny + 1)] + (DAT)0.25 * Txyc[ix + 1 + (iy + 1) * (nx + 1) + iz * (nx + 1) * (ny + 1)] + (DAT)0.25 * Txyc[ix + 0 + (iy + 1) * (nx + 1) + iz * (nx + 1) * (ny + 1)] + Txyc[ix + 1 + (iy + 0) * (nx + 1) + iz * (nx + 1) * (ny + 1)] * ((DAT)1.0 / (DAT)4.0));
    }
    if (iz < nz && iy < ny && ix < nx) {
        //Txz[ix + (iy) * (nx + 0) + iz * nx * ny] = ((DAT)0.25 * Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)0.25 * Txzc[ix + 1 + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)0.25 * Txzc[ix + iy * (nx + 1) + (iz + 1) * (nx + 1) * ny] + Txzc[ix + 1 + iy * (nx + 1) + (iz + 1) * (nx + 1) * ny] * ((DAT)1.0 / (DAT)4.0));
    }
    if (iz < nz && iy < ny && ix < nx) {
        //Tyz[ix + iy * nx + iz * nx * (ny + 0)] = ((DAT)0.25 * Tyzc[ix + iy * nx + iz * nx * (ny + 1)] + (DAT)0.25 * Tyzc[ix + (iy + 1) * nx + (iz + 1) * nx * (ny + 1)] + (DAT)0.25 * Tyzc[ix + (iy + 1) * nx + iz * nx * (ny + 1)] + Tyzc[ix + iy * nx + (iz + 1) * nx * (ny + 1)] * ((DAT)1.0 / (DAT)4.0));
    }
    if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
        //av4xzDDD_Txyc[ix + iy * nx + iz * nx * (ny + 1)] =  100.0 * Txyc[ix + (iy + 0) * (nx + 1) + (iz - 0) * (nx + 1) * (ny + 1)];
    }
}//

__global__ void jaumannD2(DAT* divWxy, DAT* divWxz, DAT* divWyz, DAT* Tdiffxy, DAT* Tdiffxz, DAT* Tdiffyz, DAT* advPt_old, DAT* advTxx_old, DAT* advTyy_old, DAT* advTzz_old, DAT* advTxyc_old, DAT* advTxzc_old, DAT* advTyzc_old, DAT* Txye, DAT* x, DAT* y, DAT* rad, DAT* radc, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Txy, DAT* Txz, DAT* Tyz, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txzc_old, DAT* Tyzc_old, DAT* Exyc, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, DAT divV0, DAT eta_ve, DAT eta, DAT G, DAT dt, DAT dt_rho, DAT Vpdt, DAT Lx, DAT Re, const DAT K, const DAT G0, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT rad0) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y 
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    def_sizes(divWxy, nx, ny, nz);
    def_sizes(divWxz, nx, ny, nz);
    def_sizes(divWyz, nx, ny, nz);
    def_sizes(Tdiffxy, nx, ny, nz);
    def_sizes(Tdiffxz, nx, ny, nz);
    def_sizes(Tdiffyz, nx, ny, nz);
    //def_sizes(Txyc_old, nx + 1, ny + 1, nz);
    def_sizes(Txyc, nx + 1, ny + 1, nz);
    //def_sizes(Txzc_old, nx + 1, ny, nz + 1);
    //def_sizes(Tyzc_old, nx, ny+1, nz+1);
    def_sizes(Txzc, nx + 1, ny, nz + 1);
    def_sizes(Tyzc, nx, ny + 1, nz + 1);
    
    if ( iz > 0 && iz < (nz - 1) && iy > 0 && iy < (ny - 1) && ix > 0 && ix < (nx - 1) ) {
        Txx_old[ix + iy * nx + iz * nx * ny] = Txx_old[ix + iy * nx + iz * nx * ny] + (DAT)2.0 * ( Txy[ix + (iy) * (nx + 0) + iz * nx * ny] * divWxy[ix + iy * nx + iz * nx * ny] + Txz[ix + (iy) * (nx + 0) + iz * nx * ny] * divWxz[ix + iy * nx + iz * nx * ny]);
        Tyy_old[ix + iy * nx + iz * nx * ny] = Tyy_old[ix + iy * nx + iz * nx * ny] - (DAT)2.0 * (  Txy[ix + (iy) * (nx + 0) + iz * nx * ny] * divWxy[ix + iy * nx + iz * nx * ny]  - Tyz[ix + (iy) * (nx + 0) + iz * nx * ny] * divWyz[ix + iy * nx + iz * nx * ny]);
        Tzz_old[ix + iy * nx + iz * nx * ny] = Tzz_old[ix + iy * nx + iz * nx * ny] - (DAT)2.0 * (  Txz[ix + (iy) * (nx + 0) + iz * nx * ny] * divWxz[ix + iy * nx + iz * nx * ny]  + Tyz[ix + (iy) * (nx + 0) + iz * nx * ny] * divWyz[ix + iy * nx + iz * nx * ny]);
    }
    if (iz < nz && iy > 0 && iy < ny && ix>0 && ix < nx) {
        Txyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] =  Txyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] +av4(Tdiffxy) * av4(divWxy) + av4xzD(Tyzc) * av4(divWxz) + av4yzD(Txzc) * av4(divWyz);//
    }
    if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) {
        Txzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Txzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + av4xz(Tdiffxz) * av4xz(divWxz) +av4D3(Tyzc) * av4xz(divWxy) - av4yzDD(Txyc) * av4xz(divWyz);
    }
    if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
        Tyzc_old[ix + iy * nx + iz * nx * (ny + 1)] = Tyzc_old[ix + iy * nx + iz * nx * (ny + 1)] + av4yz(Tdiffyz) * av4yz(divWyz) - av4xzDDD(Txyc)* av4yz(divWxz) - av4DDD(Txzc) * av4yz(divWxy);//-  av4yz(divWxz)  av4xzDDD(Txyc_old) *0.0*av4xzDDD(Txyc_old) *    -av4xz(Txyc_old) * av4yz(divWxy);
    }//- 0.25 * Txyc[ix-0 + (iy) * (nx + 1) + (iz-1) * (nx+1) * (ny+1)]
#undef Exx2
#undef Eyy2
#undef Ezz2
 }

__global__ void adv2(DAT* Prf_old, DAT* advPrf_old, DAT* advPt_old, DAT* advTxx_old, DAT* advTyy_old, DAT* advTzz_old, DAT* advTxyc_old, DAT* advTxzc_old, DAT* advTyzc_old, DAT* Txye, DAT* x, DAT* y, DAT* rad, DAT* radc, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Txy, DAT* Txyc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txzc_old, DAT* Tyzc_old, DAT* Exyc, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, DAT divV0, DAT eta_ve, DAT eta, DAT G, DAT dt, DAT dt_rho, DAT Vpdt, DAT Lx, DAT Re, const DAT K, const DAT G0, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT rad0) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y 
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    if (iz > 0 && iz < nz - 1 && iy > 0 && iy < ny - 1 && ix > 0 && ix < nx - 1) {
        Pt_old[ix + iy * nx + iz * nx * ny] = Pt_old[ix + iy * nx + iz * nx * ny] + advPt_old[ix + iy * nx + iz * nx * ny] * dt; //
        Prf_old[ix + iy * nx + iz * nx * ny] = Prf_old[ix + iy * nx + iz * nx * ny] + advPrf_old[ix + iy * nx + iz * nx * ny] * dt;
        Txx_old[ix + iy * nx + iz * nx * ny] = Txx_old[ix + iy * nx + iz * nx * ny] + advTxx_old[ix + iy * nx + iz * nx * ny] * dt;
        Tyy_old[ix + iy * nx + iz * nx * ny] = Tyy_old[ix + iy * nx + iz * nx * ny] + advTyy_old[ix + iy * nx + iz * nx * ny] * dt;
        Tzz_old[ix + iy * nx + iz * nx * ny] = Tzz_old[ix + iy * nx + iz * nx * ny] + advTzz_old[ix + iy * nx + iz * nx * ny] * dt;
    }
    if (iz > 0 && iz < nz-1 && iy > 1 && iy < ny-1 && ix>1 && ix < nx-1) {
        Txyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = Txyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] + advTxyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] * dt;
    }
    if (iz > 1 && iz < nz-1 && iy>0 && iy < ny-1 && ix>1 && ix < nx-1) {
        Txzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Txzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + advTxzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] * dt;
    }
    if (iz > 1 && iz < nz-1 && iy>1 && iy < ny-1 && ix>0 && ix < nx-1) {
        Tyzc_old[ix + iy * nx + iz * nx * (ny + 1)] = Tyzc_old[ix + iy * nx + iz * nx * (ny + 1)]+  advTyzc_old[ix + iy * nx + iz * nx * (ny + 1)] * dt;
    }
}

__global__ void compute_SaveFields0(DAT* advTxyc_old, DAT* advTxzc_old, DAT* advTyzc_old, DAT* J2U, DAT* divVsaveExx, DAT* Grm, DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
#define Exyc2U  (  (DAT)0.5*((Uy[ix + iy* nx + iz*nx  *(ny+1)] - Uy[ix-1 + iy* nx + iz*nx*(ny+1)  ]) * ((DAT)1.0 / dx) + (Ux[ix + iy* (nx+1)+ iz*(nx+1)*(ny-0)] - Ux[ix  +(iy-1)*(nx+1) +  iz   *(nx+1)*ny]) * ((DAT)1.0 / dy) )   )
#define Exzc2U  (  (DAT)0.5*((Uz[ix + iy* nx + iz*nx  * ny   ] - Uz[ix-1 + iy* nx + iz*nx* ny     ])*((DAT)1.0/dx) + (Ux[ix + iy* (nx+1)+ iz*(nx+1)* ny   ] - Ux[ix + iy* (nx+1)    + (iz-1)*(nx+1)*ny])*((DAT)1.0/dz) )   )
#define Eyzc2U  (  (DAT)0.5*((Uy[ix + iy* nx + iz*nx*  (ny+1)] - Uy[ix + iy* nx + (iz-1)*nx*(ny+1)])*((DAT)1.0/dz) + (Uz[ix + iy* nx    + iz*nx*ny        ] - Uz[ix + (iy-1)* nx    +  iz   * nx   *ny])*((DAT)1.0/dy) )   )

   
        if (iz > -1 && iz < nz - 0 && iy > 0 && iy < ny - 0 && ix>0 && ix < nx - 0) {
            //advTxyc_old[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = Exyc2U;
        }
        if (iz > 0 && iz < nz - 0 && iy>-1 && iy < ny - 0 && ix>0 && ix < nx - 0) {
            //advTxzc_old[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = Exzc2U;
        }
        if (iz > 0 && iz < nz - 0 && iy>0 && iy < ny - 0 && ix>-1 && ix < nx - 0) {
            //advTyzc_old[ix + iy * nx + iz * nx * (ny + 1)] = Eyzc2U;
        }
    

}
__global__ void compute_SaveFields(DAT* Ux_loop, DAT* advTxyc_old, DAT* advTxzc_old, DAT* advTyzc_old, DAT* J2U, DAT* divVsaveExy, DAT* divVsaveExz, DAT* divVsaveEyz, DAT* divVsaveExx, DAT* divVsaveEyy, DAT* divVsaveEzz, DAT* Grm, DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz, DAT dt) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
#define ExxU1  (   (dt * Vx[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  dt * Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
#define EyyU1  (   (dt * Vy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  dt * Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
#define EzzU1  (   (dt * Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  dt * Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )
#define ExxU  (   (Ux[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  Ux[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
#define EyyU  (   (Uy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  Uy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
#define EzzU  (   (Uz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  Uz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )
#define Exyc22  (  (DAT)0.5*((Vy[ix + iy* nx + iz*nx  *(ny+1)] - Vy[ix-1 + iy* nx + iz*nx*(ny+1)  ]) * ((DAT)1.0 / dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)*(ny-0)] - Vx[ix  +(iy-1)*(nx+1) +  iz   *(nx+1)*ny]) * ((DAT)1.0 / dy) )   )
#define Exzc22  (  (DAT)0.5*((Vz[ix + iy* nx + iz*nx  * ny   ] - Vz[ix-1 + iy* nx + iz*nx* ny     ])*((DAT)1.0/dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)* ny   ] - Vx[ix + iy* (nx+1)    + (iz-1)*(nx+1)*ny])*((DAT)1.0/dz) )   )
#define Eyzc22  (  (DAT)0.5*((Vy[ix + iy* nx + iz*nx*  (ny+1)] - Vy[ix + iy* nx + (iz-1)*nx*(ny+1)])*((DAT)1.0/dz) + (Vz[ix + iy* nx    + iz*nx*ny        ] - Vz[ix + (iy-1)* nx    +  iz   * nx   *ny])*((DAT)1.0/dy) )   )

    if (iz < nz && iy < ny && ix < nx) {
        //DAT divU = (ExxU + EyyU + EzzU);
        //divVsave[ix + iy * nx + iz * nx * ny] = (DAT)1.0 / (DAT)3.0 * (Exx + (DAT)0.0*Eyy + (DAT)0.0 * Ezz);
        divVsaveExx[ix + iy * nx + iz * nx * ny] = (ExxU1);
        divVsaveEyy[ix + iy * nx + iz * nx * ny] = (EyyU1);
        divVsaveEzz[ix + iy * nx + iz * nx * ny] = (EzzU1);
    }

    if (iz > 0 && iz < (nz - 1) && iy > 0 && iy < ny - 1 && ix>0 && ix < nx) {
        Ux_loop[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = dt * Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)];
    }

    if (iz < nz && iy>0 && iy < ny && ix>0 && ix < nx) {
        //DAT divU = (ExxU + EyyU + EzzU);
        //divVsave[ix + iy * nx + iz * nx * ny] = (DAT)1.0 / (DAT)3.0 * (Exx + (DAT)0.0*Eyy + (DAT)0.0 * Ezz);
        divVsaveExy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = (DAT)1.0 * (Exyc22);
    }
    if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) {
        divVsaveExz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (DAT)1.0 * (Exzc22);
    }
    if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
        divVsaveEyz[ix + iy * nx + iz * nx * (ny + 1)] = (DAT)1.0 * (Eyzc22);
    }
}

__global__ void compute_Dissipation(DAT* etam, DAT* Dis, DAT* Ux_loop, DAT* advTxyc_old, DAT* advTxzc_old, DAT* advTyzc_old, DAT* J2U, DAT* divVsaveExy, DAT* divVsaveExz, DAT* divVsaveEyz, DAT* divVsaveExx, DAT* divVsaveEyy, DAT* divVsaveEzz, DAT* Grm, DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz, DAT dt) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
#define ExxU1  (   (dt * Vx[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  dt * Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
#define EyyU1  (   (dt * Vy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  dt * Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
#define EzzU1  (   (dt * Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  dt * Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )
#define ExxU  (   (Ux[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  Ux[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
#define EyyU  (   (Uy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  Uy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
#define EzzU  (   (Uz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  Uz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )
#define Exyc22  (  (DAT)0.5*((Vy[ix + iy* nx + iz*nx  *(ny+1)] - Vy[ix-1 + iy* nx + iz*nx*(ny+1)  ]) * ((DAT)1.0 / dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)*(ny-0)] - Vx[ix  +(iy-1)*(nx+1) +  iz   *(nx+1)*ny]) * ((DAT)1.0 / dy) )   )
#define Exzc22  (  (DAT)0.5*((Vz[ix + iy* nx + iz*nx  * ny   ] - Vz[ix-1 + iy* nx + iz*nx* ny     ])*((DAT)1.0/dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)* ny   ] - Vx[ix + iy* (nx+1)    + (iz-1)*(nx+1)*ny])*((DAT)1.0/dz) )   )
#define Eyzc22  (  (DAT)0.5*((Vy[ix + iy* nx + iz*nx*  (ny+1)] - Vy[ix + iy* nx + (iz-1)*nx*(ny+1)])*((DAT)1.0/dz) + (Vz[ix + iy* nx    + iz*nx*ny        ] - Vz[ix + (iy-1)* nx    +  iz   * nx   *ny])*((DAT)1.0/dy) )   )

#define divVsmcxy    (  (DAT)0.25 *divVsaveExy[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * divVsaveExy[ix - 1 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * divVsaveExy[ix + 0 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + divVsaveExy[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define divVsmcxz    (  (DAT)0.25 *divVsaveExz[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * divVsaveExz[ix - 1 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * divVsaveExz[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + divVsaveExz[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define divVsmcyz    (  (DAT)0.25 *divVsaveEyz[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * divVsaveEyz[ix   + (iy - 1) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * divVsaveEyz[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + divVsaveEyz[ix  + (iy - 1) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )

    
    if (iz > 0 && iz < nz - 1 && iy > 0 && iy < ny - 1 && ix > 0 && ix < nx - 1) {

        Dis[ix + iy * nx + iz * nx * ny] = (DAT)2.0 * etam[ix + iy * nx + iz * nx * ny] * (  divVsaveExx[ix + iy * nx + iz * nx * ny] * divVsaveExx[ix + iy * nx + iz * nx * ny] + 
            divVsaveEyy[ix + iy * nx + iz * nx * ny] * divVsaveEyy[ix + iy * nx + iz * nx * ny] + divVsaveEzz[ix + iy * nx + iz * nx * ny] * divVsaveEzz[ix + iy * nx + iz * nx * ny] +  
            divVsmcxy* divVsmcxy + divVsmcxz* divVsmcxz + divVsmcyz* divVsmcyz  ) - (DAT)2.0/(DAT)3.0 * etam[ix + iy * nx + iz * nx * ny] *  
            (divVsaveExx[ix + iy * nx + iz * nx * ny] + divVsaveEyy[ix + iy * nx + iz * nx * ny] + divVsaveEzz[ix + iy * nx + iz * nx * ny])*  
            (divVsaveExx[ix + iy * nx + iz * nx * ny] + divVsaveEyy[ix + iy * nx + iz * nx * ny] + divVsaveEzz[ix + iy * nx + iz * nx * ny]);
    }
}

__host__ void compute_sum(DAT* residHzz, DAT* divVsaveEzz, DAT* Pt, DAT* Tzz, const int nx, const int ny, const int nz) {

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int iz = 1; iz < nz - 1; iz++) {
                residHzz[0] = residHzz[0] + Pt[ix + iy * nx + iz * nx * ny];
                residHzz[1] = residHzz[1] + Tzz[ix + iy * nx + iz * nx * ny];
                residHzz[2] = residHzz[2] + divVsaveEzz[ix + iy * nx + iz * nx * ny];
            }
        }
    }

}

__host__ void compute_sum11(DAT* residHxx, DAT* divVsaveExx, DAT* Pt, DAT* Txx, const int nx, const int ny, const int nz) {

    for (int ix = 1; ix < nx-1; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int iz = 0; iz < nz  ; iz++) {
                residHxx[0] = residHxx[0] + Pt[ix + iy * nx + iz * nx * ny];
                residHxx[1] = residHxx[1] + Txx[ix + iy * nx + iz * nx * ny];
                residHxx[2] = residHxx[2] + divVsaveExx[ix + iy * nx + iz * nx * ny];
            }
        }
    }
}

__host__ void compute_sum22(DAT* residHyy, DAT* divVsaveEyy, DAT* Pt, DAT* Tyy, const int nx, const int ny, const int nz) {

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 1; iy < ny-1; iy++) {
            for (int iz = 0; iz < nz  ; iz++) {
                residHyy[0] = residHyy[0] + Pt[ix + iy * nx + iz * nx * ny];
                residHyy[1] = residHyy[1] + Tyy[ix + iy * nx + iz * nx * ny];
                residHyy[2] = residHyy[2] + divVsaveEyy[ix + iy * nx + iz * nx * ny];
            }
        }
    }
}

__host__ void compute_sum7(DAT* residHxx , DAT* residHyy , DAT* residHzz , DAT* divVsaveExx , DAT* Pt , DAT* Txx , DAT* divVsaveEyy , DAT* Tyy , DAT* divVsaveEzz , DAT* Tzz , const int nx, const int ny, const int nz) {

    for (int ix = 0; ix < nx ; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int iz = 0; iz < nz; iz++) {
                residHxx[0] = residHxx[0] + Pt[ix + iy * nx + iz * nx * ny];
                residHxx[1] = residHxx[1] + Txx[ix + iy * nx + iz * nx * ny];
                residHxx[2] = residHxx[2] + divVsaveExx[ix + iy * nx + iz * nx * ny];

                residHyy[0] = residHyy[0] + Pt[ix + iy * nx + iz * nx * ny];
                residHyy[1] = residHyy[1] + Tyy[ix + iy * nx + iz * nx * ny];
                residHyy[2] = residHyy[2] + divVsaveEyy[ix + iy * nx + iz * nx * ny];

                residHzz[0] = residHzz[0] + Pt[ix + iy * nx + iz * nx * ny];
                residHzz[1] = residHzz[1] + Tzz[ix + iy * nx + iz * nx * ny];
                residHzz[2] = residHzz[2] + divVsaveEzz[ix + iy * nx + iz * nx * ny];
            }
        }
    }
}

__host__ void compute_sum4(DAT* residH, DAT* divVsaveExy,  DAT* Txyc, const int nx, const int ny, const int nz) {

    for (int ix = 0; ix < nx+1  ; ix++) {
        for (int iy = 0; iy < ny+1; iy++) {
            for (int iz = 1; iz < nz-1; iz++) {
                residH[0] = residH[0] + Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)];
                residH[2] = residH[2] + divVsaveExy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)];
            }
        }
    }
}

__host__ void compute_sum5(DAT* residH, DAT* divVsaveExz, DAT* Txzc, const int nx, const int ny, const int nz) {

    for (int ix = 0; ix < nx + 1; ix++) {
        for (int iy = 1; iy < ny - 1; iy++) {
            for (int iz = 0; iz < nz+1; iz++) {
                residH[0] = residH[0] + Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
                residH[2] = residH[2] + divVsaveExz[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
            }
        }
    }
}

__host__ void compute_sum6(DAT* residH, DAT* divVsaveEyz, DAT* Tyzc, const int nx, const int ny, const int nz) {

    for (int ix = 1; ix < nx - 1; ix++) {
        for (int iy = 0; iy < ny + 1; iy++) {
            for (int iz = 0; iz < nz + 1; iz++) {
                residH[0] = residH[0] + Tyzc[ix + iy * nx + iz * nx * (ny + 1)];
                residH[2] = residH[2] + divVsaveEyz[ix + iy * nx + iz * nx * (ny + 1)];
            }
        }
    }
}


//      Prf_d,     Vx_d,   Vy_d,   Vz_d,     Ux_d,    Uy_d,    Uz_d,   Pt_d,    Txx_d,    Tyy_d,              Txy_d,   Txyc_d,    Txz_d,    Txzc_d,     Tyz_d,    Tyzc_d,    Pt_old_d,    Txx_old_d,    Tyy_old_d,   Tzz_old_d,    Txy_old_d,    Txyc_old_d,     Txz_old_d,    Txzc_old_d,     Tyz_old_d,   Tyzc_old_d,    Exyc_d,    Qxft_d,   Qyft_d,   scale,    dStr_1,       dStr_2,    dStr_3,       Kdt,    Gdt,     Kr,     Gr,     eta,       rho11, dx, dy, dz, alpha1, M1, nx, ny, nz
__global__ void compute_Stress(DAT* Gdtm_etam, DAT* eta_vem, DAT* dt_rhom, DAT* Gdtm, DAT* Kdtm, DAT* Krm, DAT* Grm, DAT* Prf, DAT* Prf_old, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT* Qzft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const DAT alph, const DAT M, DAT B, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

#define Exx  (   (Vx[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
#define Eyy  (   (Vy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
#define Ezz  (   (Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )

//#define Exx    (  (Vx[(ix+1) + iy *(nx+1)]   -  Vx[ix + iy*(nx+1)]   )*((DAT)1.0/dx)   )
//#define Eyy    (  (Vy[ ix    + (iy+1)* nx   ]-  Vy[ix + iy* nx   ]   )*((DAT)1.0/dy)   )
#define Exyc2  (  (DAT)0.5*(  (Vy[ix + iy* nx + iz*nx  *(ny+1)] - Vy[ix-1 + iy* nx + iz*nx*(ny+1)  ]) * ((DAT)1.0 / dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)*(ny-0)] - Vx[ix  +(iy-1)*(nx+1) +  iz   *(nx+1)*ny]) * ((DAT)1.0 / dy) )   )
#define Exzc2  (  (DAT)0.5*((Vz[ix + iy* nx + iz*nx  * ny   ] - Vz[ix-1 + iy* nx + iz*nx* ny     ])*((DAT)1.0/dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)* ny   ] - Vx[ix + iy* (nx+1)    + (iz-1)*(nx+1)*ny])*((DAT)1.0/dz) )   )
#define Eyzc2  (  (DAT)0.5*((Vy[ix + iy* nx + iz*nx*  (ny+1)] - Vy[ix + iy* nx + (iz-1)*nx*(ny+1)])*((DAT)1.0/dz) + (Vz[ix + iy* nx    + iz*nx*ny        ] - Vz[ix + (iy-1)* nx    +  iz   * nx   *ny])*((DAT)1.0/dy) )   )

#define div_Qf ( (  Qxft[(ix+1) + iy  *(nx+1)+ (iz  )*(nx+1)*(ny  )]- Qxft[ix + iy*(nx+1)+ (iz  )*(nx+1)*(ny  )]  )*((DAT)1.0/dx) + (  Qyft[ ix + (iy+1)* nx + (iz  )*(nx  )*(ny+1)]-  Qyft[ix + iy* nx + (iz  )*(nx  )*(ny+1)]  )*((DAT)1.0/dy) + + (  Qzft[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]-  Qzft[(ix  ) + (iy  )*(nx  ) + (iz+0)*(nx  )*(ny  )]  )*((DAT)1.0/dz) )
#define Grmcxy    (  (DAT)0.25 *Grm[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Grm[ix - 1 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Grm[ix + 0 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + Grm[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define Grmcxz    (  (DAT)0.25 *Grm[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Grm[ix - 1 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * Grm[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + Grm[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define Grmcyz    (  (DAT)0.25 *Grm[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Grm[ix   + (iy - 1) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * Grm[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + Grm[ix  + (iy - 1) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )

#define Gdtmcxy    (  (DAT)0.25 *Gdtm[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm[ix - 1 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm[ix + 0 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + Gdtm[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define Gdtmcxz    (  (DAT)0.25 *Gdtm[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm[ix - 1 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * Gdtm[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + Gdtm[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define Gdtmcyz    (  (DAT)0.25 *Gdtm[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm[ix   + (iy - 1) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * Gdtm[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + Gdtm[ix  + (iy - 1) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )

#define Gdtm_etamcxy    (  (DAT)0.25 *Gdtm_etam[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm_etam[ix - 1 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm_etam[ix + 0 + (iy - 1) * (nx + 0)+ iz*nx  * ny] + Gdtm_etam[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define Gdtm_etamcxz    (  (DAT)0.25 *Gdtm_etam[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm_etam[ix - 1 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * Gdtm_etam[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + Gdtm_etam[ix - 1 + (iy + 0) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )
#define Gdtm_etamcyz    (  (DAT)0.25 *Gdtm_etam[ix + (iy + 0) * (nx + 0)+ iz*nx  * ny] + (DAT)0.25 * Gdtm_etam[ix   + (iy - 1) * (nx + 0)+ (iz-1)*nx  * ny] + (DAT)0.25 * Gdtm_etam[ix + 0 + (iy - 0) * (nx + 0)+ (iz-1)*nx  * ny] + Gdtm_etam[ix  + (iy - 1) * (nx + 0)+ iz*nx  * ny]* ((DAT)1.0 / (DAT)4.0)    )


    if (iz < nz && iy < ny && ix < nx) {
        if (Pt[ix + iy * nx + iz * nx * ny] < (DAT)0.0) {
            Pt[ix + iy * nx + iz * nx * ny] = (DAT)0.0;
        }
        DAT divV = (Exx + Eyy + Ezz); 
        Pt[ix + iy * nx + iz * nx * ny]  = (Pt[ix + iy * nx + iz * nx * ny]   - Kdtm[ix + iy * nx + iz * nx * ny] * (divV)) * ((DAT)1.0 / ((DAT)1.0 + Krm[ix + iy * nx + iz * nx * ny]));
        //Prf[ix + iy * nx + iz * nx * ny] = (Prf[ix + iy * nx + iz * nx * ny] + Prf_old[ix + iy * nx + iz * nx * ny] * Kr - Kdt * (B *divV   + (B / alph) * div_Qf)  ) * ((DAT)1.0 / ((DAT)1.0 + Kr));
        if (Pt[ix + iy * nx + iz * nx * ny] < (DAT)0.0) {
            Pt[ix + iy * nx + iz * nx * ny] = (DAT)0.0;
        }
        Txx[ix + iy * nx + iz * nx * ny] = (Txx[ix + iy * nx + iz * nx * ny]  + Gdtm[ix + iy * nx + iz * nx * ny] * (DAT)2.0 * (Exx - (DAT)1.0 / (DAT)3.0 * divV)) * ((DAT)1.0 / ((DAT)1.0  + Gdtm_etam[ix + iy * nx + iz * nx * ny] + Grm[ix + iy * nx + iz * nx * ny])); //+ 0.0 * Gdtm_etam[ix + iy * nx + iz * nx * ny] + 0.0 * Grm[ix + iy * nx + iz * nx * ny]
        Tyy[ix + iy * nx + iz * nx * ny] = (Tyy[ix + iy * nx + iz * nx * ny]  + Gdtm[ix + iy * nx + iz * nx * ny] * (DAT)2.0 * (Eyy - (DAT)1.0 / (DAT)3.0 * divV)) * ((DAT)1.0 / ((DAT)1.0  + Gdtm_etam[ix + iy * nx + iz * nx * ny] + Grm[ix + iy * nx + iz * nx * ny])); //+ 0.0 * Gdtm_etam[ix + iy * nx + iz * nx * ny] + 0.0 * Grm[ix + iy * nx + iz * nx * ny]
        Tzz[ix + iy * nx + iz * nx * ny] = (Tzz[ix + iy * nx + iz * nx * ny]  + Gdtm[ix + iy * nx + iz * nx * ny] * (DAT)2.0 * (Ezz - (DAT)1.0 / (DAT)3.0 * divV)) * ((DAT)1.0 / ((DAT)1.0  + Gdtm_etam[ix + iy * nx + iz * nx * ny] + Grm[ix + iy * nx + iz * nx * ny])); //+ 0.0 * Gdtm_etam[ix + iy * nx + iz * nx * ny] + 0.0 * Grm[ix + iy * nx + iz * nx * ny]

    }//Gdtm[ix + iy * nx + iz * nx * ny] *
    if (iz < nz && iy>0 && iy < ny && ix>0 && ix < nx) {
        Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] =  (Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)]  + Gdtmcxy * (DAT)2.0 * Exyc2) * ((DAT)1.0 / ((DAT)1.0+ Gdtm_etamcxy + Grmcxy));//Grmcxy 0.0*Gdtm_etamcxy +
    }//Gdtmcxy
    if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) {
        Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny]  + Gdtmcxz * (DAT)2.0 * Exzc2) * ((DAT)1.0 / ((DAT)1.0 + Gdtm_etamcxz + Grmcxz));//Grmcxz + 0.0 * Gdtm_etamcxz
    }//Gdtmcxz
    if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
        Tyzc[ix + iy * nx + iz * nx * (ny + 1)] = (Tyzc[ix + iy * nx + iz * nx * (ny + 1)]  + Gdtmcyz * (DAT)2.0 * Eyzc2) * ((DAT)1.0 / ((DAT)1.0+ Gdtm_etamcyz + Grmcyz));//Grmcyz + 0.0 * Gdtm_etamcyz
    }//Gdtmcyz
    //if (iz < nz && iy>0 && iy < ny && ix>0 && ix < nx) { sigma_xy[ix + iy * (nx - 1) + iz * (nx - 1) * (ny - 1)] = sigma_xy[ix + iy * (nx - 1) + iz * (nx - 1) * (ny - 1)] + c66u * dt * ((Vy[ix + 0 + iy * nx + iz * nx * (ny + 1)] - Vy[ix - 1 + iy * nx + iz * nx * (ny + 1)]) * ((DAT)1.0 / dx) + (Vx[ix + (iy + 0) * (nx + 1) + iz * (nx + 1) * (ny - 0)] - Vx[ix + (iy - 1) * (nx + 1) + iz * (nx + 1) * ny]) * ((DAT)1.0 / dy)); }
    //if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) { sigma_xz[ix + iy * (nx - 1) + iz * (nx - 1) * ny] = sigma_xz[ix + iy * (nx - 1) + iz * (nx - 1) * ny] + c55u * dt * ((Vz[ix + iy * nx + iz * nx * ny] - Vz[ix - 1 + iy * nx + iz * nx * ny]) * ((DAT)1.0 / dx) + (Vx[ix + iy * (nx + 1) + iz * (nx + 1) * ny] - Vx[ix + iy * (nx + 1) + (iz - 1) * (nx + 1) * ny]) * ((DAT)1.0 / dz)); }
    //if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) { sigma_yz[ix + iy * nx + iz * nx * (ny - 1)] = sigma_yz[ix + iy * nx + iz * nx * (ny - 1)] + c44u * dt * ((Vy[ix + iy * nx + iz * nx * (ny + 1)] - Vy[ix + iy * nx + (iz - 1) * nx * (ny + 1)]) * ((DAT)1.0 / dz) + (Vz[ix + iy * nx + iz * nx * ny] - Vz[ix + (iy - 1) * nx + iz * nx * ny]) * ((DAT)1.0 / dy)); }

}

__global__ void compute_INJ(DAT* INJ, DAT* Prf, DAT* Prf_old, DAT* Vx, DAT* Vy, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Grm, DAT* Txx, DAT* Tyy, DAT* Txy, DAT* Txyc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, DAT Krf, DAT Kfdt, const DAT rho11, const DAT dx, const DAT dy, const DAT alpha1, const DAT M1, const DAT alph, const DAT M, DAT B, DAT rho_f, DAT K, const int nx, const int ny, const int xsrc, const int ysrc, const int zsrc, const int xsrc2, const int ysrc2, const int zsrc2, int it) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    if (iz<(zsrc + 6) && iz>(zsrc - 5) && iy<(ysrc + 6) && iy>(ysrc - 5) && ix<(xsrc + 6) && ix>(xsrc - 5)) { Prf[ix + iy * nx + iz * nx * ny] = Prf[ix + iy * nx + iz * nx * ny] + M / rho_f * INJ[it]; }
    if (iz<(zsrc + 6) && iz>(zsrc - 5) && iy<(ysrc + 6) && iy>(ysrc - 5) && ix<(xsrc + 6) && ix>(xsrc - 5)) { Pt[ix + iy * nx + iz * nx * ny] = Pt[ix + iy * nx + iz * nx * ny] + K * B / rho_f *INJ[it]; }

    if (iz<(zsrc2 + 6) && iz>(zsrc2 - 5) && iy<(ysrc2 + 6) && iy>(ysrc2 - 5) && ix<(xsrc2 + 6) && ix>(xsrc2 - 5)) { Prf[ix + iy * nx + iz * nx * ny] = Prf[ix + iy * nx + iz * nx * ny] + M / rho_f * INJ[it]; }
    if (iz<(zsrc2 + 6) && iz>(zsrc2 - 5) && iy<(ysrc2 + 6) && iy>(ysrc2 - 5) && ix<(xsrc2 + 6) && ix>(xsrc2 - 5)) { Pt[ix + iy * nx + iz * nx * ny] = Pt[ix + iy * nx + iz * nx * ny] + K * B / rho_f * INJ[it]; }

}

__global__ void compute_BC1(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    if (iz < nz && iy < ny && ix < 1) {
        Prf[ix + iy * nx + iz * nx * ny] = (DAT)0.0* Pt[ix + iy * nx + iz * nx * ny];
    }
    if (iz < nz && iy < ny && ix > nx-2) {
        Prf[ix + iy * nx + iz * nx * ny] = (DAT)0.0 * Pt[ix + iy * nx + iz * nx * ny];
    }
    if (iz < nz && iy < 1 && ix < nx) {
        Prf[ix + iy * nx + iz * nx * ny] = (DAT)0.0 * Pt[ix + iy * nx + iz * nx * ny];
    }
    if (iz < nz && iy > ny-2 && ix < nx) {
        Prf[ix + iy * nx + iz * nx * ny] = (DAT)0.0 * Pt[ix + iy * nx + iz * nx * ny];
    }
    if (iz < 1 && iy < ny && ix < nx) {
        Prf[ix + iy * nx + iz * nx * ny] = (DAT)0.0 * Pt[ix + iy * nx + iz * nx * ny];
    }
    if (iz > nz-2 && iy < ny && ix < nx) {
        Prf[ix + iy * nx + iz * nx * ny] = (DAT)0.0 * Pt[ix + iy * nx + iz * nx * ny];
    }
    if (iz > 0 && iz < nz-1 && iy > 0 && iy < ny && ix>0 && ix < 2) {
       Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = (DAT)1.0 / (DAT)3.0 * Txyc[ix + 1 + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)];
    }
    if (iz > 0 && iz < nz-1 && iy > 0 && iy < ny && ix>nx - 2 && ix < nx) {
        Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = (DAT)1.0 / (DAT)3.0 * Txyc[ix - 1 + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)];
    }

    if (iz > 0 && iz < nz  && iy > 0 && iy < ny-1 && ix>0 && ix < 2) {
        Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (DAT)1.0 / (DAT)3.0 * Txzc[ix+1 + iy * (nx + 1) + iz * (nx + 1) * ny];
    }
    if (iz > 0 && iz < nz  && iy > 0 && iy < ny-1 && ix>nx - 2 && ix < nx) {
        Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (DAT)1.0 / (DAT)3.0 * Txzc[ix-1  + iy * (nx + 1) + (iz-0) * (nx + 1) * ny];
    }

    if (iz > 0 && iz < nz && ix > 0 && ix < nx - 1 && iy>0 && iy < 2) {
        Tyzc[ix + iy * nx + iz * nx * (ny + 1)] =  (DAT)1.0 / (DAT)3.0 * Tyzc[ix + (iy+1) * nx + iz * nx * (ny + 1)];
    }
    if (iz > 0 && iz < nz && ix > 0 && ix < nx - 1 && iy>ny-2 && iy < ny) {
        Tyzc[ix + iy * nx + iz * nx * (ny + 1)] =  (DAT)1.0 / (DAT)3.0 * Tyzc[ix + (iy-1) * nx + iz * nx * (ny + 1)];
    }
}

__global__ void compute_BC2(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 
    if (iz > 0 && iz < nz - 1 && ix > 0 && ix < nx && iy>0 && iy < 2) {
        Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = (DAT)1.0 / (DAT)3.0 * Txyc[ix  + (iy + 1) * (nx + 1) + iz * (nx + 1) * (ny + 1)];
    }
    if (iz > 0 && iz < nz - 1 && ix > 0 && ix < nx && iy>ny-2 && iy < ny) {
        Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = (DAT)1.0 / (DAT)3.0 * Txyc[ix + (iy - 1) * (nx + 1) + iz * (nx + 1) * (ny + 1)];
    }

    if (iz > 0 && iz < 2 && iy > 0 && iy < ny - 1 && ix>0 && ix < nx) {
        Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (DAT)1.0 / (DAT)3.0 * Txzc[ix  + iy * (nx + 1) + (iz + 1) * (nx + 1) * ny];
    }
    if (iz > nz - 2 && iz < nz && iy > 0 && iy < ny - 1 && ix>0 && ix < nx) {
        Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = (DAT)1.0 / (DAT)3.0 * Txzc[ix + iy * (nx + 1) + (iz - 1) * (nx + 1) * ny];
    }

    if (iy > 0 && iy < ny && ix > 0 && ix < nx - 1 && iz>0 && iz < 2) {
        Tyzc[ix + iy * nx + iz * nx * (ny + 1)] = (DAT)1.0 / (DAT)3.0 * Tyzc[ix + iy * nx + (iz+1) * nx * (ny + 1)];
    }
    if (iy > 0 && iy < ny && ix > 0 && ix < nx - 1 && iz>nz - 2 && iz < nz) {
        Tyzc[ix + iy * nx + iz * nx * (ny + 1)] = (DAT)1.0 / (DAT)3.0 * Tyzc[ix + iy * nx + (iz-1) * nx * (ny + 1)];
    }
}

__global__ void compute_BC3v2(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz < (nz - 0) && iy>-1 && iy < ny+1 && ix < 1) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = (DAT)0.0;
    }
    if (iz < (nz - 0) && iy>-1 && iy < ny+1 && ix>nx - 2 && ix < nx) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = (DAT)0.0;
    }// FOR Cxx test

    if (iz < 1 && iy < ny  && ix>-1 && ix < nx + 1) {
        Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = (DAT)0.0;
    }
    if (iz > (nz -2) && iz < nz &&  iy < ny && ix>-1 && ix < nx + 1) {
        Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = (DAT)0.0;
    }
}

// TEST 4
__global__ void compute_BC3(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz < (nz - 0) && iy>0 && iy < ny && ix < 1) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = (DAT)1E4;
    }
    if (iz < (nz - 0) && iy>0 && iy < ny && ix>nx - 2 && ix < nx) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = (DAT)0.0;
    }
}

__global__ void compute_BC4(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz < (nz - 0) && iy>-1 && iy < 1 && ix < nx) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + (iy + 1) * (nx)+iz * (nx) * (ny + 1)];
    }
    if (iz < (nz - 0) && iy>ny - 2 && iy < ny && ix < nx) {
        Vy[ix + (iy + 1) * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + (iy - 0) * (nx)+iz * (nx) * (ny + 1)];
    }
}

// TEST 5
__global__ void compute_BC5(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz > 0 && iz < nz && iy < (ny - 0) && ix < 1) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = (DAT)1E4;
    }
    if (iz > 0 && iz < nz && iy < (ny - 0) && ix>nx - 2 && ix < nx) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = (DAT)0.0;
    }
}

__global__ void compute_BC6(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz > -1 && iz < 1 && iy < (ny - 0) && ix < nx) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = Vz[ix + iy * (nx)+(iz + 1) * (nx) * (ny)];
    }
    if (iz > nz - 2 && iz < nz && iy < (ny - 0) && ix < nx) {
        Vz[ix + iy * (nx)+(iz + 1) * (nx) * (ny)] = Vz[ix + iy * (nx)+iz * (nx) * (ny)];
    }
}

// TEST 6
__global__ void compute_BC7(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz > 0 && iz < nz && iy < (1) && ix < nz) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = (DAT)1E4;
    }
    if (iz > 0 && iz < nz && iy >ny - 2 && iy < (ny) && ix < nx) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = (DAT)0.0;
    }
}

__global__ void compute_BC8(DAT* Prf, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Ux, DAT* Uy, DAT* Uz, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txz, DAT* Txzc, DAT* Tyz, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Txz_old, DAT* Txzc_old, DAT* Tyz_old, DAT* Tyzc_old, DAT* Exyc, DAT* Qxft, DAT* Qyft, DAT scale, DAT dStr_1, DAT dStr_2, DAT dStr_3, DAT Kdt, DAT Gdt, DAT Kr, DAT Gr, DAT eta, const DAT rho11, const DAT dx, const DAT dy, const DAT dz, const DAT alpha1, const DAT M1, const int nx, const int ny, const int nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z 

    if (iz > -1 && iz < 1 && iy < (ny - 0) && ix < nx) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = Vz[ix + iy * (nx)+(iz + 1) * (nx) * (ny)];
    }
    if (iz > nz - 2 && iz < nz && iy < (ny - 0) && ix < nx) {
        Vz[ix + iy * (nx)+(iz + 1) * (nx) * (ny)] = Vz[ix + iy * (nx)+iz * (nx) * (ny)];
    }
}

__global__ void compute_Stress2(DAT* coh0m, DAT* Prf, DAT* Txye, DAT* lam, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Vy_old, DAT* Vydif, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txz, DAT* Tyz, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exy, DAT* Exyc, DAT* coh, DAT* J2, DAT* J2c, DAT* cohc, DAT* Qxft, DAT* Qyft, DAT* Qxold, DAT* Qyold, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, const DAT fric, const DAT coh0, const DAT damp, const DAT dt, const DAT K, const DAT G0, DAT dt_rho, const DAT rho12, const DAT rho22, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT Dnx, const DAT Dny, const DAT chi, const DAT eta_k1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z


    if (iz < nz && iy < ny && ix < nx) {
        Txy[ix + (iy) * (nx + 0) + iz * nx * ny] = (   (DAT)0.25 * Txyc[ix + (iy + 0) * (nx + 1) + iz * (nx + 1) * (ny + 1)] + (DAT)0.25 * Txyc[ix + 1 + (iy + 1) * (nx + 1) + iz * (nx + 1) * (ny + 1)] + (DAT)0.25 * Txyc[ix + 0 + (iy + 1) * (nx + 1) + iz * (nx + 1) * (ny + 1)] + Txyc[ix + 1 + (iy + 0) * (nx + 1) + iz * (nx + 1) * (ny + 1)] * ((DAT)1.0 / (DAT)4.0)   );
    }
    if (iz < nz && iy < ny && ix < nx) {
        Txz[ix + (iy) * (nx + 0) + iz * nx * ny] =  ( (DAT)0.25 * Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)0.25 * Txzc[ix+1 + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)0.25 * Txzc[ix + iy * (nx + 1) + (iz+1) * (nx + 1) * ny] + Txzc[ix+1 + iy * (nx + 1) + (iz+1) * (nx + 1) * ny] * ((DAT)1.0 / (DAT)4.0)  );
    }
    if (iz < nz && iy < ny && ix < nx) {
        Tyz[ix + iy * nx + iz * nx * (ny + 0)] =   ( (DAT)0.25 * Tyzc[ix + iy * nx + iz * nx * (ny + 1)] + (DAT)0.25 * Tyzc[ix + (iy+1) * nx + (iz+1) * nx * (ny + 1)] + (DAT)0.25 * Tyzc[ix + (iy+1) * nx + iz * nx * (ny + 1)] + Tyzc[ix + iy * nx + (iz+1) * nx * (ny + 1)] * ((DAT)1.0 / (DAT)4.0)  );
    }
    if (iz < nz && iy < ny && ix < nx) {
        coh[ix + iy * nx + iz * nx * ny] = (Pt[ix + iy * nx + iz * nx * ny] - Prf[ix + iy * nx + iz * nx * ny]) * fric + coh0m[ix + iy * nx + iz * nx * ny];
    }
}

__global__ void compute_V0stop(DAT* Prf, DAT* Txye, DAT* lam, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Vy_old, DAT* Vydif, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txz, DAT* Tyz, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exy, DAT* Exyc, DAT* coh, DAT* J2, DAT* J2c, DAT* cohc, DAT* Qxft, DAT* Qyft, DAT* Qxold, DAT* Qyold, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, const DAT fric, const DAT coh0, const DAT damp, const DAT dt, const DAT K, const DAT G0, DAT dt_rho, const DAT rho12, const DAT rho22, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT Dnx, const DAT Dny, const DAT chi, const DAT eta_k1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z

    if (iz < nz && iy < ny && ix < nx) {
        //J2[ix + iy * nx + iz * nx * ny] = sqrt((Txx[ix + iy * nx + iz * nx * ny] * Txx[ix + iy * nx + iz * nx * ny] + Tyy[ix + iy * nx + iz * nx * ny] * Tyy[ix + iy * nx + iz * nx * ny] + Tzz[ix + iy * nx + iz * nx * ny] * Tzz[ix + iy * nx + iz * nx * ny]) * ((DAT)1.0 / (DAT)2.0) + Txy[ix + iy * nx + iz * nx * ny] * Txy[ix + iy * nx + iz * nx * ny] + Txz[ix + iy * nx + iz * nx * ny] * Txz[ix + iy * nx + iz * nx * ny] + Tyz[ix + iy * nx + iz * nx * ny] * Tyz[ix + iy * nx + iz * nx * ny]);
       // lam[ix + iy * nx + iz * nx * ny] = ((DAT)1.0 - coh[ix + iy * nx + iz * nx * ny] / J2[ix + iy * nx + iz * nx * ny]);

        if (lam[ix + iy * nx + iz * nx * ny] > (DAT)1.0) { lam[ix + iy * nx + iz * nx * ny] = (DAT)1.0; } //
        if (lam[ix + iy * nx + iz * nx * ny] < (DAT)1.0) {
            //flag_lam = 1.0;
            DAT sc_lam =  lam[ix + iy * nx + iz * nx * ny];
            Txx[ix + iy * nx + iz * nx * ny] = Txx[ix + iy * nx + iz * nx * ny] * sc_lam;
            Tyy[ix + iy * nx + iz * nx * ny] = Tyy[ix + iy * nx + iz * nx * ny] * sc_lam;
            Tzz[ix + iy * nx + iz * nx * ny] = Tzz[ix + iy * nx + iz * nx * ny] * sc_lam;
            Txy[ix + iy * nx + iz * nx * ny] = Txy[ix + iy * nx + iz * nx * ny] * sc_lam;
            Txz[ix + iy * nx + iz * nx * ny] = Txz[ix + iy * nx + iz * nx * ny] * sc_lam;
            Tyz[ix + iy * nx + iz * nx * ny] = Tyz[ix + iy * nx + iz * nx * ny] * sc_lam;
            //J2[ix + iy * nx + iz * nx * ny] = sqrt((Txx[ix + iy * nx + iz * nx * ny] * Txx[ix + iy * nx + iz * nx * ny] + Tyy[ix + iy * nx + iz * nx * ny] * Tyy[ix + iy * nx + iz * nx * ny] + Tzz[ix + iy * nx + iz * nx * ny] * Tzz[ix + iy * nx + iz * nx * ny]) * ((DAT)1.0 / (DAT)2.0) + Txy[ix + iy * nx + iz * nx * ny] * Txy[ix + iy * nx + iz * nx * ny] + Txz[ix + iy * nx + iz * nx * ny] * Txz[ix + iy * nx + iz * nx * ny] + Tyz[ix + iy * nx + iz * nx * ny] * Tyz[ix + iy * nx + iz * nx * ny]);
            //lama[ix + iy * nx + iz * nx * ny] = lama[ix + iy * nx + iz * nx * ny] + lam[ix + iy * nx + iz * nx * ny];
        }
    }


}

__global__ void compute_V0(DAT* Prf, DAT* Txye, DAT* lam, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Vy_old, DAT* Vydif, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txz, DAT* Tyz, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exy, DAT* Exyc, DAT* coh, DAT* J2, DAT* J2c, DAT* cohc, DAT* Qxft, DAT* Qyft, DAT* Qxold, DAT* Qyold, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, const DAT fric, const DAT coh0, const DAT damp, const DAT dt, const DAT K, const DAT G0, DAT dt_rho, const DAT rho12, const DAT rho22, DAT eta_ve, DAT eta_reg, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT Dnx, const DAT Dny, const DAT chi, const DAT eta_k1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z

    if (iz < nz && iy < ny && ix < nx) {
        J2[ix + iy * nx + iz * nx * ny] = sqrt((Txx[ix + iy * nx + iz * nx * ny] * Txx[ix + iy * nx + iz * nx * ny] + Tyy[ix + iy * nx + iz * nx * ny] * Tyy[ix + iy * nx + iz * nx * ny] + Tzz[ix + iy * nx + iz * nx * ny] * Tzz[ix + iy * nx + iz * nx * ny]) * ((DAT)1.0 / (DAT)2.0) +  Txy[ix + iy * nx + iz * nx * ny] * Txy[ix + iy * nx + iz * nx * ny] + Txz[ix + iy * nx + iz * nx * ny] * Txz[ix + iy * nx + iz * nx * ny] + Tyz[ix + iy * nx + iz * nx * ny] * Tyz[ix + iy * nx + iz * nx * ny]);
        DAT F = J2[ix + iy * nx + iz * nx * ny] - coh[ix + iy * nx + iz * nx * ny];
        if (F < (DAT)0.0) { F = (DAT)0.0; } //
        lam[ix + iy * nx + iz * nx * ny] = ((DAT)1.0 - F / J2[ix + iy * nx + iz * nx * ny] * eta_ve / (eta_ve + eta_reg));
        //lam[ix + iy * nx + iz * nx * ny] = (   (DAT)1.0 - coh[ix + iy * nx + iz * nx * ny] / J2[ix + iy * nx + iz * nx * ny]   );

        //if (lam[ix + iy * nx + iz * nx * ny] < (DAT)0.0) { lam[ix + iy * nx + iz * nx * ny] = (DAT)0.0; } //
        if (F > (DAT)0.0) {
            //flag_lam = 1.0;
            DAT sc_lam =   lam[ix + iy * nx + iz * nx * ny];
            Txx[ix + iy * nx + iz * nx * ny] = Txx[ix + iy * nx + iz * nx * ny] * sc_lam;
            Tyy[ix + iy * nx + iz * nx * ny] = Tyy[ix + iy * nx + iz * nx * ny] * sc_lam;
            Tzz[ix + iy * nx + iz * nx * ny] = Tzz[ix + iy * nx + iz * nx * ny] * sc_lam;
            Txy[ix + iy * nx + iz * nx * ny] =  Txy[ix + iy * nx + iz * nx * ny] * sc_lam;
            Txz[ix + iy * nx + iz * nx * ny] =  Txz[ix + iy * nx + iz * nx * ny] * sc_lam;
            Tyz[ix + iy * nx + iz * nx * ny] =  Tyz[ix + iy * nx + iz * nx * ny] * sc_lam;
            //J2[ix + iy * nx + iz * nx * ny] = sqrt((Txx[ix + iy * nx + iz * nx * ny] * Txx[ix + iy * nx + iz * nx * ny] + Tyy[ix + iy * nx + iz * nx * ny] * Tyy[ix + iy * nx + iz * nx * ny] + Tzz[ix + iy * nx + iz * nx * ny] * Tzz[ix + iy * nx + iz * nx * ny]) * ((DAT)1.0 / (DAT)2.0) + Txy[ix + iy * nx + iz * nx * ny] * Txy[ix + iy * nx + iz * nx * ny] + Txz[ix + iy * nx + iz * nx * ny] * Txz[ix + iy * nx + iz * nx * ny] + Tyz[ix + iy * nx + iz * nx * ny] * Tyz[ix + iy * nx + iz * nx * ny]);
            //lama[ix + iy * nx + iz * nx * ny] = lama[ix + iy * nx + iz * nx * ny] + lam[ix + iy * nx + iz * nx * ny];
        }
    }


}

__global__ void compute_V(DAT* Prf, DAT* Txye, DAT* lam, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Vz_old, DAT* Qyft_old, DAT* Vzdif, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txz, DAT* Tyz, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exy, DAT* Exyc, DAT* coh, DAT* J2, DAT* J2c, DAT* cohc, DAT* Qxft, DAT* Qyft, DAT* Qxold, DAT* Qyold, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, const DAT fric, const DAT coh0, const DAT damp, const DAT dt, const DAT K, const DAT G0, DAT dt_rho, const DAT rho12, const DAT rho22, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz, const DAT Dnx, const DAT Dny, const DAT chi, const DAT eta_k1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
    //if (iz < nz && iy < ny && ix < nx) {
        //J2[ix + iy * nx + iz * nx * ny] = sqrt((Txx[ix + iy * nx + iz * nx * ny] * Txx[ix + iy * nx + iz * nx * ny] + Tyy[ix + iy * nx + iz * nx * ny] * Tyy[ix + iy * nx + iz * nx * ny] + Tzz[ix + iy * nx + iz * nx * ny] * Tzz[ix + iy * nx + iz * nx * ny]) * ((DAT)1.0 / (DAT)2.0) + Txy[ix + iy * nx + iz * nx * ny] * Txy[ix + iy * nx + iz * nx * ny] + Txz[ix + iy * nx + iz * nx * ny] * Txz[ix + iy * nx + iz * nx * ny] + Tyz[ix + iy * nx + iz * nx * ny] * Tyz[ix + iy * nx + iz * nx * ny]);
        //DAT F = J2[ix + iy * nx + iz * nx * ny] - coh[ix + iy * nx + iz * nx * ny];
        //if (F < (DAT)0.0) { F = (DAT)0.0; }
    //if (lam[ix + iy * nx + iz * nx * ny] > (DAT)0.0) {
        //if (F > (DAT)0.0) {
        if (iz < nz && iy>0 && iy < ny && ix>0 && ix < nx) {
            //Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] =  (  ((DAT)0.25 * lam[ix + (iy + 0) * (nx + 0) + iz * (nx + 0) * (ny + 0)] + (DAT)0.25 * lam[ix - 1 + (iy - 1) * (nx + 0) + iz * (nx + 0) * (ny + 0)] + (DAT)0.25 * lam[ix + 0 + (iy - 1) * (nx + 0) + iz * (nx + 0) * (ny + 0)] + lam[ix - 1 + (iy + 0) * (nx + 0) + iz * (nx + 0) * (ny + 0)] * ((DAT)1.0 / (DAT)4.0))    ) * Txyc[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)];//av4(lam)
        }
        if (iz > 0 && iz < nz && iy < ny && ix>0 && ix < nx) {
            //Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny] =  (  ((DAT)0.25 * lam[ix + (iy + 0) * (nx + 0) + iz * (nx + 0) * (ny + 0)] + (DAT)0.25 * lam[ix - 1 + (iy ) * (nx + 0) + (iz - 1) * (nx + 0) * (ny + 0)] + (DAT)0.25 * lam[ix + 0 + (iy ) * (nx + 0) + (iz - 1) * (nx + 0) * (ny + 0)] + lam[ix - 1 + (iy + 0) * (nx + 0) + iz * (nx + 0) * (ny + 0)] * ((DAT)1.0 / (DAT)4.0))) * Txzc[ix + iy * (nx + 1) + iz * (nx + 1) * ny];
        }
        if (iz > 0 && iz < nz && iy>0 && iy < ny && ix < nx) {
            //Tyzc[ix + iy * nx + iz * nx * (ny + 1)] =  (  ((DAT)0.25 * lam[ix + (iy + 0) * (nx + 0) + iz * (nx + 0) * (ny + 0)] + (DAT)0.25 * lam[ix  + (iy - 1) * (nx + 0) + (iz - 1) * (nx + 0) * (ny + 0)] + (DAT)0.25 * lam[ix + 0 + (iy - 1) * (nx + 0) + iz * (nx + 0) * (ny + 0)] + lam[ix  + (iy + 0) * (nx + 0) + (iz - 1) * (nx + 0) * (ny + 0)] * ((DAT)1.0 / (DAT)4.0))) * Tyzc[ix + iy * nx + iz * nx * (ny + 1)];
        }
        //                }
        if (iz > 0 && iz < (nz - 1) && iy>0 && iy < ny && ix>0 && ix < (nx - 1)) {
        //Vy_old[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)];
        //Qyft_old[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)];
        }
        if (iz > 0 && iz < nz && iy < (ny - 0) && ix < (nx - 0)) {
            Vz_old[ix + iy * (nx)+iz * (nx) * (ny)] = Vz[ix + iy * (nx)+iz * (nx) * (ny)]; //- 0.0*rhotg 
        }
    //}
#undef div_Sigmax
#undef div_Sigmay
#undef div_Sigmaz
}


__global__ void compute_V2(DAT* dt_rhom, DAT* QxoldINCR, DAT* QyoldINCR, DAT* QzoldINCR, DAT* k_etaf, DAT* Prf, DAT* Txye, DAT* lam, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Vz_old, DAT* Qyft_old, DAT* Vzdif,DAT* Qydif, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exy, DAT* Exyc, DAT* coh, DAT* J2, DAT* J2c, DAT* cohc, DAT* Qxft, DAT* Qyft, DAT* Qzft, DAT* Qxold, DAT* Qyold, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, const DAT fric, const DAT coh0, const DAT damp, const DAT dt, const DAT K, const DAT G0, DAT dt_rho, const DAT rho12, const DAT rho22, const DAT dx, const DAT dy, const DAT dz, DAT iM22,DAT rhof0g, DAT rhotg, const int nx, const int ny, const int nz, const DAT Dnx, const DAT Dny, const DAT chi, const DAT eta_k1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
#define div_Sigmax  ( ( Txx[ix+0 + iy*nx + iz*nx*ny]-Txx[(ix-1) + (iy  )*nx + (iz  )*nx*ny] - Pt[ix+0 + iy*nx + iz*nx*ny]+Pt[(ix-1) + (iy  )*nx + (iz  )*nx*ny])*((DAT)1.0/dx) + ( Txyc[ix   + (iy+1)*(nx+1) + iz*(nx+1)*(ny+1)] - Txyc[ix + (iy-0)*(nx+1) + iz*(nx+1)*(ny+1)])*((DAT)1.0/dy)    +   (Txzc[ix + (iy+0)*(nx+1) + (iz+1)* (nx+1)*(ny+0)] - Txzc[ix + iy*(nx+1) + iz*(nx+1)*(ny+0)])*((DAT)1.0/dz) )
#define div_Sigmay  ( ( Tyy[ix + iy*nx + iz*nx*ny]-Tyy[(ix  ) + (iy-1)*nx + (iz  )*nx*ny]   - Pt[ix + iy*nx + iz*nx*ny]+Pt[(ix  ) + (iy-1)*nx + (iz  )*nx*ny] )*((DAT)1.0/dy) +  ( Txyc[ix+1 + (iy  )*(nx+1) + iz*(nx+1)*(ny+1)] - Txyc[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)])*((DAT)1.0/dx)    +    (Tyzc[ix + (iy  )*(nx+0) + (iz+1)* (nx+0)*(ny+1)] - Tyzc[ix + iy*(nx+0) + iz*(nx+0)*(ny+1)])*((DAT)1.0/dz) )
#define div_Sigmaz  ( ( Tzz[ix + iy*nx + iz*nx*ny]-Tzz[(ix  ) + (iy  )*nx + (iz-1)*nx*ny]   - Pt[ix + iy*nx + iz*nx*ny]+Pt[(ix  ) + (iy  )*nx + (iz-1)*nx*ny] )*((DAT)1.0/dz) + ( Txzc[ix+1 + (iy  )*(nx+1) + iz*(nx+1)*(ny+0)] - Txzc[ix + iy*(nx+1) + iz*(nx+1)*(ny+0)])*((DAT)1.0/dx)     +   (Tyzc[ix + (iy+1)*(nx+0) +  iz   * (nx+0)*(ny+1)] - Tyzc[ix + iy*(nx+0) + iz*(nx+0)*(ny+1)])*((DAT)1.0/dy) )
//#define Q_gradPrfx (  (Prf[ix + iy*nx]-Prf[ix-1 +  iy   *nx]  )*((DAT)1.0/dx) + ((DAT)1.0-chi )*( Qxold[ix + iy*(nx+1)] )*eta_k1   )
//#define Q_gradPrfy (  (Prf[ix + iy*nx]-Prf[ix   + (iy-1)*nx]  )*((DAT)1.0/dy) + ((DAT)1.0-chi )*( Qyold[ix + iy*(nx)]   )*eta_k1   )
#define   dt_rhom_avx ( (DAT)0.5 * (dt_rhom[ix - 1 + (iy) * (nx + 0) + iz * nx * ny] + dt_rhom[ix + iy * (nx + 0) + iz * nx * ny]) )
#define   dt_rhom_avy ( (DAT)0.5 * (dt_rhom[ix + (iy - 1) * (nx)+iz * nx * (ny + 0)] + dt_rhom[ix + iy * (nx)+iz * nx * (ny + 0)]) )
#define   dt_rhom_avz ( (DAT)0.5 * (dt_rhom[ix + (iy - 0) * (nx)+(iz - 1) * nx * (ny + 0)] + dt_rhom[ix + iy * (nx)+iz * nx * (ny + 0)]) )

    //if (iz > 0 && iz < (nz - 1) && iy > 0 && iy < ny - 1 && ix>0 && ix < nx) {
        //DAT k_etaf_avx = (DAT)0.5 * (k_etaf[ix - 1 + (iy) * (nx + 0) + iz * nx * ny] + k_etaf[ix + iy * (nx + 0) + iz * nx * ny]);;
        //Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = (Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] * ((DAT)1.0 / dt_rhof / rhof) - 1.0 / rhof * Q_gradPrfx) * ((DAT)1.0 / (((DAT)1.0 / dt_rhof / rhof) + chi * 1.0 / rhof / k_etaf_avx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] ));/// k_etaf_avx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)]
        //Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + (-Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] - k_etaf_avx * (Prf[ix + iy * nx + iz * nx * ny] - Prf[ix - 1 + iy * nx + iz * nx * ny]) * ((DAT)1.0 / dx)) / ((DAT)1.0 + k_etaf_avx  / dt_rho);
        //Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + (-Qxft[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] +(QxoldINCR[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] / dt - iM22 *  (Prf[ix + iy * nx + iz * nx * ny] - Prf[ix - 1 + iy * nx + iz * nx * ny]) * ((DAT)1.0 / dx)) / (1.0 / dt + iM22 / k_etaf_avx)) / ((DAT)1.0 + k_etaf_avx / dt_rho);

    //}
   // if (iz > 0 && iz < (nz - 1) && iy > 0 && iy < ny && ix>0 && ix < nx - 1) {
        //DAT k_etaf_avy = (DAT)0.5 * (k_etaf[ix + (iy - 1) * (nx)+iz * nx * (ny + 0)] + k_etaf[ix + iy * (nx)+iz * nx * (ny + 0)]);
        //Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] = (Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] * ((DAT)1.0 / dt_rhof / rhof) - 1.0 / rhof * Q_gradPrfy) * ((DAT)1.0 / (((DAT)1.0 / dt_rhof / rhof) + chi * 1.0 / rhof / k_etaf_avy[ix + iy * (nx)+iz * (nx) * (ny + 1)]));/// * eta_k1 k_etaf_avy[ix + iy * (nx)+iz * (nx) * (ny + 1)]
        //Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] + (-Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] - k_etaf_avy  * (Prf[ix + iy * nx + iz * nx * ny] - Prf[ix + (iy - 1) * nx + iz * nx * ny]) * ((DAT)1.0 / dy)) / ((DAT)1.0 + k_etaf_avy  / dt_rho);
        //Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] + (-Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] +(QyoldINCR[ix + iy * (nx)+iz * (nx) * (ny + 1)] / dt - iM22 * (Prf[ix + iy * nx + iz * nx * ny] - Prf[ix + (iy - 1) * nx + iz * nx * ny]) * ((DAT)1.0 / dy)) / (1.0 / dt + iM22 / k_etaf_avy)) / ((DAT)1.0 + k_etaf_avy / dt_rho);
        //Qydif[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Qyft[ix + iy * (nx)+iz * (nx) * (ny + 1)] - Qyft_old[ix + iy * (nx)+iz * (nx) * (ny + 1)];
   // }
   // if (iz > 0 && iz < nz && iy>0 && iy < (ny - 1) && ix>0 && ix < (nx - 1)) {
        //DAT k_etaf_avz = (DAT)0.5 * (k_etaf[ix + (iy - 0) * (nx)+(iz - 1) * nx * (ny + 0)] + k_etaf[ix + iy * (nx)+iz * nx * (ny + 0)]);
        //Qzft[ix + iy * (nx)+iz * (nx) * (ny)] = (Qzft[ix + iy * (nx)+iz * (nx) * (ny)] * ((DAT)1.0 / dt_rhof / rhof) - (DAT)1.0 / rhof * Q_gradPrfz) * ((DAT)1.0 / (((DAT)1.0 / dt_rhof / rhof) + chi * 1.0 / rhof / k_etaf_avz[ix + iy * (nx)+iz * (nx) * (ny)] )); //k_etaf_avz[ix + iy * (nx)+iz * (nx) * (ny)]
        //Qzft[ix + iy * (nx)+iz * (nx) * (ny)] = Qzft[ix + iy * (nx)+iz * (nx) * (ny)] + (-Qzft[ix + iy * (nx)+iz * (nx) * (ny)] - k_etaf_avz  * (Prf[ix + iy * nx + iz * nx * ny] - Prf[ix + iy * nx + (iz - 1) * nx * ny]) * ((DAT)1.0 / dz) ) / ((DAT)1.0 + k_etaf_avz  / dt_rho);//+ rhof0g
        //Qzft[ix + iy * (nx)+iz * (nx) * (ny)] = Qzft[ix + iy * (nx)+iz * (nx) * (ny)] + (-Qzft[ix + iy * (nx)+iz * (nx) * (ny)] +(QzoldINCR[ix + iy * (nx)+iz * (nx) * (ny)] / dt - iM22 *( (Prf[ix + iy * nx + iz * nx * ny] - Prf[ix + iy * nx + (iz - 1) * nx * ny]) * ((DAT)1.0 / dz) +  rhof0g)) / (1.0 / dt + iM22 / k_etaf_avz)) / ((DAT)1.0 + k_etaf_avz / dt_rho);//+ rhof0g

   // }

    //if (iy<ny && ix>0 && ix<nx){
    //    Qxft[ix + iy*(nx+1)] = ( Qxold[ix + iy*(nx+1)]*((DAT)1.0/dt) - rho22* Q_gradPrfx - rho12* div_Sigmax)* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*rho22*eta_k1 )  ) *(DAT)1.0/((DAT)1.0 + damp/Dnx ); }
    //if (iy>0 && iy<ny && ix<nx){
    //    Qyft[ix + iy*(nx  )] = ( Qyold[ix + iy*(nx  )]*((DAT)1.0/dt) - rho22* Q_gradPrfy - rho12* div_Sigmay)* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*rho22*eta_k1 )  ) *(DAT)1.0/((DAT)1.0 + damp/Dny ); }    

    if (iz < (nz - 0) && iy < (ny - 0) && ix>0 && ix < nx) {
        Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + dt_rhom_avx * div_Sigmax  ; //
        //Ux[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Ux[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] * dt;
    }
    if ( iz < (nz - 0) && iy>0 && iy < ny && ix < (nx - 0)) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] + dt_rhom_avy * div_Sigmay  ;//
        //Vydif[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] - Vy_old[ix + iy * (nx)+iz * (nx) * (ny + 1)];
        //Uy[ix + iy * (nx)+iz * (nx) * (ny + 1)] =    Uy[ix + iy * (nx)+iz * (nx) * (ny + 1)] + Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)]*dt;
    }
    if (iz > 0 && iz < nz && iy < (ny - 0) && ix < (nx - 0)) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = Vz[ix + iy * (nx)+iz * (nx) * (ny)] + dt_rhom_avz * (div_Sigmaz )  ; //- 0.0*rhotg 
        Vzdif[ix + iy * (nx)+iz * (nx) * (ny)] = Vz[ix + iy * (nx)+iz * (nx) * (ny)] - Vz_old[ix + iy * (nx)+iz * (nx) * (ny)];
    }
#undef div_Sigmax
#undef div_Sigmay
#undef div_Sigmaz
}
__global__ void compute_V2sh(DAT* dt_rhom, DAT* QxoldINCR, DAT* QyoldINCR, DAT* QzoldINCR, DAT* k_etaf, DAT* Prf, DAT* Txye, DAT* lam, DAT* lama, DAT* Vx, DAT* Vy, DAT* Vz, DAT* Vz_old, DAT* Qyft_old, DAT* Vzdif, DAT* Qydif, DAT* Ux, DAT* Uy, DAT* Pt, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txyc, DAT* Txzc, DAT* Tyzc, DAT* Pt_old, DAT* Txx_old, DAT* Tyy_old, DAT* Tzz_old, DAT* Txy_old, DAT* Txyc_old, DAT* Exy, DAT* Exyc, DAT* coh, DAT* J2, DAT* J2c, DAT* cohc, DAT* Qxft, DAT* Qyft, DAT* Qzft, DAT* Qxold, DAT* Qyold, DAT scale, DAT dStr0_1, DAT dStr0_2, DAT dStr0_3, const DAT fric, const DAT coh0, const DAT damp, const DAT dt, const DAT K, const DAT G0, DAT dt_rho, const DAT rho12, const DAT rho22, const DAT dx, const DAT dy, const DAT dz, DAT iM22, DAT rhof0g, DAT rhotg, const int nx, const int ny, const int nz, const DAT Dnx, const DAT Dny, const DAT chi, const DAT eta_k1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // thread ID, dimension z
#define div_Sigmax  ( ( Txx[ix+0 + iy*nx + iz*nx*ny]-Txx[(ix-1) + (iy  )*nx + (iz  )*nx*ny] - Pt[ix+0 + iy*nx + iz*nx*ny]+Pt[(ix-1) + (iy  )*nx + (iz  )*nx*ny])*((DAT)1.0/dx) + ( Txyc[ix   + (iy+1)*(nx+1) + iz*(nx+1)*(ny+1)] - Txyc[ix + (iy-0)*(nx+1) + iz*(nx+1)*(ny+1)])*((DAT)1.0/dy)    +   (Txzc[ix + (iy+0)*(nx+1) + (iz+1)* (nx+1)*(ny+0)] - Txzc[ix + iy*(nx+1) + iz*(nx+1)*(ny+0)])*((DAT)1.0/dz) )
#define div_Sigmay  ( ( Tyy[ix + iy*nx + iz*nx*ny]-Tyy[(ix  ) + (iy-1)*nx + (iz  )*nx*ny]   - Pt[ix + iy*nx + iz*nx*ny]+Pt[(ix  ) + (iy-1)*nx + (iz  )*nx*ny] )*((DAT)1.0/dy) +  ( Txyc[ix+1 + (iy  )*(nx+1) + iz*(nx+1)*(ny+1)] - Txyc[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)])*((DAT)1.0/dx)    +    (Tyzc[ix + (iy  )*(nx+0) + (iz+1)* (nx+0)*(ny+1)] - Tyzc[ix + iy*(nx+0) + iz*(nx+0)*(ny+1)])*((DAT)1.0/dz) )
#define div_Sigmaz  ( ( Tzz[ix + iy*nx + iz*nx*ny]-Tzz[(ix  ) + (iy  )*nx + (iz-1)*nx*ny]   - Pt[ix + iy*nx + iz*nx*ny]+Pt[(ix  ) + (iy  )*nx + (iz-1)*nx*ny] )*((DAT)1.0/dz) + ( Txzc[ix+1 + (iy  )*(nx+1) + iz*(nx+1)*(ny+0)] - Txzc[ix + iy*(nx+1) + iz*(nx+1)*(ny+0)])*((DAT)1.0/dx)     +   (Tyzc[ix + (iy+1)*(nx+0) +  iz   * (nx+0)*(ny+1)] - Tyzc[ix + iy*(nx+0) + iz*(nx+0)*(ny+1)])*((DAT)1.0/dy) )

#define   dt_rhom_avx ( (DAT)0.5 * (dt_rhom[ix - 1 + (iy) * (nx + 0) + iz * nx * ny] + dt_rhom[ix + iy * (nx + 0) + iz * nx * ny]) )
#define   dt_rhom_avy ( (DAT)0.5 * (dt_rhom[ix + (iy - 1) * (nx)+iz * nx * (ny + 0)] + dt_rhom[ix + iy * (nx)+iz * nx * (ny + 0)]) )
#define   dt_rhom_avz ( (DAT)0.5 * (dt_rhom[ix + (iy - 0) * (nx)+(iz - 1) * nx * (ny + 0)] + dt_rhom[ix + iy * (nx)+iz * nx * (ny + 0)]) )

    if (iz > 0 && iz < (nz - 1) && iy > 0 && iy < ny - 1 && ix>0 && ix < nx) {
        Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] = Vx[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] + dt_rhom_avx * div_Sigmax;
    }

    if (iz > 0 && iz < (nz - 1) && iy > 0 && iy < ny && ix>0 && ix < nx - 1) {
        Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] + dt_rhom_avy * div_Sigmay;
        //Vydif[ix + iy * (nx)+iz * (nx) * (ny + 1)] = Vy[ix + iy * (nx)+iz * (nx) * (ny + 1)] - Vy_old[ix + iy * (nx)+iz * (nx) * (ny + 1)];
    }

    if (iz > 0 && iz < nz && iy>0 && iy < (ny - 1) && ix>0 && ix < (nx - 1)) {
        Vz[ix + iy * (nx)+iz * (nx) * (ny)] = Vz[ix + iy * (nx)+iz * (nx) * (ny)] + dt_rhom_avz * (div_Sigmaz);
    }
#undef div_Sigmax
#undef div_Sigmay
#undef div_Sigmaz
}

////////// ========================================  MAIN  ======================================== //////////
int main(){
    size_t i, N;
    int it;
    N = nx*ny*nz; DAT mem = (DAT)1e-9*(DAT)N*sizeof(DAT);
    set_up_gpu();

    load(pa1, 19, 1, "pa1.dat");
    DAT dt = pa1_h[0], rhof0g = pa1_h[1], rhotg = pa1_h[2],  K = pa1_h[4], B = pa1_h[5], alph = pa1_h[6], G0 = pa1_h[7], eta = pa1_h[8], M = pa1_h[9], iM22 = pa1_h[13];//
    int nt = pa1_h[14];
    DAT eta_ve = pa1_h[16];
    int TestN = pa1_h[17];
    DAT K_G = pa1_h[18];
    DAT G = G0;
    const DAT coh0 = 1e-2 * G0;                 // cohesion, Pa

    if ( (TestN == 1) ) {
        printf("\n  -------------------------------------------------------------------------- ");
        printf("\n  | FastCijkl_GPU3D_v1:  Effective elastic properties of composites  | ");
        printf("\n  --------------------------------------------------------------------------  \n\n");
        printf("Local size: %dx%dx%d (%1.4f GB) %d iterations ...\n", nx, ny, nz, mem * 20.0, niter);
        printf("Launching (%dx%dx%d) grid of (%dx%dx%d) blocks.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    }
    printf("\n-----------------------------------");
    printf("\nTest: %d \n", TestN);
    printf("-----------------------------------\n"); fflush(stdout);
    const DAT chi    = (DAT)0.5;
    // Initial arrays
    int int_gridx = (int)grid.x;
    int int_gridy = (int)grid.y;
    int int_gridz = (int)grid.z;
    zeros(__device_maxval, int_gridx, int_gridy, int_gridz);
    zeros(x        ,8  ,8  ,8  );
    zeros(y        ,8  ,8  ,8  );
    zeros(z        ,8  ,8  ,8  );
    zeros(Txy, 8 + 1, 8, 8);
    zeros(Txz, 8 + 1, 8, 8);
    zeros(Tyz, 8 + 1, 8, 8);
    zeros(Txyc,nx+1,ny+1,nz  );
    zeros(Txzc,nx+1,ny  ,nz+1);
    zeros(Tyzc,nx  ,ny+1,nz+1);
    zeros(Prf_old, 8, 8, 8);
    zeros(Qxft     , 8 + 1, 8, 8);
    zeros(Qyft     , 8 + 1, 8, 8);
    zeros(Qzft     , 8 + 1, 8, 8);
    zeros(Qxold    ,8+1,8  ,8  );
    zeros(Qyold    ,8  ,8+1,8  );
    zeros(Qzold    ,8  ,8,8+1  );
    zeros(QxoldINCR, 8 + 1, 8, 8);
    zeros(QyoldINCR, 8 + 1, 8, 8);
    zeros(QzoldINCR, 8 + 1, 8, 8);
    zeros(k_etaf, 8, 8, 8);
    int xsrc = round(0.78 * x_s / dx); 
    int ysrc = round(y_s / dy); 
    int zsrc = round(z_s / dz); 
    int xsrc2 = round(1.07 *x_s / dx);
    int ysrc2 = round( 1.0* y_s / dy);
    int zsrc2 = round(1.1*z_s / dz);  
    zeros(rad, 8 + 1, 8 + 1, 8 + 1);
    zeros(radc, 8 + 1, 8 + 1, 8 + 1);
    zeros(lama, 8, 8, 8);
    zeros(lam, 8, 8, 8);
    zeros(Txye, 8, 8, 8);
    zeros(Pt_old, 8, 8, 8);

    zeros(Vy_old, 8, 8, 8);
    zeros(Vydif, 8, 8, 8);
    zeros(Vz_old, nx, ny , nz + 1);
    zeros(Vzdif, nx, ny , nz + 1);
    zeros(Qyft_old, 8, 8, 8);
    zeros(Qydif, 8, 8, 8);
    zeros(Ux, 8 + 1, 8, 8);
    zeros(Uy, 8 + 1, 8, 8);
    zeros(Uz, 8 + 1, 8, 8);
    zeros(Exy, 8, 8, 8);
    zeros(Exyc, 8, 8, 8);
    zeros(coh, 8, 8, 8);
    zeros(J2, 8, 8, 8);
    zeros(J2c, 8 + 1, 8 + 1, 8+1);
    zeros(cohc, 8 + 1, 8 + 1, 8+1);

    zeros(Vfx, 8, 8, 8);
    zeros(Vfy, 8, 8, 8);
    zeros(Vfz, 8, 8, 8);

    zeros(divWxy, 8, 8, 8);
    zeros(divWxz, 8, 8, 8);
    zeros(divWyz, 8, 8, 8);
    zeros(Tdiffxy, 8, 8, 8);
    zeros(Tdiffxz, 8, 8, 8);
    zeros(Tdiffyz, 8, 8, 8);
    zeros(advPt_old, 8, 8, 8);
    zeros(advPrf_old, 8, 8, 8);
    zeros(advTxx_old, 8, 8, 8);
    zeros(advTyy_old, 8, 8, 8);
    zeros(advTzz_old, 8, 8, 8);
    zeros(advTxyc_old, 8, 8, 8);
    zeros(advTxzc_old, 8, 8, 8);
    zeros(advTyzc_old, 8, 8, 8);
    zeros(Grm, nx, ny, nz); 
    zeros(Krm, nx, ny, nz);
    zeros(Dis, nx, ny, nz);

        zeros(divVsaveExx, nx, ny, nz);
        zeros(divVsaveEyy, nx, ny, nz);
        zeros(divVsaveEzz, nx, ny, nz);

        zeros(divVsaveExy, nx + 1, ny + 1, nz);
        zeros(divVsaveExz, nx + 1, ny, nz + 1);
        zeros(divVsaveEyz, nx, ny + 1, nz + 1);

        zeros(J2U, 8, 8, 8);

        zeros(Pt, nx, ny, nz);
        zeros(Prf, 8, 8, 8);
        zeros(coh0m, 8, 8, 8);
        zeros(Txx, nx, ny, nz);
        zeros(Tyy, nx, ny, nz);
        zeros(Tzz, nx, ny, nz);
         
          zeros(Gdtm_etam, nx, ny, nz);
          zeros(dt_rhom, nx, ny, nz);
          zeros(Gdtm, nx, ny, nz);
          zeros(Kdtm, nx, ny, nz); 
          zeros(Ux_loop, nx+1, ny, nz);
          //def_sizes(Ux_loop, nx + 1, ny, nz );

        zeros(INJ, 1000, 1, 1);
    load3(Gm, nx, ny, nz, "Gm.dat");
    load3(Km, nx, ny, nz, "Km.dat");
    load3(etam, nx, ny, nz, "etam.dat");
    load3(eta_vem, nx, ny, nz, "eta_vem.dat");
    load3(Vx, nx+1, ny, nz, "Vx.dat");
    load3(Vy, nx, ny+1, nz, "Vy.dat");
    load3(Vz, nx, ny , nz + 1, "Vz.dat");

    zeros(Txx_old, 8, 8, 8);
    zeros(Tyy_old, 8, 8, 8);
    zeros(Tzz_old, 8, 8, 8);
    zeros(Txy_old, 8, 8, 8);
    zeros(Txyc_old, 8, 8, 8);
    zeros(Txz_old, 8, 8, 8);
    zeros(Txzc_old, 8, 8, 8);
    zeros(Tyz_old, 8, 8, 8);
    zeros(Tyzc_old, 8, 8, 8);

    DAT device_MAX = (DAT)0.0;
    DAT err_Vydif_MAX = (DAT)0.0;

    int  isave = 0;
    DAT resid = 0.0;

    DAT Gdt = 0.0;
    DAT Gr = 0.0;
    DAT Kdt = 0.0;
    DAT Kr = 0.0;
    DAT dt_rho = 0.0;
    DAT Vpdt = CFL * dz;
    DAT Re = 2.0 * sqrt(3.0) / 1.0 * PI;

    zeros_h(residH, 3, 1, 1);
    zeros_h(residHxx, 3, 1, 1);
    zeros_h(residHyy, 3, 1, 1);
    zeros_h(residHzz, 3, 1, 1);

    // Initial condition
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);
    cudaEventRecord(startD);

    if ((TestN == 4) || (TestN == 5) || (TestN == 6)) {
        epsi = 9.0 * 1e-7;//1e-7;
    }

    for (it=0;it<nt;it++){
        cudaDeviceSynchronize();

        jaumannD << <grid, block >> > (etam_d, Gdtm_etam_d, eta_vem_d,  dt_rhom_d,  Gdtm_d,  Kdtm_d, Km_d,  Krm_d, Gm_d, Grm_d, divWxy_d, divWxz_d, divWyz_d, Tdiffxy_d, Tdiffxz_d, Tdiffyz_d, advPt_old_d, advTxx_old_d, advTyy_old_d, advTzz_old_d, advTxyc_old_d, advTxzc_old_d, advTyzc_old_d, Txye_d, x_d, y_d, rad_d, radc_d, lama_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Pt_d, Txx_d, Tyy_d, Txy_d, Txz_d, Tyz_d, Txyc_d, Txzc_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txzc_old_d, Tyzc_old_d, Exyc_d, scale, dStr0_1, dStr0_2, dStr0_3, divV0, eta_ve, eta, G, dt, dt_rho, Vpdt, Lx, Re, K, G0, rho11, dx, dy, dz, nx, ny, nz, rad0, Gdt, Kdt, K_G);
        cudaDeviceSynchronize();

        iter = 0; resid = (DAT)2.0 * epsi;//
        cudaDeviceSynchronize();
        while (iter < niter && resid>epsi) {//
            //compute_BC3v2 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
            cudaDeviceSynchronize();
            if (TestN == 4) {
                compute_BC3 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
                cudaDeviceSynchronize();
                compute_BC4 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
                cudaDeviceSynchronize();
            }
            if (TestN == 5) {
                compute_BC5 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
                cudaDeviceSynchronize();
                compute_BC6 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
                cudaDeviceSynchronize();
            }
            if (TestN == 6) {
                compute_BC7 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
                cudaDeviceSynchronize();
                compute_BC8 << <grid, block >> > (Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz);
                cudaDeviceSynchronize();
            }
            compute_Stress << <grid, block >> > (Gdtm_etam_d, eta_vem_d, dt_rhom_d, Gdtm_d, Kdtm_d, Krm_d, Grm_d, Prf_d, Prf_old_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, Qzft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, alph, M, B, nx, ny, nz);
            cudaDeviceSynchronize();

                compute_V << <grid, block >> > (Prf_d, Txye_d, lam_d, lama_d, Vx_d, Vy_d, Vz_d, Vz_old_d, Qyft_old_d, Vzdif_d,  Ux_d, Uy_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txz_d, Tyz_d, Txyc_d, Txzc_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Exy_d, Exyc_d, coh_d, J2_d, J2c_d, cohc_d, Qxft_d, Qyft_d, Qxold_d, Qyold_d, scale, dStr0_1, dStr0_2, dStr0_3, fric, coh0, damp, dt, K, G0, dt_rho, rho12, rho22, dx, dy, dz, nx, ny, nz, Dnx, Dny, chi, eta_k1);
                cudaDeviceSynchronize();

            if ((TestN == 1) || (TestN == 2) || (TestN == 3) || (TestN == 7)){
                compute_V2 << <grid, block >> > (dt_rhom_d, QxoldINCR_d, QyoldINCR_d, QzoldINCR_d, k_etaf_d, Prf_d, Txye_d, lam_d, lama_d, Vx_d, Vy_d, Vz_d, Vz_old_d, Qyft_old_d, Vzdif_d, Qydif_d, Ux_d, Uy_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txzc_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Exy_d, Exyc_d, coh_d, J2_d, J2c_d, cohc_d, Qxft_d, Qyft_d, Qzft_d, Qxold_d, Qyold_d, scale, dStr0_1, dStr0_2, dStr0_3, fric, coh0, damp, dt, K, G0, dt_rho, rho12, rho22, dx, dy, dz, iM22, rhof0g, rhotg, nx, ny, nz, Dnx, Dny, chi, eta_k1);
                cudaDeviceSynchronize();
            }
            if ((TestN == 4) || (TestN == 5) || (TestN == 6)) {
                compute_V2sh << <grid, block >> > (dt_rhom_d, QxoldINCR_d, QyoldINCR_d, QzoldINCR_d, k_etaf_d, Prf_d, Txye_d, lam_d, lama_d, Vx_d, Vy_d, Vz_d, Vz_old_d, Qyft_old_d, Vzdif_d, Qydif_d, Ux_d, Uy_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txzc_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Exy_d, Exyc_d, coh_d, J2_d, J2c_d, cohc_d, Qxft_d, Qyft_d, Qzft_d, Qxold_d, Qyold_d, scale, dStr0_1, dStr0_2, dStr0_3, fric, coh0, damp, dt, K, G0, dt_rho, rho12, rho22, dx, dy, dz, iM22, rhof0g, rhotg, nx, ny, nz, Dnx, Dny, chi, eta_k1);
                cudaDeviceSynchronize();
            }
            if ((iter % nout) == 1 && (iter > (8*nz))) {
                __MPI_max(Vzdif);   err_Vydif_MAX = device_MAX;
                //__MPI_max(Vy);   err_Vy_MAX = device_MAX;
                resid = (err_Vydif_MAX/( (DAT)1.0) );
            }
            iter = iter + 1;
            cudaDeviceSynchronize();
        }//iter
        if (!(resid > 0 || resid == 0 || resid < 0)) { printf("\n !! ERROR: resid=Nan, break (it=%d, iter=%d) !! \n\n", (it + 1), (iter + 1)); break; }
        cudaEventRecord(stopD);
        cudaEventSynchronize(stopD);
        if ((it > -1)) {
            printf("> it=%05d > iter=%05d  , ||resid||=%6.2e, ||Vzdif_MAX||=%3.3e\n", it, iter, resid, err_Vydif_MAX); fflush(stdout);
        }
        if ((it % nsave) == 0) {
            compute_SaveFields << <grid, block >> > (Ux_loop_d, advTxyc_old_d, advTxzc_old_d, advTyzc_old_d, J2U_d, divVsaveExy_d, divVsaveExz_d, divVsaveEyz_d, divVsaveExx_d, divVsaveEyy_d, divVsaveEzz_d, Grm_d, Prf_d, Vx_d, Vy_d, Vz_d, Ux_d, Uy_d, Uz_d, Pt_d, Txx_d, Tyy_d, Tzz_d, Txy_d, Txyc_d, Txz_d, Txzc_d, Tyz_d, Tyzc_d, Pt_old_d, Txx_old_d, Tyy_old_d, Tzz_old_d, Txy_old_d, Txyc_old_d, Txz_old_d, Txzc_old_d, Tyz_old_d, Tyzc_old_d, Exyc_d, Qxft_d, Qyft_d, scale, dStr_1, dStr_2, dStr_3, Kdt, Gdt, Kr, Gr, eta, rho11, dx, dy, dz, alpha1, M1, nx, ny, nz, dt);
            cudaDeviceSynchronize();

            if (TestN == 3) {
                gather(divVsaveEzz); gather(Pt); gather(Tzz);
                compute_sum(residHzz_h, divVsaveEzz_h, Pt_h, Tzz_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residHzz_h, 3, "residH", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residHzz_h[i] = (DAT)0.0; }
            }

            if (TestN == 1) {
                gather(divVsaveExx); gather(Pt); gather(Txx);
                compute_sum11(residHxx_h, divVsaveExx_h, Pt_h, Txx_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residHxx_h, 3, "residH", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residHxx_h[i] = (DAT)0.0; }
            }

            if (TestN == 2) {
                gather(divVsaveEyy); gather(Pt); gather(Tyy);
                compute_sum22(residHyy_h, divVsaveEyy_h, Pt_h, Tyy_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residHyy_h, 3, "residH", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residHyy_h[i] = (DAT)0.0; }
            }

            if (TestN == 7) {
                gather(divVsaveExx); gather(Pt); gather(Txx); gather(divVsaveEyy); gather(Tyy); gather(divVsaveEzz);  gather(Tzz);
                compute_sum7(residHxx_h, residHyy_h, residHzz_h, divVsaveExx_h, Pt_h, Txx_h, divVsaveEyy_h,   Tyy_h, divVsaveEzz_h,  Tzz_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residHxx_h, 3, "residHxx", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residHxx_h[i] = (DAT)0.0; }
                save_array(residHyy_h, 3, "residHyy", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residHyy_h[i] = (DAT)0.0; }
                save_array(residHzz_h, 3, "residHzz", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residHzz_h[i] = (DAT)0.0; }
            }

            if (TestN == 4) {
                gather(divVsaveExy);  gather(Txyc);
                compute_sum4(residH_h, divVsaveExy_h,  Txyc_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residH_h, 3, "residH", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residH_h[i] = (DAT)0.0; }
            }

            if (TestN == 5) {
                gather(divVsaveExz);  gather(Txzc);
                compute_sum5(residH_h, divVsaveExz_h, Txzc_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residH_h, 3, "residH", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residH_h[i] = (DAT)0.0; }
            }

            if (TestN == 6) {
                gather(divVsaveEyz);  gather(Tyzc);
                compute_sum6(residH_h, divVsaveEyz_h, Tyzc_h, nx, ny, nz);
                cudaDeviceSynchronize();
                save_array(residH_h, 3, "residH", isave);
                for (i = 0; i < (3) * (1) * (1); i++) { residH_h[i] = (DAT)0.0; }
            }


            if ((TestN == 1) || (TestN == 2) || (TestN == 3) || (TestN == 7)) {
                //SaveArray(divVsaveExx, "divVsaveExx")
                    //SaveArray(divVsaveEyy, "divVsaveEyy")
                    //SaveArray(divVsaveEzz, "divVsaveEzz")

                    //SaveArray(Pt, "Pt");
                //SaveArray(Txx, "Txx");
                //SaveArray(Tzz, "Tzz");
                //SaveArray(Ux_loop, "Ux_loop");
                //SaveArray(Dis, "Dis");
                //SaveArray(Tyy, "Tyy");
                //printf("> DynamicX > iter=%05d  , ||resid||=%6.2e     ", iter, resid); fflush(stdout);
            }
            if ((TestN == 4) || (TestN == 5) || (TestN == 6)) {
               // SaveArray(divVsaveExy, "divVsaveExy")
                //    SaveArray(divVsaveExz, "divVsaveExz")
                //    SaveArray(divVsaveEyz, "divVsaveEyz")
                //    SaveArray(Txyc, "Txyc");
                //SaveArray(Txzc, "Txzc");
                //SaveArray(Tyzc, "Tyzc");
            }
                isave = isave + 1;
        }
    }

    milliseconds = 0;
    size_t N2 = nx * ny * nz;
    cudaEventElapsedTime(&milliseconds, startD, stopD);
    GPUinfo[0] = 1E-3 * milliseconds;
    GPUinfo[1] = iter / GPUinfo[0];
    GPUinfo[2] = (DAT)N2 * (iter) * (DAT)9 * PRECIS / ((DAT)1e9);
    printf("\nGPU summary: MTPeff = %.2f [GB/s]", GPUinfo[2]);
    printf("\n  time is %.2f s after %d iterations, i.e., %.2f it/s\n", GPUinfo[0], iter, GPUinfo[1]); fflush(stdout);
    ///////////================================================================================ POSTPROCESS ====////
    save_info();  // Save simulation infos and coords (.inf files)
    // clear host memory & clear device memory
    free_all(x);
    free_all(y);
    free_all(z);
    free_all(Vx);
    free_all(Vy);
    free_all(Vz);
    free_all(Pt);
    free_all(Prf);
    free_all(Qxft);
    free_all(Qyft);
    free_all(Qzft);
    free_all(Qxold);
    free_all(Qyold);
    free_all(Qzold);
    clean_cuda();
    return 0;
}

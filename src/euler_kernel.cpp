#include <hip/hip_runtime.h>

__global__ void __launch_bounds__(384)
euler_i_kernel_central_hip_(int nv, int nx, int ny, int nz, int ng, int istart, int iend, int endi, int endj, int endk, int lmax,
                            double* fhat_trans_gpu, double* temperature_trans_gpu, double* fl_trans_gpu,
                            double* dcoe_gpu, double* dcsidx_gpu, double* wv_trans_gpu)
{
    int j = threadIdx.x + blockIdx.x*blockDim.x;
    int k = threadIdx.y + blockIdx.y*blockDim.y;

    if(j >= endj || k >= endk) 
        return;
    int NY_ = ny+2*ng;
    int NX_ = nx+2*ng;
    int NZ_ = nz+2*ng;
    
    for(int i = istart; i <= iend; i++){
        double ft1 = 0.0;
        double ft2 = 0.0;
        double ft3 = 0.0;
        double ft4 = 0.0;
        double ft5 = 0.0;
        double ft6 = 0.0;
        for(int l = 1; l <= lmax; l++){
            double uvs1 = 0.0;
            double uvs2 = 0.0;
            double uvs3 = 0.0;
            double uvs4 = 0.0;
            double uvs5 = 0.0;
            double uvs6 = 0.0; 
            for(int m = 0; m <= (l-1); m++){
                int j_i_m_k   = (j+ng) + (i-1-m+ng)*  NY_ + (k+ng)*NY_*NX_;
                int j_i_m_l_k = (j+ng) + (i-1-m+l+ng)*NY_ + (k+ng)*NY_*NX_;

                double rhom  =  wv_trans_gpu[j_i_m_k + NY_*NX_*NZ_*0] + wv_trans_gpu[j_i_m_l_k + NY_*NX_*NZ_*0];

                double uui   = wv_trans_gpu[j_i_m_k + NY_*NX_*NZ_*1];
                double vvi   = wv_trans_gpu[j_i_m_k + NY_*NX_*NZ_*2];
                double wwi   = wv_trans_gpu[j_i_m_k + NY_*NX_*NZ_*3];
                double ppi   = wv_trans_gpu[j_i_m_k + NY_*NX_*NZ_*0]*temperature_trans_gpu[j_i_m_k];
                double enti  = wv_trans_gpu[j_i_m_k + NY_*NX_*NZ_*4];

                double uuip   = wv_trans_gpu[j_i_m_l_k + NY_*NX_*NZ_*1];
                double vvip   = wv_trans_gpu[j_i_m_l_k + NY_*NX_*NZ_*2];
                double wwip   = wv_trans_gpu[j_i_m_l_k + NY_*NX_*NZ_*3];
                double ppip   = wv_trans_gpu[j_i_m_l_k + NY_*NX_*NZ_*0]*temperature_trans_gpu[j_i_m_l_k];
                double entip  = wv_trans_gpu[j_i_m_l_k + NY_*NX_*NZ_*4];

                double uv_part = (uui+uuip) * rhom;
                uvs1 = uvs1 + uv_part * (2.0);
                uvs2 = uvs2 + uv_part * (uui+uuip);
                uvs3 = uvs3 + uv_part * (vvi+vvip);
                uvs4 = uvs4 + uv_part * (wwi+wwip);
                uvs5 = uvs5 + uv_part * (enti+entip);
                uvs6 = uvs6 + (2.0)*(ppi+ppip);
            } //m
            double dcoe_gpu_local = dcoe_gpu[l-1 + (lmax-1)*4];
            ft1 = ft1 + dcoe_gpu_local*uvs1;
            ft2 = ft2 + dcoe_gpu_local*uvs2;
            ft3 = ft3 + dcoe_gpu_local*uvs3;
            ft4 = ft4 + dcoe_gpu_local*uvs4;
            ft5 = ft5 + dcoe_gpu_local*uvs5;
            ft6 = ft6 + dcoe_gpu_local*uvs6;
        } //l
        int j_i_k = (j+ng) + (i-1+ng)*NY_ + (k+ng)*NY_*NX_;
//printf("fhat_trans_gpu %d  %d  j:%d  i:%d  k:%d \n", j_i_k, NY_*NX_*NZ_, j, i, k);
        fhat_trans_gpu[j_i_k + 0*NY_*NX_*NZ_] = 0.250*ft1;
        fhat_trans_gpu[j_i_k + 1*NY_*NX_*NZ_] = 0.250*ft2 + 0.50*ft6;
        fhat_trans_gpu[j_i_k + 2*NY_*NX_*NZ_] = 0.250*ft3;
        fhat_trans_gpu[j_i_k + 3*NY_*NX_*NZ_] = 0.250*ft4;
        fhat_trans_gpu[j_i_k + 4*NY_*NX_*NZ_] = 0.250*ft5;
    } //i
    //Update net flux 
    if(iend == endi)
    {
        for(int i = 1; i <= endi; i++){
            int j_i_k = (j+ng) + (i-1+ng)*NY_ + (k+ng)*NY_*NX_;
            int j_i_1_k = (j+ng) + (i-2+ng)*NY_ + (k+ng)*NY_*NX_;
            for(int m = 0; m <5; m++){ //m-1
                fl_trans_gpu[j + (i-1)*ny + k*ny*nx + m*ny*nx*nz] = \
                (fhat_trans_gpu[j_i_k + m*NY_*NX_*NZ_]-fhat_trans_gpu[j_i_1_k + m*NY_*NX_*NZ_])*dcsidx_gpu[i-1];
            }
        }
    }
}

__global__ void __launch_bounds__(384)
euler_j_kernel_central_hip_(int nv, int nx, int ny, int nz, int ng, int jstart, int jend, int endi, int endj, int endk,
                            int lmax, int iflow, double* fhat_gpu, double* temperature_gpu, double* fl_gpu,
                            double* dcoe_gpu, double* detady_gpu, double* wv_gpu)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int k = threadIdx.y + blockIdx.y*blockDim.y;

    if(i >= endi || k >= endk) 
        return;
    int NY_ = ny+2*ng;
    int NX_ = nx+2*ng;
    int NZ_ = nz+2*ng;
    
    for(int j = 0; j <= endj; j++){
        double ft1 = 0.0;
        double ft2 = 0.0;
        double ft3 = 0.0;
        double ft4 = 0.0;
        double ft5 = 0.0;
        double ft6 = 0.0;
        for(int l = 1; l <= lmax; l++){
            double uvs1 = 0.0;
            double uvs2 = 0.0;
            double uvs3 = 0.0;
            double uvs4 = 0.0;
            double uvs5 = 0.0;
            double uvs6 = 0.0; 
            for(int m = 0; m <= (l-1); m++){

                int i_j_m_k   = (i+ng) + (j-1-m+ng)*  NX_ + (k+ng)*NY_*NX_;
                int i_j_m_l_k = (i+ng) + (j-1-m+l+ng)*NX_ + (k+ng)*NY_*NX_;

                double rhom  =  wv_gpu[i_j_m_k + NY_*NX_*NZ_*0] + wv_gpu[i_j_m_l_k + NY_*NX_*NZ_*0];

                double uui   = wv_gpu[i_j_m_k + NY_*NX_*NZ_*1];
                double vvi   = wv_gpu[i_j_m_k + NY_*NX_*NZ_*2];
                double wwi   = wv_gpu[i_j_m_k + NY_*NX_*NZ_*3];
                double ppi   = wv_gpu[i_j_m_k + NY_*NX_*NZ_*0]*temperature_gpu[i_j_m_k];
                double enti  = wv_gpu[i_j_m_k + NY_*NX_*NZ_*4];

                double uuip   = wv_gpu[i_j_m_l_k + NY_*NX_*NZ_*1];
                double vvip   = wv_gpu[i_j_m_l_k + NY_*NX_*NZ_*2];
                double wwip   = wv_gpu[i_j_m_l_k + NY_*NX_*NZ_*3];
                double ppip   = wv_gpu[i_j_m_l_k + NY_*NX_*NZ_*0]*temperature_gpu[i_j_m_l_k];
                double entip  = wv_gpu[i_j_m_l_k + NY_*NX_*NZ_*4];

                double uv_part = (vvi+vvip) * rhom;
                uvs1 = uvs1 + uv_part * (2.0);
                uvs2 = uvs2 + uv_part * (uui+uuip);
                uvs3 = uvs3 + uv_part * (vvi+vvip);
                uvs4 = uvs4 + uv_part * (wwi+wwip);
                uvs5 = uvs5 + uv_part * (enti+entip);
                uvs6 = uvs6 + (2.0)*(ppi+ppip);
            } //m
            double dcoe_gpu_local = dcoe_gpu[l-1 + (lmax-1)*4];
            ft1 = ft1 + dcoe_gpu_local*uvs1;
            ft2 = ft2 + dcoe_gpu_local*uvs2;
            ft3 = ft3 + dcoe_gpu_local*uvs3;
            ft4 = ft4 + dcoe_gpu_local*uvs4;
            ft5 = ft5 + dcoe_gpu_local*uvs5;
            ft6 = ft6 + dcoe_gpu_local*uvs6;
        } //l
        int i_j_k = (i+ng) + (j-1+ng)*NX_ + (k+ng)*NY_*NX_;
//printf("fhat_gpu %d  %d  i:%d  j:%d  k:%d \n", i_j_k, NY_*NX_*NZ_, j, i, k);
        double fh1 = 0.250*ft1;
        double fh2 = 0.250*ft2;
        double fh3 = 0.250*ft3;
        double fh4 = 0.250*ft4;
        double fh5 = 0.250*ft5;
        if (iflow==0) {
            if (j==0 || j==endj) {
                fh1 = 0.0;
                fh2 = 0.0;
                fh3 = 0.0;
                fh4 = 0.0;
                fh5 = 0.0;
            }
        }
        fh3 = fh3 + 0.50*ft6;

        fhat_gpu[i_j_k + 0*NY_*NX_*NZ_] = fh1;
        fhat_gpu[i_j_k + 1*NY_*NX_*NZ_] = fh2;
        fhat_gpu[i_j_k + 2*NY_*NX_*NZ_] = fh3;
        fhat_gpu[i_j_k + 3*NY_*NX_*NZ_] = fh4;
        fhat_gpu[i_j_k + 4*NY_*NX_*NZ_] = fh5;
    } //i
    //Update net flux 
    for(int j = jstart; j <= endj; j++){

        int i_j_k_0 = i + (j-1)*nx + k*ny*nx;
        int i_j_k = (i+ng) + (j-1+ng)*NX_ + (k+ng)*NY_*NX_;
        int i_j_1_k = (i+ng) + (j-2+ng)*NX_ + (k+ng)*NY_*NX_;

        for(int m = 0; m <5; m++){ //m-1
            fl_gpu[i_j_k_0 + m*ny*nx*nz] = fl_gpu[i_j_k_0 + m*ny*nx*nz] + \
            (fhat_gpu[i_j_k + m*NY_*NX_*NZ_] - fhat_gpu[i_j_1_k + m*NY_*NX_*NZ_])*detady_gpu[j-1];
        }
    }
}

__global__ void __launch_bounds__(384)
euler_k_kernel_central_hip_(int nv, int nx, int ny, int nz, int ng, int kstart, int kend, int endi, int endj, int endk,
                            int lmax, double* fhat_gpu, double* temperature_gpu, double* fl_gpu,
                            double* dcoe_gpu, double* dzitdz_gpu, double* wv_gpu)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    if(i >= endi || j >= endj) 
        return;
    int NY_ = ny+2*ng;
    int NX_ = nx+2*ng;
    int NZ_ = nz+2*ng;
    
    for(int k = 0; k <= endk; k++){
        double ft1 = 0.0;
        double ft2 = 0.0;
        double ft3 = 0.0;
        double ft4 = 0.0;
        double ft5 = 0.0;
        double ft6 = 0.0;
        for(int l = 1; l <= lmax; l++){
            double uvs1 = 0.0;
            double uvs2 = 0.0;
            double uvs3 = 0.0;
            double uvs4 = 0.0;
            double uvs5 = 0.0;
            double uvs6 = 0.0; 
            for(int m = 0; m <= (l-1); m++){

                int i_j_k_m = (i+ng) + (j+ng)*NX_ + (k-1-m+ng)*NY_*NX_;
                int i_j_k_m_l = (i+ng) + (j+ng)*NX_ + (k-1-m+l+ng)*NY_*NX_;

                double rhom  =  wv_gpu[i_j_k_m + NY_*NX_*NZ_*0] + wv_gpu[i_j_k_m_l + NY_*NX_*NZ_*0];

                double uui   = wv_gpu[i_j_k_m + NY_*NX_*NZ_*1];
                double vvi   = wv_gpu[i_j_k_m + NY_*NX_*NZ_*2];
                double wwi   = wv_gpu[i_j_k_m + NY_*NX_*NZ_*3];
                double ppi   = wv_gpu[i_j_k_m + NY_*NX_*NZ_*0]*temperature_gpu[i_j_k_m];
                double enti  = wv_gpu[i_j_k_m + NY_*NX_*NZ_*4];

                double uuip   = wv_gpu[i_j_k_m_l + NY_*NX_*NZ_*1];
                double vvip   = wv_gpu[i_j_k_m_l + NY_*NX_*NZ_*2];
                double wwip   = wv_gpu[i_j_k_m_l + NY_*NX_*NZ_*3];
                double ppip   = wv_gpu[i_j_k_m_l + NY_*NX_*NZ_*0]*temperature_gpu[i_j_k_m_l];
                double entip  = wv_gpu[i_j_k_m_l + NY_*NX_*NZ_*4];

                double uv_part = (wwi+wwip) * rhom;
                uvs1 = uvs1 + uv_part * (2.0);
                uvs2 = uvs2 + uv_part * (uui+uuip);
                uvs3 = uvs3 + uv_part * (vvi+vvip);
                uvs4 = uvs4 + uv_part * (wwi+wwip);
                uvs5 = uvs5 + uv_part * (enti+entip);
                uvs6 = uvs6 + (2.0)*(ppi+ppip);
            } //m
            double dcoe_gpu_local = dcoe_gpu[l-1 + (lmax-1)*4];
            ft1 = ft1 + dcoe_gpu_local*uvs1;
            ft2 = ft2 + dcoe_gpu_local*uvs2;
            ft3 = ft3 + dcoe_gpu_local*uvs3;
            ft4 = ft4 + dcoe_gpu_local*uvs4;
            ft5 = ft5 + dcoe_gpu_local*uvs5;
            ft6 = ft6 + dcoe_gpu_local*uvs6;
        } //l
        int i_j_k = (i+ng) + (j+ng)*NX_ + (k-1+ng)*NY_*NX_;
//printf("fhat_gpu %d  %d  i:%d  j:%d  k:%d \n", i_j_k, NY_*NX_*NZ_, j, i, k);
        fhat_gpu[i_j_k + 0*NY_*NX_*NZ_] = 0.250*ft1;
        fhat_gpu[i_j_k + 1*NY_*NX_*NZ_] = 0.250*ft2;
        fhat_gpu[i_j_k + 2*NY_*NX_*NZ_] = 0.250*ft3;
        fhat_gpu[i_j_k + 3*NY_*NX_*NZ_] = 0.250*ft4 + 0.50*ft6;
        fhat_gpu[i_j_k + 4*NY_*NX_*NZ_] = 0.250*ft5;
    } //i
    //Update net flux 
    for(int k = kstart; k <= endk; k++){

        int i_j_k_0 = i + (j)*nx + (k-1)*ny*nx;
        int i_j_k = (i+ng) + (j+ng)*NX_ + (k-1+ng)*NY_*NX_;
        int i_j_k_1 = (i+ng) + (j+ng)*NX_ + (k-2+ng)*NY_*NX_;
        for(int m = 0; m <5; m++){ //m-1
            fl_gpu[i_j_k_0 + m*ny*nx*nz] = fl_gpu[i_j_k_0 + m*ny*nx*nz] + \
            (fhat_gpu[i_j_k + m*NY_*NX_*NZ_] - fhat_gpu[i_j_k_1 + m*NY_*NX_*NZ_])*dzitdz_gpu[k-1];
        }
    }
}

extern "C"
{
   void euler_i_kernel_central_hip(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               int nv, int nx, int ny, int nz, int ng, int istart, int iend, int endi, int endj, int endk, int lmax,
               double* fhat_trans_gpu, double* temperature_trans_gpu, double* fl_trans_gpu,
               double* dcoe_gpu, double* dcsidx_gpu, double* wv_trans_gpu)
   {
      hipLaunchKernelGGL((euler_i_kernel_central_hip_), *grid, *block, shmem, stream,         //shmem and stream
                          nv, nx, ny, nz, ng, istart, iend, endi, endj, endk, lmax,
                          fhat_trans_gpu, temperature_trans_gpu, fl_trans_gpu,
                          dcoe_gpu, dcsidx_gpu, wv_trans_gpu);
   }

   void euler_j_kernel_central_hip(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               int nv,int nx,int ny,int nz,int ng, int jstart, int jend,int endi,int endj,int endk, int lmax, int iflow,
               double* fhat_gpu, double* temperature_gpu, double* fl_gpu,
               double* dcoe_gpu, double* detady_gpu, double* wv_gpu)
   {
      hipLaunchKernelGGL((euler_j_kernel_central_hip_), *grid, *block, shmem, stream,         //shmem and stream
                          nv, nx, ny, nz, ng, jstart, jend, endi, endj, endk, lmax, iflow,
                          fhat_gpu, temperature_gpu, fl_gpu,
                          dcoe_gpu, detady_gpu, wv_gpu);
   }

   void euler_k_kernel_central_hip(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               int nv,int nx,int ny,int nz,int ng, int kstart, int kend,int endi,int endj,int endk, int lmax,
               double* fhat_gpu, double* temperature_gpu, double* fl_gpu,
               double* dcoe_gpu, double* dzitdz_gpu, double* wv_gpu)
   {
      hipLaunchKernelGGL((euler_k_kernel_central_hip_), *grid, *block, shmem, stream,         //shmem and stream
                          nv, nx, ny, nz, ng, kstart, kend, endi, endj, endk, lmax,
                          fhat_gpu, temperature_gpu, fl_gpu,
                          dcoe_gpu, dzitdz_gpu, wv_gpu);
   }

}

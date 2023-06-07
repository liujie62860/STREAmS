#include <hip/hip_runtime.h>

__global__ void euler_i_kernel_wv_(double *wv_trans_gpu, double *wv_gpu, int nx, int ny, int nz, int ng)
{
#if 0
  do k=1,nz
   do i=1-ng,nx+ng
    do j=1,ny
     do iv=1,nv
      wv_trans_gpu(j,i,k,iv) = wv_gpu(i,j,k,iv)
     enddo
    enddo
   enddo
  enddo
#endif

   size_t kv_offset = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*blockIdx.z;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx+2*ng; i += blockDim.x){
       wv_trans_gpu[(j+ng) + i*(ny+2*ng) + kv_offset] = wv_gpu[i + (j+ng)*(nx+2*ng) + kv_offset];
   }
}

__global__ void euler_i_kernel_temp_(double *temperature_trans_gpu, double *temperature_gpu, int nx, int ny, int nz, int ng)
{
#if 0
  do k=1,nz
   do i=1-ng,nx+ng
    do j=1,ny
      temperature_trans_gpu(j,i,k) = temperature_gpu(i,j,k)
    enddo
   enddo
  enddo
#endif
   size_t k_offset = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng);
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx+2*ng; i += blockDim.x){
       temperature_trans_gpu[(j+ng) + i*(ny+2*ng) + k_offset] = temperature_gpu[i + (j+ng)*(nx+2*ng) + k_offset];
   }
}

__global__ void euler_i_kernel_fl_(double *fl_gpu, double *fl_trans_gpu, int nx, int ny, int nz, int ng)
{
#if 0
  do k=1,nz
   do j=1,ny
    do i=1,nx
     do iv=1,nv
      fl_gpu(i,j,k,iv) = fl_trans_gpu(j,i,k,iv)
     enddo
    enddo
   enddo
  enddo
#endif
   size_t kv_offset = ny*nx*blockIdx.y + ny*nx*nz*blockIdx.z;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx; i += blockDim.x){
       fl_gpu[i + j*nx + kv_offset] = fl_trans_gpu[j + i*ny + kv_offset];
   }
}

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

__global__ void rk_kernel_fln_fl_(double *fln_gpu, double *fl_gpu, int nx, int ny, int nz, double rhodt)
{
#if 0
  do k=1,nz
   do j=1,ny
    do i=1,nx
     do m=1,nv
      fln_gpu(i,j,k,m) = -rhodt*fl_gpu(i,j,k,m)
      fl_gpu(i,j,k,m) = 0._mykind
     enddo
    enddo
   enddo
  enddo
#endif
   size_t kv_offset = ny*nx*blockIdx.y + ny*nx*nz*blockIdx.z;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx; i += blockDim.x){
       fln_gpu[i + j*nx + kv_offset] = -1.0 * rhodt * fl_gpu[i + j*nx + kv_offset];
       fl_gpu[i + j*nx + kv_offset]  = (double)0.0;
   }
}

__global__ void rk_kernel_fln_(double *fln_gpu, double *fl_gpu, int nx, int ny, int nz, double gamdt)
{
#if 0
  do k=1,nz
   do j=1,ny
    do i=1,nx
     do m=1,nv
      fln_gpu(i,j,k,m) = fln_gpu(i,j,k,m)-gamdt*fl_gpu(i,j,k,m)
     enddo
    enddo
   enddo
  enddo
#endif
   size_t kv_offset = ny*nx*blockIdx.y + ny*nx*nz*blockIdx.z;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx; i += blockDim.x){
       fln_gpu[i + j*nx + kv_offset] = fln_gpu[i + j*nx + kv_offset] - \
                                       gamdt * fl_gpu[i + j*nx + kv_offset];
   }
}

__global__ void rk_kernel_w_fln_(double *w_gpu, double *fln_gpu, int nx, int ny, int nz, int ng)
{
#if 0
  do k=1,nz
   do j=1,ny
    do i=1,nx
     do m=1,nv
      w_gpu(i,j,k,m) = w_gpu(i,j,k,m)+fln_gpu(i,j,k,m)
     enddo
    enddo
   enddo
  enddo
#endif
   size_t kv_offset = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*blockIdx.z;
   size_t kv_offset_fl = ny*nx*blockIdx.y + ny*nx*nz*blockIdx.z;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx; i += blockDim.x){
       w_gpu[i+ng + (j+ng)*(nx+2*ng) + kv_offset] = w_gpu[i+ng + (j+ng)*(nx+2*ng) + kv_offset] + \
                                                 fln_gpu[i + j*nx + kv_offset_fl];
   }
}

__global__ void prims_kernel_(double *w_gpu, double *wv_gpu, double *temperature_gpu, int nx, int ny, int nz,
                              int ng, int ng_, double gm1)
{
   //ng_ is for prims_int reuse
   size_t kv_offset1 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng_) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*0;
   size_t kv_offset2 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng_) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*1;
   size_t kv_offset3 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng_) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*2;
   size_t kv_offset4 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng_) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*3;
   size_t kv_offset5 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng_) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*4;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < (nx+2*(ng-ng_)); i += blockDim.x){
       size_t ij_offset = i+ng_ + (j+ng_)*(nx+2*ng);

       double rho  = w_gpu[ij_offset + kv_offset1];
       double rhou = w_gpu[ij_offset + kv_offset2];
       double rhov = w_gpu[ij_offset + kv_offset3];
       double rhow = w_gpu[ij_offset + kv_offset4];
       double rhoe = w_gpu[ij_offset + kv_offset5];
       double ri   = 1.0 / rho;
       double uu   = rhou*ri;
       double vv   = rhov*ri;
       double ww   = rhow*ri;
       double qq   = 0.50 * (uu*uu+vv*vv+ww*ww);
       double pp   = gm1*(rhoe-rho*qq);

       temperature_gpu[ij_offset + (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng_)] = pp*ri;
       wv_gpu[ij_offset + kv_offset1] = rho;
       wv_gpu[ij_offset + kv_offset2] = uu;
       wv_gpu[ij_offset + kv_offset3] = vv;
       wv_gpu[ij_offset + kv_offset4] = ww;
       wv_gpu[ij_offset + kv_offset5] = (rhoe+pp)/rho;
   }
}

__global__ void bcwall_staggered_kernel_1_(double *w_gpu, int nx, int ny, int nz, int ng,
                                           double gm1, double gm, double t0)
{
   size_t kv_offset1 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*0;
   size_t kv_offset2 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*1;
   size_t kv_offset3 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*2;
   size_t kv_offset4 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*3;
   size_t kv_offset5 = (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*4;
   for( int i = threadIdx.x; i < nx; i += blockDim.x){
     for(int l=1; l <= ng; l++){
       size_t il_offset  = i+ng + (l-1+ng)*(nx+2*ng);
       size_t i1l_offset = i+ng + (ng-l)*(nx+2*ng); 

       double rho  = w_gpu[il_offset + kv_offset1];
       double uu   = w_gpu[il_offset + kv_offset2] / rho;
       double vv   = w_gpu[il_offset + kv_offset3] / rho;
       double ww   = w_gpu[il_offset + kv_offset4] / rho;
       double rhoe = w_gpu[il_offset + kv_offset5];
       double qq   = 0.50 * (uu*uu+vv*vv+ww*ww);
       double pp   = gm1  * (rhoe-rho*qq);
       double tt   = pp   / rho;
       tt   = 2.0 * t0 - tt;
       rho  = pp  / tt;
       w_gpu[i1l_offset + kv_offset1] =  rho;
       w_gpu[i1l_offset + kv_offset2] = -rho*uu;
       w_gpu[i1l_offset + kv_offset3] = -rho*vv;
       w_gpu[i1l_offset + kv_offset4] = -rho*ww;
       w_gpu[i1l_offset + kv_offset5] =  pp*gm + qq*rho;
     }
   }
}

__global__ void visflx_kernel_(double *wv_gpu, double *fhat_gpu, double *fl_gpu, int nx, int ny, int nz, int ng,
                               double *coeff_deriv1_gpu, double *dcsidx_gpu, double *detady_gpu, double *dzitdz_gpu, 
                               double *temperature_gpu, double *dcsidx2_gpu, double *detady2_gpu, double *dzitdz2_gpu,
                               double *coeff_clap_gpu, double *dcsidxs_gpu, double *detadys_gpu, double *dzitdzs_gpu, 
                               double sqgmr2, double s2tinf, double sqgmr, double vtexp, double ggmopr, int ivis, int visc_type)
{
   int j = blockIdx.x;
   int k = blockIdx.y;
   size_t xyzDim = (nx+2*ng) * (ny+2*ng) * (nz+2*ng);

   double uu, vv, ww, ux, vx, wx, tx, uy, vy, wy, ty, uz, vz, wz, tz;
   double ulapx, ulapy, ulapz, vlapx, vlapy, vlapz, wlapx, wlapy, wlapz, tlapx, tlapy, tlapz;

   for( int i = threadIdx.x; i < nx; i += blockDim.x) {
       size_t ijk_ng = i+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);

       uu = wv_gpu[ijk_ng + 1*xyzDim];
       vv = wv_gpu[ijk_ng + 2*xyzDim];
       ww = wv_gpu[ijk_ng + 3*xyzDim];
       ux = 0.0;       vx = 0.0;       wx = 0.0;       tx = 0.0;
       uy = 0.0;       vy = 0.0;       wy = 0.0;       ty = 0.0;
       uz = 0.0;       vz = 0.0;       wz = 0.0;       tz = 0.0;
       size_t iljk, i_ljk, ijlk, ij_lk, ijkl, ijk_l;
       for(int l=1; l <= (ivis/2); l++) {
           double ccl = coeff_deriv1_gpu[l-1];
           iljk  = i+l+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           i_ljk = i-l+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           ijlk  = i+ng   + (j+l+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           ij_lk = i+ng   + (j-l+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           ijkl  = i+ng   + (j+ng)*(nx+2*ng) + (k+l+ng)*(nx+2*ng)*(ny+2*ng);
           ijk_l = i+ng   + (j+ng)*(nx+2*ng) + (k-l+ng)*(nx+2*ng)*(ny+2*ng);

           ux = ux+ccl*(wv_gpu[iljk + xyzDim*1] - wv_gpu[i_ljk + xyzDim*1]);
           vx = vx+ccl*(wv_gpu[iljk + xyzDim*2] - wv_gpu[i_ljk + xyzDim*2]);
           wx = wx+ccl*(wv_gpu[iljk + xyzDim*3] - wv_gpu[i_ljk + xyzDim*3]);
           tx = tx+ccl*(temperature_gpu[iljk]   -temperature_gpu[i_ljk]);

           uy = uy+ccl*(wv_gpu[ijlk + xyzDim*1] - wv_gpu[ij_lk + xyzDim*1]);
           vy = vy+ccl*(wv_gpu[ijlk + xyzDim*2] - wv_gpu[ij_lk + xyzDim*2]);
           wy = wy+ccl*(wv_gpu[ijlk + xyzDim*3] - wv_gpu[ij_lk + xyzDim*3]);
           ty = ty+ccl*(temperature_gpu[ijlk]  - temperature_gpu[ij_lk]);

           uz = uz+ccl*(wv_gpu[ijkl + xyzDim*1] - wv_gpu[ijk_l + xyzDim*1]);
           vz = vz+ccl*(wv_gpu[ijkl + xyzDim*2] - wv_gpu[ijk_l + xyzDim*2]);
           wz = wz+ccl*(wv_gpu[ijkl + xyzDim*3] - wv_gpu[ijk_l + xyzDim*3]);
           tz = tz+ccl*(temperature_gpu[ijkl]   - temperature_gpu[ijk_l]);
       }
       ux = ux*dcsidx_gpu[i];
       vx = vx*dcsidx_gpu[i];
       wx = wx*dcsidx_gpu[i];
       tx = tx*dcsidx_gpu[i];
       uy = uy*detady_gpu[j];
       vy = vy*detady_gpu[j];
       wy = wy*detady_gpu[j];
       ty = ty*detady_gpu[j];
       uz = uz*dzitdz_gpu[k];
       vz = vz*dzitdz_gpu[k];
       wz = wz*dzitdz_gpu[k];
       tz = tz*dzitdz_gpu[k];

       ulapx = coeff_clap_gpu[0]*uu;
       ulapy = ulapx;
       ulapz = ulapx;
       vlapx = coeff_clap_gpu[0]*vv;
       vlapy = vlapx;
       vlapz = vlapx;
       wlapx = coeff_clap_gpu[0]*ww;
       wlapy = wlapx;
       wlapz = wlapx;
       tlapx = coeff_clap_gpu[0]*temperature_gpu[ijk_ng];
       tlapy = tlapx;
       tlapz = tlapx;

       for(int l=1; l <= (ivis/2); l++) {
           double clapl = coeff_clap_gpu[l];
           iljk  = i+l+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           i_ljk = i-l+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           ijlk  = i+ng   + (j+l+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           ij_lk = i+ng   + (j-l+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng);
           ijkl  = i+ng   + (j+ng)*(nx+2*ng) + (k+l+ng)*(nx+2*ng)*(ny+2*ng);
           ijk_l = i+ng   + (j+ng)*(nx+2*ng) + (k-l+ng)*(nx+2*ng)*(ny+2*ng);

           ulapx = ulapx + clapl*(wv_gpu[iljk + xyzDim*1] + wv_gpu[i_ljk + xyzDim*1]);
           ulapy = ulapy + clapl*(wv_gpu[ijlk + xyzDim*1] + wv_gpu[ij_lk + xyzDim*1]);
           ulapz = ulapz + clapl*(wv_gpu[ijkl + xyzDim*1] + wv_gpu[ijk_l + xyzDim*1]);
           vlapx = vlapx + clapl*(wv_gpu[iljk + xyzDim*2] + wv_gpu[i_ljk + xyzDim*2]);
           vlapy = vlapy + clapl*(wv_gpu[ijlk + xyzDim*2] + wv_gpu[ij_lk + xyzDim*2]);
           vlapz = vlapz + clapl*(wv_gpu[ijkl + xyzDim*2] + wv_gpu[ijk_l + xyzDim*2]);
           wlapx = wlapx + clapl*(wv_gpu[iljk + xyzDim*3] + wv_gpu[i_ljk + xyzDim*3]);
           wlapy = wlapy + clapl*(wv_gpu[ijlk + xyzDim*3] + wv_gpu[ij_lk + xyzDim*3]);
           wlapz = wlapz + clapl*(wv_gpu[ijkl + xyzDim*3] + wv_gpu[ijk_l + xyzDim*3]);
           tlapx = tlapx + clapl*(temperature_gpu[iljk]   + temperature_gpu[i_ljk]);
           tlapy = tlapy + clapl*(temperature_gpu[ijlk]  + temperature_gpu[ij_lk]);
           tlapz = tlapz + clapl*(temperature_gpu[ijkl]   + temperature_gpu[ijk_l]);
       }
       ulapx = ulapx*dcsidxs_gpu[i]+ux*dcsidx2_gpu[i];
       vlapx = vlapx*dcsidxs_gpu[i]+vx*dcsidx2_gpu[i];
       wlapx = wlapx*dcsidxs_gpu[i]+wx*dcsidx2_gpu[i];
       tlapx = tlapx*dcsidxs_gpu[i]+tx*dcsidx2_gpu[i];
       ulapy = ulapy*detadys_gpu[j]+uy*detady2_gpu[j];
       vlapy = vlapy*detadys_gpu[j]+vy*detady2_gpu[j];
       wlapy = wlapy*detadys_gpu[j]+wy*detady2_gpu[j];
       tlapy = tlapy*detadys_gpu[j]+ty*detady2_gpu[j];
       ulapz = ulapz*dzitdzs_gpu[k]+uz*dzitdz2_gpu[k];
       vlapz = vlapz*dzitdzs_gpu[k]+vz*dzitdz2_gpu[k];
       wlapz = wlapz*dzitdzs_gpu[k]+wz*dzitdz2_gpu[k];
       tlapz = tlapz*dzitdzs_gpu[k]+tz*dzitdz2_gpu[k];

       double ulap  = ulapx+ulapy+ulapz;
       double vlap  = vlapx+vlapy+vlapz;
       double wlap  = wlapx+wlapy+wlapz;
       double tlap  = tlapx+tlapy+tlapz;

       double div3l   = ux+vy+wz;
       div3l = div3l/3.0;
       fhat_gpu[ijk_ng + xyzDim*5] = div3l;
       double tt   = temperature_gpu[ ijk_ng];
       double rmut, drmutdt;
       if (visc_type==1) {
           rmut    = sqgmr * pow(tt, vtexp);
           drmutdt = rmut*vtexp/tt;
       } else {
           double tt2     = tt*tt;
           double sqrtt   = sqrt(tt);
           double sdivt   = s2tinf/tt;
           double sdivt1  = 1.0 + sdivt;
           rmut    = sqgmr2*sqrtt/sdivt1;//  ! molecular viscosity

           drmutdt = sqgmr2*0.50/sqrtt+rmut*s2tinf/tt2;
           drmutdt = drmutdt/sdivt1;
       }

       double rmutx = drmutdt*tx;
       double rmuty = drmutdt*ty;
       double rmutz = drmutdt*tz;
       double sig11 = 2.0*(ux-div3l);
       double sig12 = uy+vx;
       double sig13 = uz+wx;
       double sig22 = 2.0*(vy-div3l);
       double sig23 = vz+wy;
       double sig33 = 2.0*(wz-div3l);

       double sigx = rmutx*sig11 +  rmuty*sig12 +  rmutz*sig13 +  rmut*(ulap);
       double sigy = rmutx*sig12 +  rmuty*sig22 +  rmutz*sig23 +  rmut*(vlap);
       double sigz = rmutx*sig13 +  rmuty*sig23 +  rmutz*sig33 +  rmut*(wlap);
       double sigq = sigx*uu+sigy*vv+sigz*ww+ \
                     (sig11*ux+sig12*uy+sig13*uz+sig12*vx+sig22*vy+sig23*vz+sig13*wx+sig23*wy+sig33*wz)*rmut+ \
                     (rmutx*tx+rmuty*ty+rmutz*tz+rmut*tlap)*ggmopr;

       size_t fl_offset = i + j*nx + k*nx*ny;
       fl_gpu[fl_offset + nx*ny*nz*1] -= sigx;
       fl_gpu[fl_offset + nx*ny*nz*2] -= sigy;
       fl_gpu[fl_offset + nx*ny*nz*3] -= sigz;
       fl_gpu[fl_offset + nx*ny*nz*4] -= sigq;
   }
}

__global__ void visflx_div_kernel_(double *wv_gpu, double *fhat_gpu, double *fl_gpu, int nx, int ny, int nz, int ng,
                                   double *coeff_deriv1_gpu, double *dcsidx_gpu, double *detady_gpu, double *dzitdz_gpu,
                                   double *temperature_gpu, double sqgmr2, double s2tinf, double sqgmr,
                                   double vtexp, int ivis, int visc_type)
{
   int j = blockIdx.x;
   int k = blockIdx.y;
   size_t xyzDim = (nx+2*ng) * (ny+2*ng) * (nz+2*ng);
   size_t xyzoffset6 = xyzDim*5;
   for( int i = threadIdx.x; i < nx; i += blockDim.x) {
       size_t ijk_ng = i+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng) ;

       double uu = wv_gpu[ijk_ng + 1*xyzDim];
       double vv = wv_gpu[ijk_ng + 2*xyzDim];
       double ww = wv_gpu[ijk_ng + 3*xyzDim];
       double divx3l  = 0.0;
       double divy3l  = 0.0;
       double divz3l  = 0.0;

       for(int l=1; l <= (ivis/2); l++) {
           double ccl = coeff_deriv1_gpu[l-1];
           divx3l = divx3l+ccl*(fhat_gpu[i+l+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng) + xyzoffset6] - \
                                fhat_gpu[i-l+ng + (j+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng) + xyzoffset6]);
           divy3l = divy3l+ccl*(fhat_gpu[i+ng + (j+l+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng) + xyzoffset6] - \
                                fhat_gpu[i+ng + (j-l+ng)*(nx+2*ng) + (k+ng)*(nx+2*ng)*(ny+2*ng) + xyzoffset6]);
           divz3l = divz3l+ccl*(fhat_gpu[i+ng + (j+ng)*(nx+2*ng) + (k+l+ng)*(nx+2*ng)*(ny+2*ng) + xyzoffset6] - \
                                fhat_gpu[i+ng + (j+ng)*(nx+2*ng) + (k-l+ng)*(nx+2*ng)*(ny+2*ng) + xyzoffset6]);
       }
       divx3l = divx3l*dcsidx_gpu[i];
       divy3l = divy3l*detady_gpu[j];
       divz3l = divz3l*dzitdz_gpu[k];

       double tt   = temperature_gpu[ ijk_ng];
       double rmut;
       if (visc_type==1) {
           rmut    = sqgmr * pow(tt, vtexp);
       } else {
         /* double tt2     = tt*tt; // no use */
           double sqrtt   = sqrt(tt);
           double sdivt   = s2tinf/tt;
           double sdivt1  = 1.0 + sdivt;
           rmut    = sqgmr2*sqrtt/sdivt1;//  ! molecular viscosity
       }
       double sigx = rmut*divx3l;
       double sigy = rmut*divy3l;
       double sigz = rmut*divz3l;
       double sigq = sigx*uu+sigy*vv+sigz*ww;

       size_t fl_offset = i + j*nx + k*nx*ny;
       fl_gpu[fl_offset + nx*ny*nz*1] -= sigx;
       fl_gpu[fl_offset + nx*ny*nz*2] -= sigy;
       fl_gpu[fl_offset + nx*ny*nz*3] -= sigz;
       fl_gpu[fl_offset + nx*ny*nz*4] -= sigq;
   }
}

__global__ void pgrad_kernel_1_(double *bulk5, double *w_gpu, double *fln_gpu, double *temperature_gpu, double *yn_gpu,
                                int nx, int ny, int nz, int ng)
{
#if 0
  do k=1,nz
   do j=1,ny
    do i=1,nx
     dy = yn_gpu(j+1)-yn_gpu(j)
     bulk_1 = bulk_1 + fln_gpu(i,j,k,1)*dy
     bulk_2 = bulk_2 + fln_gpu(i,j,k,2)*dy
     bulk_3 = bulk_3 + w_gpu(i,j,k,1)*dy
     bulk_4 = bulk_4 + w_gpu(i,j,k,2)*dy
     bulk_5 = bulk_5 + w_gpu(i,j,k,1)*temperature_gpu(i,j,k)*dy
    enddo
   enddo
  enddo
#endif
   int j = blockIdx.x;
   size_t jk_offset = (nx+2*ng)*(j+ng) + (ny+2*ng)*(nx+2*ng)*(blockIdx.y+ng);
   size_t jk_offset_fln =  nx*j + ny*nx*blockIdx.y;
   double dy = yn_gpu[j+1] - yn_gpu[j];
   double bulk_tmp_0 = 0.0;
   double bulk_tmp_1 = 0.0;
   double bulk_tmp_2 = 0.0;
   double bulk_tmp_3 = 0.0;
   double bulk_tmp_4 = 0.0;

   __shared__ double bulk5_shm[5];
   if(threadIdx.x < 5) bulk5_shm[threadIdx.x] = 0.0;
   __syncthreads();

   for( int i = threadIdx.x; i < nx; i += blockDim.x){    
       bulk_tmp_0 += fln_gpu[i + jk_offset_fln + ny*nx*nz*0] * dy;
       bulk_tmp_1 += fln_gpu[i + jk_offset_fln + ny*nx*nz*1] * dy;
       bulk_tmp_2 += w_gpu[i+ng + jk_offset + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*0] * dy;
       bulk_tmp_3 += w_gpu[i+ng + jk_offset + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*1] * dy;
       bulk_tmp_4 += w_gpu[i+ng + jk_offset + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*0] * dy * temperature_gpu[i+ng + jk_offset];
   }

   {
       __syncthreads();
       int delta = 1;
       for(int i=0; i<6 ;i++) {
           bulk_tmp_0 += __shfl_down(bulk_tmp_0,delta,64);
           bulk_tmp_1 += __shfl_down(bulk_tmp_1,delta,64);
           bulk_tmp_2 += __shfl_down(bulk_tmp_2,delta,64);
           bulk_tmp_3 += __shfl_down(bulk_tmp_3,delta,64);
           bulk_tmp_4 += __shfl_down(bulk_tmp_4,delta,64);
           delta+=delta;
       }
       __syncthreads();
       if (threadIdx.x%64 == 0) {
           atomicAdd(&bulk5_shm[0], bulk_tmp_0);
           atomicAdd(&bulk5_shm[1], bulk_tmp_1);
           atomicAdd(&bulk5_shm[2], bulk_tmp_2);
           atomicAdd(&bulk5_shm[3], bulk_tmp_3);
           atomicAdd(&bulk5_shm[4], bulk_tmp_4);
       }
       __syncthreads();
       if(threadIdx.x < 5) {
           atomicAdd(&bulk5[threadIdx.x], bulk5_shm[threadIdx.x]);
       }
   }
}

__global__ void pgrad_kernel_2_(double *wv_gpu, double *fln_gpu, int nx, int ny, int nz, int ng,
                                 double bulk5g_gpu_1, double bulk5g_gpu_2)
{
#if 0
 do k=1,nz
  do j=1,ny
   do i=1,nx
    uu = wv_gpu(i,j,k,2)
    fln_gpu(i,j,k,1) = fln_gpu(i,j,k,1) -    bulk5g_gpu(1)
    fln_gpu(i,j,k,2) = fln_gpu(i,j,k,2) -    bulk5g_gpu(2)
    fln_gpu(i,j,k,5) = fln_gpu(i,j,k,5) - uu*bulk5g_gpu(2)
   enddo
  enddo
 enddo
#endif
   int k = blockIdx.y;
   int j = blockIdx.x;
   for( int i = threadIdx.x; i < nx; i += blockDim.x){ 
       size_t ijk_offset_fl = i + j*nx + ny*nx*k;
       double uu = wv_gpu[i+ng + (nx+2*ng)*(j+ng) + (ny+2*ng)*(nx+2*ng)*(k+ng) + (ny+2*ng)*(nx+2*ng)*(nz+2*ng)*1];
       fln_gpu[ijk_offset_fl + nx*ny*nz*0] -= bulk5g_gpu_1;
       fln_gpu[ijk_offset_fl + nx*ny*nz*1] -= bulk5g_gpu_2;
       fln_gpu[ijk_offset_fl + nx*ny*nz*4] -= uu*bulk5g_gpu_2;
   }
}

extern "C"
{

   void euler_i_kernel_wv(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *wv_trans_gpu, double *wv_gpu, int nx, int ny, int nz, int ng)
   {
      hipLaunchKernelGGL((euler_i_kernel_wv_), *grid, *block, shmem, stream,
                          wv_trans_gpu, wv_gpu, nx, ny, nz, ng);
   }

   void euler_i_kernel_temp(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *temperature_trans_gpu, double *temperature_gpu, int nx, int ny, int nz, int ng)
   {
      hipLaunchKernelGGL((euler_i_kernel_temp_), *grid, *block, shmem, stream,
                          temperature_trans_gpu, temperature_gpu, nx, ny, nz, ng);
   }

   void euler_i_kernel_fl(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *fl_gpu, double *fl_trans_gpu, int nx, int ny, int nz, int ng)
   {
      hipLaunchKernelGGL((euler_i_kernel_fl_), *grid, *block, shmem, stream,
                          fl_gpu, fl_trans_gpu, nx, ny, nz, ng);
   }

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

   void rk_kernel_fln_fl(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *fln_gpu, double *fl_gpu, int nx, int ny, int nz, double rhodt)
   {
      hipLaunchKernelGGL((rk_kernel_fln_fl_), *grid, *block, shmem, stream,
                          fln_gpu, fl_gpu, nx, ny, nz, rhodt);
   }

   void rk_kernel_fln(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *fln_gpu, double *fl_gpu, int nx, int ny, int nz, double gamdt)
   {
      hipLaunchKernelGGL((rk_kernel_fln_), *grid, *block, shmem, stream,
                          fln_gpu, fl_gpu, nx, ny, nz, gamdt);
   }

   void rk_kernel_w_fln(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *w_gpu, double *fln_gpu, int nx, int ny, int nz, int ng)
   {
      hipLaunchKernelGGL((rk_kernel_w_fln_), *grid, *block, shmem, stream,
                          w_gpu, fln_gpu, nx, ny, nz, ng);
   }

   void prims_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *w_gpu, double *wv_gpu, double *temperature_gpu, int nx, int ny, int nz, int ng, int ng_, double gm1)
   {
      hipLaunchKernelGGL((prims_kernel_), *grid, *block, shmem, stream,
                          w_gpu, wv_gpu, temperature_gpu, nx, ny, nz, ng, ng_, gm1);
   }

   void bcwall_staggered_kernel_1(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *w_gpu, int nx, int ny, int nz, int ng, double gm1, double gm, double t0)
   {
      hipLaunchKernelGGL((bcwall_staggered_kernel_1_), *grid, *block, shmem, stream,
                          w_gpu, nx, ny, nz, ng, gm1, gm, t0);
   }

   void visflx_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                          double *wv_gpu, double *fhat_gpu, double *fl_gpu, int nx, int ny, int nz, int ng,
                          double *coeff_deriv1_gpu, double *dcsidx_gpu, double *detady_gpu, double *dzitdz_gpu,
                          double *temperature_gpu, double *dcsidx2_gpu, double *detady2_gpu, double *dzitdz2_gpu,
                          double *coeff_clap_gpu, double *dcsidxs_gpu, double *detadys_gpu, double *dzitdzs_gpu,
                          double sqgmr2, double s2tinf, double sqgmr, double vtexp, double ggmopr, int ivis, int visc_type)
   {
      hipLaunchKernelGGL((visflx_kernel_), *grid, *block, shmem, stream,
                          wv_gpu, fhat_gpu, fl_gpu, nx, ny, nz, ng, coeff_deriv1_gpu,
                          dcsidx_gpu, detady_gpu, dzitdz_gpu, temperature_gpu, dcsidx2_gpu, detady2_gpu, dzitdz2_gpu,
                          coeff_clap_gpu, dcsidxs_gpu, detadys_gpu, dzitdzs_gpu,
                          sqgmr2, s2tinf, sqgmr, vtexp, ggmopr, ivis, visc_type);
   }

   void visflx_div_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                          double *wv_gpu, double *fhat_gpu, double *fl_gpu, int nx, int ny, int nz, int ng,
                          double *coeff_deriv1_gpu, double *dcsidx_gpu, double *detady_gpu, double *dzitdz_gpu,
                          double *temperature_gpu, double sqgmr2, double s2tinf, 
                          double sqgmr, double vtexp, int ivis, int visc_type)
   {
      hipLaunchKernelGGL((visflx_div_kernel_), *grid, *block, shmem, stream,
                          wv_gpu, fhat_gpu, fl_gpu, nx, ny, nz, ng, coeff_deriv1_gpu, 
                          dcsidx_gpu, detady_gpu, dzitdz_gpu, temperature_gpu, sqgmr2, s2tinf, 
                          sqgmr, vtexp, ivis, visc_type);
   }

   void pgrad_kernel_1(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                       double *bulk5, double *w_gpu, double *fln_gpu, double *temperature_gpu, double *yn_gpu,
                       int nx, int ny, int nz, int ng)
   {
      hipLaunchKernelGGL((pgrad_kernel_1_), *grid, *block, shmem, stream,
                          bulk5, w_gpu, fln_gpu, temperature_gpu, yn_gpu, nx, ny, nz, ng);
   }

   void pgrad_kernel_2(dim3* grid, dim3* block, int shmem, hipStream_t stream,
               double *wv_gpu, double *fln_gpu, int nx, int ny, int nz, int ng, double bulk5g_gpu_1, double bulk5g_gpu_2)
   {
      hipLaunchKernelGGL((pgrad_kernel_2_), *grid, *block, shmem, stream,
                          wv_gpu, fln_gpu, nx, ny, nz, ng, bulk5g_gpu_1, bulk5g_gpu_2);
   }

}

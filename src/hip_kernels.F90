module hip_kernels
 use hipfort_types

 interface

   Subroutine rk_kernel_fln_fl( grid, block,shmem,hipStream, &
                             fln_gpu, fl_gpu, nx, ny, nz, rhodt) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, fln_gpu, fl_gpu
   real(8),value :: rhodt
   integer(c_int),value :: shmem, nx, ny, nz
   End Subroutine rk_kernel_fln_fl

   Subroutine rk_kernel_fln( grid, block,shmem,hipStream, &
                             fln_gpu, fl_gpu, nx, ny, nz, gamdt) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, fln_gpu, fl_gpu
   real(8),value :: gamdt
   integer(c_int),value :: shmem, nx, ny, nz
   End Subroutine rk_kernel_fln

   Subroutine rk_kernel_w_fln( grid, block,shmem,hipStream, &
                               w_gpu, fln_gpu, nx, ny, nz, ng) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, w_gpu, fln_gpu
   integer(c_int),value :: shmem, nx, ny, nz, ng
   End Subroutine rk_kernel_w_fln

   Subroutine euler_i_kernel_wv( grid, block,shmem,hipStream, &
                            wv_trans_gpu, wv_gpu, nx, ny, nz, ng) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, wv_trans_gpu, wv_gpu
   integer(c_int),value :: shmem, nx, ny, nz, ng
   End Subroutine euler_i_kernel_wv

   Subroutine euler_i_kernel_temp( grid, block,shmem,hipStream, &
                            temperature_trans_gpu, temperature_gpu, nx, ny, nz, ng) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, temperature_trans_gpu, temperature_gpu
   integer(c_int),value :: shmem, nx, ny, nz, ng
   End Subroutine euler_i_kernel_temp

   Subroutine euler_i_kernel_fl( grid, block,shmem,hipStream, &
                                   fl_gpu, fl_trans_gpu, nx, ny, nz, ng) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, fl_gpu, fl_trans_gpu
   integer(c_int),value :: shmem, nx, ny, nz, ng
   End Subroutine euler_i_kernel_fl

   Subroutine euler_i_kernel_central_hip( grid, block,shmem,hipStream, &
                            nv, nx, ny, nz, ng, istart, iend, endi, endj, endk, lmax, &
                            fhat_trans_gpu, temperature_trans_gpu, fl_trans_gpu, &
                            dcoe_gpu, dcsidx_gpu, wv_trans_gpu) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, fhat_trans_gpu, temperature_trans_gpu, &
                        fl_trans_gpu, dcoe_gpu, dcsidx_gpu, wv_trans_gpu
   integer(c_int),value :: shmem, nv, nx, ny, nz, ng, istart, iend, endi, endj, endk, lmax
   End Subroutine euler_i_kernel_central_hip

   subroutine euler_j_kernel_central_hip( grid, block,shmem,hipStream, &
                            nv,nx,ny,nz,ng, jstart, jend, endi,endj,endk,lmax,iflow, &
                            fhat_gpu, temperature_gpu, fl_gpu, &
                            dcoe_gpu, detady_gpu, wv_gpu) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, fhat_gpu, temperature_gpu, &
                        fl_gpu, dcoe_gpu, detady_gpu, wv_gpu
   integer(c_int),value :: shmem, nv,nx,ny,nz,ng, jstart,jend, endi,endj,endk,lmax,iflow
   End subroutine euler_j_kernel_central_hip

   subroutine euler_k_kernel_central_hip( grid, block,shmem,hipStream, &
                            nv,nx,ny,nz,ng, jstart, jend, endi,endj,endk,lmax, &
                            fhat_gpu, temperature_gpu, fl_gpu, &
                            dcoe_gpu, dzitdz_gpu, wv_gpu) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, fhat_gpu, temperature_gpu, &
                        fl_gpu, dcoe_gpu, dzitdz_gpu, wv_gpu
   integer(c_int),value :: shmem, nv,nx,ny,nz,ng, jstart,jend, endi,endj,endk,lmax
   End subroutine euler_k_kernel_central_hip

   Subroutine prims_kernel( grid, block,shmem,hipStream, &
                            w_gpu, wv_gpu, temperature_gpu, nx, ny, nz, ng, ng_, gm1) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, w_gpu, wv_gpu, temperature_gpu
   real(8),value :: gm1
   integer(c_int),value :: shmem, nx, ny, nz, ng, ng_
   End Subroutine prims_kernel

   Subroutine bcwall_staggered_kernel_1( grid, block,shmem,hipStream, &
                            w_gpu, nx, ny, nz, ng, gm1, gm, t0) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, w_gpu
   real(8),value :: gm1, gm , t0
   integer(c_int),value :: shmem, nx, ny, nz, ng
   End Subroutine bcwall_staggered_kernel_1

   Subroutine visflx_kernel( grid, block,shmem,hipStream, &
                          wv_gpu, fhat_gpu, fl_gpu, nx, ny, nz, ng, coeff_deriv1_gpu, &
                          dcsidx_gpu, detady_gpu, dzitdz_gpu, temperature_gpu, dcsidx2_gpu, detady2_gpu, &
                          dzitdz2_gpu, coeff_clap_gpu, dcsidxs_gpu, detadys_gpu, dzitdzs_gpu, & 
                          sqgmr2, s2tinf, sqgmr, vtexp, ggmopr, ivis, visc_type) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, wv_gpu, fhat_gpu, fl_gpu, coeff_deriv1_gpu, &
                        dcsidx_gpu, detady_gpu, dzitdz_gpu, temperature_gpu, &
                        dcsidx2_gpu, detady2_gpu, dzitdz2_gpu, &
                        coeff_clap_gpu, dcsidxs_gpu, detadys_gpu, dzitdzs_gpu
   real(8),value :: sqgmr2, s2tinf, sqgmr, vtexp, ggmopr
   integer(c_int),value :: shmem, nx, ny, nz, ng, ivis, visc_type
   End Subroutine visflx_kernel

   Subroutine visflx_div_kernel( grid, block,shmem,hipStream, &
                          wv_gpu, fhat_gpu, fl_gpu, nx, ny, nz, ng, coeff_deriv1_gpu, &
                          dcsidx_gpu, detady_gpu, dzitdz_gpu, temperature_gpu, sqgmr2, s2tinf, &
                          sqgmr, vtexp, ivis, visc_type) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, wv_gpu, fhat_gpu, fl_gpu, coeff_deriv1_gpu, &
                        dcsidx_gpu, detady_gpu, dzitdz_gpu, temperature_gpu
   real(8),value :: sqgmr2, s2tinf, sqgmr, vtexp
   integer(c_int),value :: shmem, nx, ny, nz, ng, ivis, visc_type
   End Subroutine visflx_div_kernel

   subroutine pgrad_kernel_1( grid, block,shmem,hipStream, &
                              bulk5, w_gpu, fln_gpu, temperature_gpu, yn_gpu, &
                              nx,ny,nz,ng) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, bulk5, w_gpu, fln_gpu, temperature_gpu, yn_gpu
   integer(c_int),value :: shmem, nx,ny,nz,ng
   End subroutine pgrad_kernel_1

   Subroutine pgrad_kernel_2( grid, block,shmem,hipStream, &
                            wv_gpu, fln_gpu, nx, ny, nz, ng, bulk5g_gpu_1, bulk5g_gpu_2) bind(c)
   use iso_c_binding
   use hipfort_types
   implicit none
   type(dim3) :: grid, block
   type(c_ptr),value :: hipStream, wv_gpu, fln_gpu
   real(8),value :: bulk5g_gpu_1, bulk5g_gpu_2
   integer(c_int),value :: shmem, nx, ny, nz, ng
   End Subroutine pgrad_kernel_2

 endinterface
endmodule hip_kernels

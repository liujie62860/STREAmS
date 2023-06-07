subroutine rk
!
! 3-rd order RK solver to advance in time
!
 use mod_streams
 use mod_euler
#ifdef USE_OMP_HIP
  use iso_c_binding
  use hipfort      ! use hipfort
  use hipfort_check
  use hip_kernels
#endif
 implicit none
!
 integer :: i,j,k,m,istep,ilat
 real(mykind) :: alp,dt,gam,gamdt,rho,rhodt
 real(mykind) :: elapsed,startTiming,endTiming
!
 real(mykind) :: tt,st,et
#ifdef USE_OMP_HIP
 type(dim3) :: grid, tBlock
!#define CUDA_ASYNC
#endif
!
 if (cfl>0._mykind) then
  dt = dtmin*cfl 
 else
  dt = dtmin
 endif
 dtglobal = dt
!
! Loop on the 'nstage' stages of RK solver
!
 do istep=1,3 ! Alan Wray 3rd order RK method
!
  rho = rhovec(istep) ! Coefficient for nonlinear terms
  gam = gamvec(istep) ! Coefficient for nonlinear terms
  alp = gam+rho       ! Coefficient for linear terms
  rhodt = rho*dt
  gamdt = gam*dt
  alpdt = alp*dt
!
!st = mpi_wtime()
#ifdef USE_OMP_HIP
 grid   = dim3(ny,nz,nv)
 tBlock = dim3(64,1,1)
 call rk_kernel_fln_fl(grid, tBlock, 0, hipStream, &
                    fln_gpu_ptr, fl_gpu_ptr, nx, ny, nz, rhodt)
 call hipCheck(hipDeviceSynchronize())
#else
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
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
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
#endif
!et = mpi_wtime()
!tt = et-st
!if (masterproc) write(error_unit,*) 'RK-I time =', tt
!
#ifdef CUDA_ASYNC
  call bc(0)
  call bcswap_prepare()
  call prims_int()
  call euler_i(0+iorder/2,nx-iorder/2)
  call bcswap()
  call prims_ghost()
!
! Evaluation of Eulerian fluxes
!
  call euler_i(0,iorder/2-1)
  if (ibc(2)==4.or.ibc(2)==8) then
   call euler_i(nx+1-iorder/2,nx-1)
  else
   call euler_i(nx+1-iorder/2,nx)
  endif
#else
  call updateghost()
  call prims()
  if (ibc(2)==4.or.ibc(2)==8) then
   call euler_i(0,nx-1)
  else
   call euler_i(0,nx)
  endif
#endif
!
#ifdef CUDA_ASYNC
  call visflx()
  call bcswapdiv_prepare()
  call euler_j()
  call bcswapdiv()
  if (ndim==3) call euler_k()
!
  if (istep == 3 .and. tresduc<1._mykind) then
   call sensor()
   call bcswapduc_prepare()
   call visflx_div() ! No Cuda Sync here
   call bcswapduc()
  else
   call visflx_div()
   !@cuf iercuda=cudaDeviceSynchronize()
  endif
!
#else
  call euler_j()
  if (ndim==3) call euler_k()
  call visflx()
  call bcswapdiv_prepare()
  call bcswapdiv()
  if (istep == 3 .and. tresduc<1.) then
   call sensor()
   call bcswapduc_prepare()
   call bcswapduc()
  endif
  call visflx_div() ! No Cuda Sync here
  !@cuf iercuda=cudaDeviceSynchronize()
#endif
!
! Call to non-reflecting b.c. (to update f_x, g_y and h_z on the boundaries)
  call bc(1)
! 
#ifdef USE_OMP_HIP 
!#undef CUDA_ASYNC
! call hipCheck(hipDeviceSynchronize())
 grid   = dim3(ny,nz,nv)
 tBlock = dim3(64,1,1)
 call rk_kernel_fln(grid, tBlock, 0, hipStream, &
                    fln_gpu_ptr, fl_gpu_ptr, nx, ny, nz, gamdt)
 call hipCheck(hipDeviceSynchronize())
#else
 !$cuf kernel do(3) <<<*,*>>>
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
  do k=1,nz
   do j=1,ny
    do i=1,nx
     do m=1,nv
      fln_gpu(i,j,k,m) = fln_gpu(i,j,k,m)-gamdt*fl_gpu(i,j,k,m)
     enddo
    enddo
   enddo
  enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
#endif
!
  if (iflow==0) then
   call pgrad()
   dpdx = -dpdx/alpdt
  endif
!
! Updating solution in inner nodes
! 
#ifdef USE_OMP_HIP
 grid   = dim3(ny,nz,nv)
 tBlock = dim3(64,1,1)
 call rk_kernel_w_fln(grid, tBlock, 0, hipStream, &
                      w_gpu_ptr, fln_gpu_ptr, nx, ny, nz, ng)
 call hipCheck(hipDeviceSynchronize())
#else
 !$cuf kernel do(3) <<<*,*>>>
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
  do k=1,nz
   do j=1,ny
    do i=1,nx
     do m=1,nv
      w_gpu(i,j,k,m) = w_gpu(i,j,k,m)+fln_gpu(i,j,k,m)
     enddo
    enddo
   enddo
  enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
#endif

#ifdef USE_CUDA
 iercuda = cudaGetLastError()
 if (iercuda /= cudaSuccess) then
  call fail_input("CUDA ERROR! Try to reduce the number of Euler threads in cuda_definitions.h: "//cudaGetErrorString(iercuda))
 endif
#endif
!
  alpdtold  = alpdt 
  dfupdated = .false.
!
 enddo
!
 telaps = telaps+dt
!
end subroutine rk

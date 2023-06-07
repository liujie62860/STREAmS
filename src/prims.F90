subroutine prims
!
! Evaluation of the temperature field
!
 use mod_streams
#ifdef USE_OMP_HIP
  use iso_c_binding
  use hipfort      ! use hipfort
  use hipfort_check
  use hip_kernels
#endif
 implicit none
!
 real(mykind) :: rho,rhou,rhov,rhow,rhoe
 real(mykind) :: ri,uu,vv,ww,qq,pp
 integer :: i,j,k
!
#ifdef USE_OMP_HIP
 type(dim3) :: grid, tBlock

 grid   = dim3(ny+2*ng,nz+2*ng,1)
 tBlock = dim3(128,1,1)
 call prims_kernel(grid, tBlock, 0, hipStream, &
                   w_gpu_ptr, wv_gpu_ptr, temperature_gpu_ptr, nx, ny, nz, ng, 0, gm1)
 call hipCheck(hipDeviceSynchronize()) 
#else
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
 do k=1-ng,nz+ng
  do j=1-ng,ny+ng
   do i=1-ng,nx+ng
    rho  = w_gpu(i,j,k,1)
    rhou = w_gpu(i,j,k,2)
    rhov = w_gpu(i,j,k,3)
    rhow = w_gpu(i,j,k,4)
    rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho
    uu   = rhou*ri
    vv   = rhov*ri
    ww   = rhow*ri
    qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq)
    temperature_gpu(i,j,k) = pp*ri

    wv_gpu(i,j,k,1) = rho
    wv_gpu(i,j,k,2) = uu
    wv_gpu(i,j,k,3) = vv
    wv_gpu(i,j,k,4) = ww
    wv_gpu(i,j,k,5) = (rhoe+pp)/rho
   enddo
  enddo
 enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
#endif
!
end subroutine prims

subroutine prims_int
!
! Evaluation of the temperature field (internal nodes)
!
 use mod_streams
 implicit none
!
 real(mykind) :: rho,rhou,rhov,rhow,rhoe
 real(mykind) :: ri,uu,vv,ww,qq,pp
 integer :: i,j,k
!
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
 do k=1,nz
  do j=1,ny
   do i=1,nx
    rho  = w_gpu(i,j,k,1)
    rhou = w_gpu(i,j,k,2)
    rhov = w_gpu(i,j,k,3)
    rhow = w_gpu(i,j,k,4)
    rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho
    uu   = rhou*ri
    vv   = rhov*ri
    ww   = rhow*ri
    qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq)
    temperature_gpu(i,j,k) = pp*ri

    wv_gpu(i,j,k,1) = rho
    wv_gpu(i,j,k,2) = uu
    wv_gpu(i,j,k,3) = vv
    wv_gpu(i,j,k,4) = ww
    wv_gpu(i,j,k,5) = (rhoe+pp)/rho
   enddo
  enddo
 enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
!
end subroutine prims_int

subroutine prims_ghost
!
! Evaluation of the temperature field (ghost nodes)
!
 use mod_streams
 implicit none
!
 real(mykind) :: rho,rhou,rhov,rhow,rhoe
 real(mykind) :: ri,uu,vv,ww,qq,pp
 integer :: i,j,k
!
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
 do k=1-ng,0 ; do j=1-ng,ny+ng ; do i=1-ng,nx+ng 
    rho  = w_gpu(i,j,k,1) ; rhou = w_gpu(i,j,k,2) ; rhov = w_gpu(i,j,k,3) ; rhow = w_gpu(i,j,k,4) ; rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho ; uu   = rhou*ri ; vv   = rhov*ri ; ww   = rhow*ri ; qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq) ; temperature_gpu(i,j,k) = pp*ri
    wv_gpu(i,j,k,1) = rho ; wv_gpu(i,j,k,2) = uu ; wv_gpu(i,j,k,3) = vv ; wv_gpu(i,j,k,4) = ww ; wv_gpu(i,j,k,5) = (rhoe+pp)/rho
 enddo ; enddo ; enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
 do k=nz+1,nz+ng ; do j=1-ng,ny+ng ; do i=1-ng,nx+ng 
    rho  = w_gpu(i,j,k,1) ; rhou = w_gpu(i,j,k,2) ; rhov = w_gpu(i,j,k,3) ; rhow = w_gpu(i,j,k,4) ; rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho ; uu   = rhou*ri ; vv   = rhov*ri ; ww   = rhow*ri ; qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq) ; temperature_gpu(i,j,k) = pp*ri
    wv_gpu(i,j,k,1) = rho ; wv_gpu(i,j,k,2) = uu ; wv_gpu(i,j,k,3) = vv ; wv_gpu(i,j,k,4) = ww ; wv_gpu(i,j,k,5) = (rhoe+pp)/rho
 enddo ; enddo ; enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
 do k=1-ng,nz+ng ; do j=1-ng,0 ; do i=1-ng,nx+ng 
    rho  = w_gpu(i,j,k,1) ; rhou = w_gpu(i,j,k,2) ; rhov = w_gpu(i,j,k,3) ; rhow = w_gpu(i,j,k,4) ; rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho ; uu   = rhou*ri ; vv   = rhov*ri ; ww   = rhow*ri ; qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq) ; temperature_gpu(i,j,k) = pp*ri
    wv_gpu(i,j,k,1) = rho ; wv_gpu(i,j,k,2) = uu ; wv_gpu(i,j,k,3) = vv ; wv_gpu(i,j,k,4) = ww ; wv_gpu(i,j,k,5) = (rhoe+pp)/rho
 enddo ; enddo ; enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3) 
 do k=1-ng,nz+ng ; do j=ny+1,ny+ng ; do i=1-ng,nx+ng 
    rho  = w_gpu(i,j,k,1) ; rhou = w_gpu(i,j,k,2) ; rhov = w_gpu(i,j,k,3) ; rhow = w_gpu(i,j,k,4) ; rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho ; uu   = rhou*ri ; vv   = rhov*ri ; ww   = rhow*ri ; qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq) ; temperature_gpu(i,j,k) = pp*ri
    wv_gpu(i,j,k,1) = rho ; wv_gpu(i,j,k,2) = uu ; wv_gpu(i,j,k,3) = vv ; wv_gpu(i,j,k,4) = ww ; wv_gpu(i,j,k,5) = (rhoe+pp)/rho
 enddo ; enddo ; enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3) 
 do k=1-ng,nz+ng ; do j=1-ng,ny+ng ; do i=1-ng,0
    rho  = w_gpu(i,j,k,1) ; rhou = w_gpu(i,j,k,2) ; rhov = w_gpu(i,j,k,3) ; rhow = w_gpu(i,j,k,4) ; rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho ; uu   = rhou*ri ; vv   = rhov*ri ; ww   = rhow*ri ; qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq) ; temperature_gpu(i,j,k) = pp*ri
    wv_gpu(i,j,k,1) = rho ; wv_gpu(i,j,k,2) = uu ; wv_gpu(i,j,k,3) = vv ; wv_gpu(i,j,k,4) = ww ; wv_gpu(i,j,k,5) = (rhoe+pp)/rho
 enddo ; enddo ; enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3) 
 do k=1-ng,nz+ng ; do j=1-ng,ny+ng ; do i=nx+1,nx+ng 
    rho  = w_gpu(i,j,k,1) ; rhou = w_gpu(i,j,k,2) ; rhov = w_gpu(i,j,k,3) ; rhow = w_gpu(i,j,k,4) ; rhoe = w_gpu(i,j,k,5)
    ri   = 1._mykind/rho ; uu   = rhou*ri ; vv   = rhov*ri ; ww   = rhow*ri ; qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
    pp   = gm1*(rhoe-rho*qq) ; temperature_gpu(i,j,k) = pp*ri
    wv_gpu(i,j,k,1) = rho ; wv_gpu(i,j,k,2) = uu ; wv_gpu(i,j,k,3) = vv ; wv_gpu(i,j,k,4) = ww ; wv_gpu(i,j,k,5) = (rhoe+pp)/rho
 enddo ; enddo ; enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
!
end subroutine prims_ghost

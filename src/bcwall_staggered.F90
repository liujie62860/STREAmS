subroutine bcwall_staggered(ilat)
!
! Apply wall boundary conditions (staggered version)
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
 integer :: i,k,l,ilat
 real(mykind) :: rho,uu,vv,ww,qq,pp,tt,rhoe
#ifdef USE_OMP_HIP
 type(dim3) :: grid, tBlock
#endif
!
 if (ilat==1) then     ! left side
 elseif (ilat==2) then ! right side
 elseif (ilat==3) then ! lower side
#ifdef USE_OMP_HIP
 grid   = dim3(1,nz,1)
 tBlock = dim3(256,1,1)
 call hipCheck(hipDeviceSynchronize())
 call bcwall_staggered_kernel_1(grid, tBlock, 0, hipStream, &
                   w_gpu_ptr, nx, ny, nz, ng, gm1, gm, t0)
 call hipCheck(hipDeviceSynchronize())
#else
 !$cuf kernel do(2) <<<*,*>>>
 !$omp target 
 !$omp teams distribute parallel do collapse(2)
  do k=1,nz
   do i=1,nx
    do l=1,ng
     rho  = w_gpu(i,l,k,1)
     uu   = w_gpu(i,l,k,2)/w_gpu(i,l,k,1)
     vv   = w_gpu(i,l,k,3)/w_gpu(i,l,k,1)
     ww   = w_gpu(i,l,k,4)/w_gpu(i,l,k,1)
     rhoe = w_gpu(i,l,k,5)
     qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
     pp   = gm1*(rhoe-rho*qq)
     tt   = pp/rho
     tt   = 2._mykind*t0-tt
     rho  = pp/tt
     w_gpu(i,1-l,k,1) =  rho
     w_gpu(i,1-l,k,2) = -rho*uu
     w_gpu(i,1-l,k,3) = -rho*vv
     w_gpu(i,1-l,k,4) = -rho*ww
     w_gpu(i,1-l,k,5) =  pp*gm+qq*rho
    enddo
   enddo
  enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
#endif
 elseif (ilat==4) then  ! upper side
 !$cuf kernel do(2) <<<*,*>>>
 !$omp target 
 !$omp teams distribute parallel do collapse(2)
  do k=1,nz
   do i=1,nx
    do l=1,ng
     rho  = w_gpu(i,ny+1-l,k,1)
     uu   = w_gpu(i,ny+1-l,k,2)/w_gpu(i,ny+1-l,k,1)
     vv   = w_gpu(i,ny+1-l,k,3)/w_gpu(i,ny+1-l,k,1)
     ww   = w_gpu(i,ny+1-l,k,4)/w_gpu(i,ny+1-l,k,1)
     rhoe = w_gpu(i,ny+1-l,k,5)
     qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
     pp   = gm1*(rhoe-rho*qq)
     tt   = pp/rho
     tt   = 2._mykind*t0-tt
     rho  = pp/tt
     w_gpu(i,ny+l,k,1) =  rho
     w_gpu(i,ny+l,k,2) = -rho*uu
     w_gpu(i,ny+l,k,3) = -rho*vv
     w_gpu(i,ny+l,k,4) = -rho*ww
     w_gpu(i,ny+l,k,5) =  pp*gm+qq*rho
    enddo
   enddo
  enddo
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
 elseif (ilat==5) then  ! back side
 elseif (ilat==6) then  ! fore side
 endif
! 
end subroutine bcwall_staggered

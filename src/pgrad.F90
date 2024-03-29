subroutine pgrad
!
! Evaluate pressure gradient for channel flow
 use mod_streams
#ifdef USE_OMP_HIP
  use iso_c_binding
  use hipfort      ! use hipfort
  use hipfort_check
  use hip_kernels
#endif
!
 implicit none
!
 integer :: i,j,k,l
 real(mykind), dimension(5) :: bulk5
 real(mykind), dimension(5) :: bulk5g
 real(mykind) :: dy,uu
 real(mykind) :: bulk_1,bulk_2,bulk_3,bulk_4,bulk_5
#ifdef USE_OMP_HIP
 type(dim3) :: grid, tBlock
#endif
!
 if(pgradf == 0._mykind .or. mod(icyc,nprint)==0) then
  bulk_1 = 0._mykind
  bulk_2 = 0._mykind
  bulk_3 = 0._mykind
  bulk_4 = 0._mykind
  bulk_5 = 0._mykind
! 
  !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3) reduction(+:bulk_1, bulk_2, bulk_3, bulk_4, bulk_5)
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
  !$omp end target
  !@cuf iercuda=cudaDeviceSynchronize()
! 
  bulk5(1) = bulk_1
  bulk5(2) = bulk_2
  bulk5(3) = bulk_3
  bulk5(4) = bulk_4
  bulk5(5) = bulk_5
! 
  call mpi_allreduce(bulk5,bulk5g,5,mpi_prec,mpi_sum,mp_cart,iermpi)
! 
  bulk5g = bulk5g/nxmax/nzmax/(yn_gpu(ny+1)-yn_gpu(1))
  dpdx = bulk5g(2)
  rhobulk = bulk5g(3)
  ubulk = bulk5g(4)/rhobulk
  tbulk = bulk5g(5)/rhobulk
 endif ! pgradf > 0.

 if(pgradf > 0._mykind) then
  bulk5g(1) = 0._mykind
  bulk5g(2) = - pgradf*gamma*rm**2*alpdt
 endif

 bulk5g_gpu = bulk5g
!
! Add forcing terms in momentum and energy equation
#ifdef USE_OMP_HIP
  grid   = dim3(ny,nz,1)
  tBlock = dim3(128,1,1)
  call pgrad_kernel_2(grid, tBlock, 0, hipStream, &
                      wv_gpu_ptr, fln_gpu_ptr, nx, ny, nz, ng, bulk5g(1), bulk5g(2))
  call hipCheck(hipDeviceSynchronize())
#else
 !$cuf kernel do(3) <<<*,*>>> 
 !$omp target 
 !$omp teams distribute parallel do collapse(3)
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
 !$omp end target
 !@cuf iercuda=cudaDeviceSynchronize()
#endif
!
 end subroutine pgrad

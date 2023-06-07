subroutine allocate_vars()
!
! Allocate variables for the computation
!
 use mod_streams

#ifdef USE_OMP_HIP
  use iso_c_binding
  use hipfort      ! use hipfort
  use hipfort_check
#endif

 implicit none
!
#ifdef USE_CUDA
 allocate(fl_trans_gpu(1:ny,1:nx,1:nz,nv))
 allocate(temperature_trans_gpu(1-ng:ny+ng,1-ng:nx+ng,1-ng:nz+ng))
 allocate(fhat_trans_gpu(1-ng:ny+ng,1-ng:nx+ng,1-ng:nz+ng,5))

 allocate(wv_trans_gpu(1-ng:ny+ng,1-ng:nx+ng,1-ng:nz+ng,nv))

 allocate(w_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,nv))
 allocate(fl_gpu(nx,ny,nz,nv))
 allocate(fln_gpu(nx,ny,nz,nv))
 allocate(temperature_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
 allocate(ducros_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
 allocate(wmean_gpu(nv,1-ng:nx+1+ng,ny))
 allocate(dcsidx_gpu(nx),dcsidx2_gpu(nx),dcsidxs_gpu(nx))
 allocate(detady_gpu(ny),detady2_gpu(ny),detadys_gpu(ny))
 allocate(dzitdz_gpu(nz),dzitdz2_gpu(nz),dzitdzs_gpu(nz))
 allocate(x_gpu(1-ng:nx+ng))
 allocate(y_gpu(1-ng:ny+ng))
 allocate(yn_gpu(1:ny+1))
 allocate(z_gpu(1-ng:nz+ng))
 allocate(xg_gpu(1-ng:nxmax+ng+1))
 allocate(coeff_deriv1_gpu(3))
 allocate(coeff_clap_gpu(0:3))
 allocate(fhat_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,6))
 allocate(ibcnr_gpu(6))
 allocate(dcoe_gpu(4,4))
 allocate(winf_gpu(nv),winf1_gpu(nv))
 allocate(rf_gpu(3,1-nfmax:ny+nfmax,1-nfmax:nz+nfmax))
 allocate(rfy_gpu(3,ny,1-nfmax:nz+nfmax))
 allocate(vf_df_gpu(3,ny,nz))
 allocate(by_df_gpu(3,ny,-nfmax:nfmax))
 allocate(bz_df_gpu(3,ny,-nfmax:nfmax))
 allocate(amat_df_gpu(3,3,ny))

 allocate(gplus_x(5,2*ng,ny,nz))
 allocate(gminus_x(5,2*ng,ny,nz))
 allocate(gplus_y(5,2*ng,nx,nz))
 allocate(gminus_y(5,2*ng,nx,nz))
 allocate(gplus_z(5,2*ng,nx,ny))
 allocate(gminus_z(5,2*ng,nx,ny))
#endif

#ifdef USE_OMP_HIP
 allocate(fl_trans_gpu(1:ny,1:nx,1:nz,nv))
 allocate(wv_trans_gpu(1-ng:ny+ng,1-ng:nx+ng,1-ng:nz+ng,nv))
 allocate(temperature_trans_gpu(1-ng:ny+ng,1-ng:nx+ng,1-ng:nz+ng))
 allocate(fhat_trans_gpu(1-ng:ny+ng,1-ng:nx+ng,1-ng:nz+ng,5))
#endif
!
 allocate(wv_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,nv))
 allocate(w_order(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,nv))

 allocate(wallpfield_gpu(nx,nz))
 allocate(slicexy_gpu(nv,nx,ny))
 allocate(vf_df_old(3,ny,nz) )
 allocate(uf(3,ny,nz) )
 allocate(evmax_mat_yz(ny,nz) )
 allocate(evmax_mat_y(ny) )
 allocate(bulk5g_gpu(5))
 allocate(rtrms_ib_gpu(ny,nz))
 allocate(rtrms_ib_1d_gpu(ny))
!
 allocate(wbuf1s_gpu(ng,ny,nz,nv))  
 allocate(wbuf2s_gpu(ng,ny,nz,nv))  
 allocate(wbuf3s_gpu(nx,ng,nz,nv))  
 allocate(wbuf4s_gpu(nx,ng,nz,nv))  
 allocate(wbuf5s_gpu(nx,ny,ng,nv))  
 allocate(wbuf6s_gpu(nx,ny,ng,nv))  
 allocate(wbuf1r_gpu(ng,ny,nz,nv))  
 allocate(wbuf2r_gpu(ng,ny,nz,nv))  
 allocate(wbuf3r_gpu(nx,ng,nz,nv))  
 allocate(wbuf4r_gpu(nx,ng,nz,nv))  
 allocate(wbuf5r_gpu(nx,ny,ng,nv))  
 allocate(wbuf6r_gpu(nx,ny,ng,nv))  
#ifdef USE_OMP_HIP
 omp_info = hipHostRegister(c_loc(wbuf1s_gpu),INT((ng*ny*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf2s_gpu),INT((ng*ny*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf3s_gpu),INT((nx*ng*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf4s_gpu),INT((nx*ng*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf5s_gpu),INT((nx*ny*ng*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf6s_gpu),INT((nx*ny*ng*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf1r_gpu),INT((ng*ny*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf2r_gpu),INT((ng*ny*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf3r_gpu),INT((nx*ng*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf4r_gpu),INT((nx*ng*nz*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf5r_gpu),INT((nx*ny*ng*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
 omp_info = hipHostRegister(c_loc(wbuf6r_gpu),INT((nx*ny*ng*nv*8),KIND=C_SIZE_T),hipHostRegisterPortable)
#endif

 allocate(divbuf1s_gpu(ng,ny,nz))  
 allocate(divbuf2s_gpu(ng,ny,nz))  
 allocate(divbuf3s_gpu(nx,ng,nz))  
 allocate(divbuf4s_gpu(nx,ng,nz))  
 allocate(divbuf5s_gpu(nx,ny,ng))  
 allocate(divbuf6s_gpu(nx,ny,ng))  
 allocate(divbuf1r_gpu(ng,ny,nz))  
 allocate(divbuf2r_gpu(ng,ny,nz))  
 allocate(divbuf3r_gpu(nx,ng,nz))  
 allocate(divbuf4r_gpu(nx,ng,nz))  
 allocate(divbuf5r_gpu(nx,ny,ng))  
 allocate(divbuf6r_gpu(nx,ny,ng))  
 allocate(ducbuf1s_gpu(ng,ny,nz))
 allocate(ducbuf2s_gpu(ng,ny,nz))
 allocate(ducbuf3s_gpu(nx,ng,nz))
 allocate(ducbuf4s_gpu(nx,ng,nz))
 allocate(ducbuf5s_gpu(nx,ny,ng))
 allocate(ducbuf6s_gpu(nx,ny,ng))
 allocate(ducbuf1r_gpu(ng,ny,nz))
 allocate(ducbuf2r_gpu(ng,ny,nz))
 allocate(ducbuf3r_gpu(nx,ng,nz))
 allocate(ducbuf4r_gpu(nx,ng,nz))
 allocate(ducbuf5r_gpu(nx,ny,ng))
 allocate(ducbuf6r_gpu(nx,ny,ng))
!
 allocate(w(nv,1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
 allocate(fl(nx,ny,nz,nv))
 allocate(fln(nx,ny,nz,nv))
 allocate(temperature(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
 allocate(ducros(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
 allocate(wmean(nv,1-ng:nx+1+ng,ny))
 allocate(dcsidx(nx),dcsidx2(nx),dcsidxs(nx))
 allocate(detady(ny),detady2(ny),detadys(ny))
 allocate(dzitdz(nz),dzitdz2(nz),dzitdzs(nz))
 allocate(x(1-ng:nx+ng))
 allocate(y(1-ng:ny+ng))
 allocate(yn(1:ny+1))
 allocate(z(1-ng:nz+ng))
 allocate(xg(1-ng:nxmax+ng+1))
 allocate(coeff_deriv1(3))
 allocate(coeff_clap(0:3))
 allocate(fhat(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,6)) ! Size increased to exchange divergence
 allocate(ibcnr(6))
 allocate(dcoe(4,4))
 allocate(winf(nv),winf1(nv))
 allocate(rf(3,1-nfmax:ny+nfmax,1-nfmax:nz+nfmax))
 allocate(rfy(3,ny,1-nfmax:nz+nfmax))
 allocate(vf_df(3,ny,nz))
 allocate(by_df(3,ny,-nfmax:nfmax))
 allocate(bz_df(3,ny,-nfmax:nfmax))
 allocate(amat_df(3,3,ny))
 allocate(wallpfield(nx,nz))
 allocate(slicexy(nv,nx,ny))
!
 allocate(ibc(6))
 allocate(dxg(nxmax))
 allocate(dyg(nymax))
 allocate(dzg(nzmax))
 allocate(w_av(nvmean,nx,ny))
 allocate(w_avzg(nvmean,nx,ny))
 allocate(w_av_1d(nvmean,ny))
 allocate(w_avxzg(nvmean,ny))
 allocate(bx_df(3,nxmax,-nfmax:nfmax))
 allocate(wbuf1s(ng,ny,nz,nv))  
 allocate(wbuf2s(ng,ny,nz,nv))  
 allocate(wbuf3s(nx,ng,nz,nv))  
 allocate(wbuf4s(nx,ng,nz,nv))  
 allocate(wbuf5s(nx,ny,ng,nv))  
 allocate(wbuf6s(nx,ny,ng,nv))  
 allocate(wbuf1r(ng,ny,nz,nv))  
 allocate(wbuf2r(ng,ny,nz,nv))  
 allocate(wbuf3r(nx,ng,nz,nv))  
 allocate(wbuf4r(nx,ng,nz,nv))  
 allocate(wbuf5r(nx,ny,ng,nv))  
 allocate(wbuf6r(nx,ny,ng,nv))  
 allocate(divbuf1s(ng,ny,nz))  
 allocate(divbuf2s(ng,ny,nz))  
 allocate(divbuf3s(nx,ng,nz))  
 allocate(divbuf4s(nx,ng,nz))  
 allocate(divbuf5s(nx,ny,ng))  
 allocate(divbuf6s(nx,ny,ng))  
 allocate(divbuf1r(ng,ny,nz))  
 allocate(divbuf2r(ng,ny,nz))  
 allocate(divbuf3r(nx,ng,nz))  
 allocate(divbuf4r(nx,ng,nz))  
 allocate(divbuf5r(nx,ny,ng))  
 allocate(divbuf6r(nx,ny,ng))  
 allocate(ducbuf1s(ng,ny,nz))
 allocate(ducbuf2s(ng,ny,nz))
 allocate(ducbuf3s(nx,ng,nz))
 allocate(ducbuf4s(nx,ng,nz))
 allocate(ducbuf5s(nx,ny,ng))
 allocate(ducbuf6s(nx,ny,ng))
 allocate(ducbuf1r(ng,ny,nz))
 allocate(ducbuf2r(ng,ny,nz))
 allocate(ducbuf3r(nx,ng,nz))
 allocate(ducbuf4r(nx,ng,nz))
 allocate(ducbuf5r(nx,ny,ng))
 allocate(ducbuf6r(nx,ny,ng))
 allocate(yg(1-ng:nymax+ng  ))
 allocate(zg(1-ng:nzmax+ng  ))
!
endsubroutine allocate_vars

subroutine copy_cpu_to_gpu()
!
 use mod_streams
 implicit none
 integer :: i,j,k,iv
!
 do iv=1,5
  do k=1-ng,nz+ng
   do j=1-ng,ny+ng
    do i=1-ng,nx+ng
     w_order(i,j,k,iv) = w(iv,i,j,k)
    enddo
   enddo
  enddo
 enddo
#ifdef USE_CUDA
 w_gpu            = w_order
!fl_gpu           = fl
!fln_gpu          = fln
!temperature_gpu  = temperature
 ducros_gpu       = ducros
 wmean_gpu        = wmean
 dcsidx_gpu       = dcsidx      
 dcsidx2_gpu      = dcsidx2      
 dcsidxs_gpu      = dcsidxs     
 detady_gpu       = detady      
 detady2_gpu      = detady2     
 detadys_gpu      = detadys     
 dzitdz_gpu       = dzitdz      
 dzitdz2_gpu      = dzitdz2     
 dzitdzs_gpu      = dzitdzs     
 x_gpu            = x
 y_gpu            = y
 yn_gpu           = yn
 z_gpu            = z
 xg_gpu           = xg
 coeff_deriv1_gpu = coeff_deriv1
 coeff_clap_gpu   = coeff_clap
!fhat_gpu         = fhat
 ibcnr_gpu        = ibcnr
 dcoe_gpu         = dcoe
 winf_gpu         = winf
 winf1_gpu        = winf1
!rf_gpu           = rf
!rfy_gpu          = rfy ! not needed
 vf_df_gpu        = vf_df
 by_df_gpu        = by_df
 bz_df_gpu        = bz_df
 amat_df_gpu      = amat_df
#else
 call move_alloc( w_order      , w_gpu           )
 call move_alloc( fl           , fl_gpu          )
 call move_alloc( fln          , fln_gpu         )
 call move_alloc( temperature  , temperature_gpu )
 call move_alloc( ducros       , ducros_gpu      )
 call move_alloc( wmean        , wmean_gpu       )
 call move_alloc( dcsidx       , dcsidx_gpu      )
 call move_alloc( dcsidx2      , dcsidx2_gpu     )
 call move_alloc( dcsidxs      , dcsidxs_gpu     )
 call move_alloc( detady       , detady_gpu      )
 call move_alloc( detady2      , detady2_gpu     )
 call move_alloc( detadys      , detadys_gpu     )
 call move_alloc( dzitdz       , dzitdz_gpu      )
 call move_alloc( dzitdz2      , dzitdz2_gpu     )
 call move_alloc( dzitdzs      , dzitdzs_gpu     )
 call move_alloc( x            , x_gpu           )
 call move_alloc( y            , y_gpu           )
 call move_alloc( yn           , yn_gpu          )
 call move_alloc( z            , z_gpu           )
 call move_alloc( xg           , xg_gpu          )
 call move_alloc( coeff_deriv1 , coeff_deriv1_gpu)
 call move_alloc( coeff_clap   , coeff_clap_gpu  )
 call move_alloc( fhat         , fhat_gpu        )
 call move_alloc( ibcnr        , ibcnr_gpu       )
 call move_alloc( dcoe         , dcoe_gpu        )
 call move_alloc( winf         , winf_gpu        )
 call move_alloc( winf1        , winf1_gpu       )
 call move_alloc( rf           , rf_gpu          )
 call move_alloc( rfy          , rfy_gpu         )
 call move_alloc( vf_df        , vf_df_gpu       )
 call move_alloc( by_df        , by_df_gpu       )
 call move_alloc( bz_df        , bz_df_gpu       )
 call move_alloc( amat_df      , amat_df_gpu     )
#endif
end subroutine copy_cpu_to_gpu

subroutine copy_gpu_to_cpu
!
 use mod_streams
 implicit none
 integer :: i,j,k,iv
!
#ifdef USE_CUDA
 w_order = w_gpu
 temperature = temperature_gpu
 vf_df = vf_df_gpu
#else
 call move_alloc(w_gpu, w_order)
 call move_alloc(temperature_gpu , temperature)
 call move_alloc(vf_df_gpu , vf_df)
 call move_alloc(x_gpu , x)
 call move_alloc(xg_gpu,xg)
 call move_alloc(y_gpu , y)
 call move_alloc(z_gpu , z)
 call move_alloc(fl_gpu, fl)
#endif
 do iv=1,5
  do k=1-ng,nz+ng
   do j=1-ng,ny+ng
    do i=1-ng,nx+ng
     w(iv,i,j,k) = w_order(i,j,k,iv)
    enddo
   enddo
  enddo
 enddo
end subroutine copy_gpu_to_cpu

subroutine reset_cpu_gpu
!
 use mod_streams
 implicit none
!
#ifdef USE_CUDA
#else
 call move_alloc(w_order, w_gpu)
 call move_alloc(temperature, temperature_gpu)
 call move_alloc(vf_df, vf_df_gpu)
 call move_alloc(x , x_gpu)
 call move_alloc(xg,xg_gpu)
 call move_alloc(y , y_gpu)
 call move_alloc(z , z_gpu)
 call move_alloc(fl, fl_gpu)
#endif
end subroutine reset_cpu_gpu

subroutine allocate_dcu_mem()
 use mod_streams

 mykindSize=c_sizeof(dcoe_gpu(1,1))
 w_order_csize    = mykindSize*(nx+2*ng)*(ny+2*ng)*(nz+2*ng)*nv
 fl_csize       = mykindSize*nx*ny*nz*nv
 temperature_csize = mykindSize*(nx+2*ng)*(ny+2*ng)*(nz+2*ng)
 indx_csize = mykindSize*ng*ny*nz
 indy_csize = mykindSize*nx*ng*nz
 indz_csize = mykindSize*nx*ny*ng

 w_gpu_ptr = omp_target_alloc( w_order_csize, mydev)
 wv_gpu_ptr = omp_target_alloc( w_order_csize, mydev)
 wv_trans_gpu_ptr = omp_target_alloc( w_order_csize, mydev)
 temperature_gpu_ptr = omp_target_alloc( temperature_csize, mydev)
 temperature_trans_gpu_ptr = omp_target_alloc( temperature_csize, mydev)
 fl_gpu_ptr = omp_target_alloc( fl_csize, mydev)
 fln_gpu_ptr = omp_target_alloc( fl_csize, mydev)
 fl_trans_gpu_ptr = omp_target_alloc( fl_csize, mydev)

 fhat_gpu_ptr = omp_target_alloc( temperature_csize*6, mydev)
 fhat_trans_gpu_ptr = omp_target_alloc( temperature_csize*5, mydev)
 dcoe_gpu_ptr = omp_target_alloc( size(dcoe)*mykindSize, mydev)
 dcsidx_gpu_ptr = omp_target_alloc( size(dcsidx_gpu)*mykindSize, mydev)
 detady_gpu_ptr = omp_target_alloc( size(detady_gpu)*mykindSize, mydev)
 dzitdz_gpu_ptr = omp_target_alloc( size(dzitdz_gpu)*mykindSize, mydev)
 coeff_deriv1_gpu_ptr = omp_target_alloc( size(coeff_deriv1)*mykindSize, mydev)
 coeff_clap_gpu_ptr = omp_target_alloc( size(coeff_clap_gpu)*mykindSize, mydev)
 dcsidx2_gpu_ptr = omp_target_alloc( size(dcsidx2_gpu)*mykindSize, mydev)
 detady2_gpu_ptr = omp_target_alloc( size(detady2_gpu)*mykindSize, mydev)
 dzitdz2_gpu_ptr = omp_target_alloc( size(dzitdz2_gpu)*mykindSize, mydev)
 dcsidxs_gpu_ptr = omp_target_alloc( size(dcsidxs_gpu)*mykindSize, mydev)
 detadys_gpu_ptr = omp_target_alloc( size(detadys_gpu)*mykindSize, mydev)
 dzitdzs_gpu_ptr = omp_target_alloc( size(dzitdzs_gpu)*mykindSize, mydev)

#define CALLHIP(x) CALL hipCheck(x)
 CALL hipCheck(hipMalloc(divbuf1s_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(divbuf2s_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(divbuf3s_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(divbuf4s_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(divbuf5s_gpu_HIP, nx, ny, ng))
 CALL hipCheck(hipMalloc(divbuf6s_gpu_HIP, nx, ny, ng))
 CALL hipCheck(hipMalloc(divbuf1r_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(divbuf2r_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(divbuf3r_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(divbuf4r_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(divbuf5r_gpu_HIP, nx, ny, ng))
 CALL hipCheck(hipMalloc(divbuf6r_gpu_HIP, nx, ny, ng))

 CALL hipCheck(hipMalloc(ducbuf1s_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(ducbuf2s_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(ducbuf3s_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(ducbuf4s_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(ducbuf5s_gpu_HIP, nx, ny, ng))
 CALL hipCheck(hipMalloc(ducbuf6s_gpu_HIP, nx, ny, ng))
 CALL hipCheck(hipMalloc(ducbuf1r_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(ducbuf2r_gpu_HIP, ng, ny, nz))
 CALL hipCheck(hipMalloc(ducbuf3r_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(ducbuf4r_gpu_HIP, nx, ng, nz))
 CALL hipCheck(hipMalloc(ducbuf5r_gpu_HIP, nx, ny, ng))
 CALL hipCheck(hipMalloc(ducbuf6r_gpu_HIP, nx, ny, ng))

 CALLHIP(hipMalloc(wbuf1s_gpu_HIP, ng, ny, nz,nv))
 CALLHIP(hipMalloc(wbuf2s_gpu_HIP, ng, ny, nz,nv))
 CALLHIP(hipMalloc(wbuf3s_gpu_HIP, nx, ng, nz,nv))
 CALLHIP(hipMalloc(wbuf4s_gpu_HIP, nx, ng, nz,nv))
 CALLHIP(hipMalloc(wbuf5s_gpu_HIP, nx, ny, ng,nv))
 CALLHIP(hipMalloc(wbuf6s_gpu_HIP, nx, ny, ng,nv))
 CALLHIP(hipMalloc(wbuf1r_gpu_HIP, ng, ny, nz,nv))
 CALLHIP(hipMalloc(wbuf2r_gpu_HIP, ng, ny, nz,nv))
 CALLHIP(hipMalloc(wbuf3r_gpu_HIP, nx, ng, nz,nv))
 CALLHIP(hipMalloc(wbuf4r_gpu_HIP, nx, ng, nz,nv))
 CALLHIP(hipMalloc(wbuf5r_gpu_HIP, nx, ny, ng,nv))
 CALLHIP(hipMalloc(wbuf6r_gpu_HIP, nx, ny, ng,nv))

 dev_off = 0
 omp_info = omp_target_associate_ptr(c_loc(w_gpu), w_gpu_ptr, w_order_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wv_gpu), wv_gpu_ptr, w_order_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wv_trans_gpu), wv_trans_gpu_ptr, w_order_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(temperature_gpu), temperature_gpu_ptr, temperature_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(temperature_trans_gpu), temperature_trans_gpu_ptr, temperature_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(fl_gpu), fl_gpu_ptr, fl_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(fln_gpu), fln_gpu_ptr, fl_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(fl_trans_gpu), fl_trans_gpu_ptr, fl_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(fhat_gpu), fhat_gpu_ptr, temperature_csize*6, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(fhat_trans_gpu), fhat_trans_gpu_ptr, temperature_csize*5, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(dcoe_gpu), dcoe_gpu_ptr, size(dcoe)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(dcsidx_gpu), dcsidx_gpu_ptr, size(dcsidx)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(detady_gpu), detady_gpu_ptr, size(detady)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(dzitdz_gpu), dzitdz_gpu_ptr, size(dzitdz)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(coeff_deriv1_gpu), coeff_deriv1_gpu_ptr, size(coeff_deriv1)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(coeff_clap_gpu), coeff_clap_gpu_ptr, size(coeff_clap_gpu)*mykindSize, dev_off, mydev)
 
 omp_info = omp_target_associate_ptr(c_loc(dcsidx2_gpu), dcsidx2_gpu_ptr, size(dcsidx2)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(detady2_gpu), detady2_gpu_ptr, size(detady2)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(dzitdz2_gpu), dzitdz2_gpu_ptr, size(dzitdz2)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(dcsidxs_gpu), dcsidxs_gpu_ptr, size(dcsidxs)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(detadys_gpu), detadys_gpu_ptr, size(detadys)*mykindSize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(dzitdzs_gpu), dzitdzs_gpu_ptr, size(dzitdzs)*mykindSize, dev_off, mydev)

 omp_info = omp_target_associate_ptr(c_loc(wbuf1s_gpu), c_loc(wbuf1s_gpu_HIP), indx_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf2s_gpu), c_loc(wbuf2s_gpu_HIP), indx_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf3s_gpu), c_loc(wbuf3s_gpu_HIP), indy_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf4s_gpu), c_loc(wbuf4s_gpu_HIP), indy_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf5s_gpu), c_loc(wbuf5s_gpu_HIP), indz_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf6s_gpu), c_loc(wbuf6s_gpu_HIP), indz_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf1r_gpu), c_loc(wbuf1r_gpu_HIP), indx_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf2r_gpu), c_loc(wbuf2r_gpu_HIP), indx_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf3r_gpu), c_loc(wbuf3r_gpu_HIP), indy_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf4r_gpu), c_loc(wbuf4r_gpu_HIP), indy_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf5r_gpu), c_loc(wbuf5r_gpu_HIP), indz_csize*nv, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(wbuf6r_gpu), c_loc(wbuf6r_gpu_HIP), indz_csize*nv, dev_off, mydev)

 omp_info = omp_target_associate_ptr(c_loc(divbuf1s_gpu), c_loc(divbuf1s_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf2s_gpu), c_loc(divbuf2s_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf3s_gpu), c_loc(divbuf3s_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf4s_gpu), c_loc(divbuf4s_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf5s_gpu), c_loc(divbuf5s_gpu_HIP), indz_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf6s_gpu), c_loc(divbuf6s_gpu_HIP), indz_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf1r_gpu), c_loc(divbuf1r_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf2r_gpu), c_loc(divbuf2r_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf3r_gpu), c_loc(divbuf3r_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf4r_gpu), c_loc(divbuf4r_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf5r_gpu), c_loc(divbuf5r_gpu_HIP), indz_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(divbuf6r_gpu), c_loc(divbuf6r_gpu_HIP), indz_csize, dev_off, mydev)

 omp_info = omp_target_associate_ptr(c_loc(ducbuf1s_gpu), c_loc(ducbuf1s_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf2s_gpu), c_loc(ducbuf2s_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf3s_gpu), c_loc(ducbuf3s_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf4s_gpu), c_loc(ducbuf4s_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf5s_gpu), c_loc(ducbuf5s_gpu_HIP), indz_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf6s_gpu), c_loc(ducbuf6s_gpu_HIP), indz_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf1r_gpu), c_loc(ducbuf1r_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf2r_gpu), c_loc(ducbuf2r_gpu_HIP), indx_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf3r_gpu), c_loc(ducbuf3r_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf4r_gpu), c_loc(ducbuf4r_gpu_HIP), indy_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf5r_gpu), c_loc(ducbuf5r_gpu_HIP), indz_csize, dev_off, mydev)
 omp_info = omp_target_associate_ptr(c_loc(ducbuf6r_gpu), c_loc(ducbuf6r_gpu_HIP), indz_csize, dev_off, mydev)

end subroutine allocate_dcu_mem

subroutine deallocate_dcu_mem()
 use mod_streams

 call move_alloc(w_order, w_gpu)
 call move_alloc(temperature, temperature_gpu)
 call move_alloc(fl, fl_gpu)

 omp_info = omp_target_disassociate_ptr(c_loc(w_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wv_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wv_trans_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(temperature_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(temperature_trans_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(fl_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(fln_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(fl_trans_gpu), mydev)

 call move_alloc(w_gpu, w_order)
 call move_alloc(temperature_gpu , temperature)
 call move_alloc(fl_gpu, fl)

 omp_info = omp_target_disassociate_ptr(c_loc(fhat_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(fhat_trans_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dcoe_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dcsidx_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(detady_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dzitdz_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(coeff_deriv1_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(coeff_clap_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dcsidx2_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(detady2_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dzitdz2_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dcsidxs_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(detadys_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(dzitdzs_gpu), mydev)

 call omp_target_free(w_gpu_ptr, mydev)
 call omp_target_free(wv_gpu_ptr, mydev)
 call omp_target_free(wv_trans_gpu_ptr, mydev)
 call omp_target_free(temperature_gpu_ptr, mydev)
 call omp_target_free(temperature_trans_gpu_ptr, mydev)
 call omp_target_free(fl_gpu_ptr, mydev)
 call omp_target_free(fln_gpu_ptr, mydev)
 call omp_target_free(fl_trans_gpu_ptr, mydev)
 call omp_target_free(fhat_gpu_ptr, mydev)
 call omp_target_free(fhat_trans_gpu_ptr, mydev)
 call omp_target_free(dcoe_gpu_ptr, mydev)
 call omp_target_free(dcsidx_gpu_ptr, mydev)
 call omp_target_free(detady_gpu_ptr, mydev)
 call omp_target_free(dzitdz_gpu_ptr, mydev)
 call omp_target_free(coeff_deriv1_gpu_ptr, mydev) 
 call omp_target_free(coeff_clap_gpu_ptr, mydev)
 call omp_target_free(dcsidx2_gpu_ptr, mydev)
 call omp_target_free(detady2_gpu_ptr, mydev)
 call omp_target_free(dzitdz2_gpu_ptr, mydev)
 call omp_target_free(dcsidxs_gpu_ptr, mydev)
 call omp_target_free(detadys_gpu_ptr, mydev)
 call omp_target_free(dzitdzs_gpu_ptr, mydev)

 omp_info = omp_target_disassociate_ptr(c_loc(wbuf1s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf2s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf3s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf4s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf5s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf6s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf1r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf2r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf3r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf4r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf5r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(wbuf6r_gpu), mydev)

 omp_info = omp_target_disassociate_ptr(c_loc(divbuf1s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf2s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf3s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf4s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf5s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf6s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf1r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf2r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf3r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf4r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf5r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(divbuf6r_gpu), mydev)

 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf1s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf2s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf3s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf4s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf5s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf6s_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf1r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf2r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf3r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf4r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf5r_gpu), mydev)
 omp_info = omp_target_disassociate_ptr(c_loc(ducbuf6r_gpu), mydev)

#define CALL_OMPFREE(x)  CALL omp_target_free(x, mydev)
#define CALLHIP(x) CALL hipCheck(x)

CALLHIP( hipFree(wbuf1s_gpu_HIP) )
CALLHIP( hipFree(wbuf2s_gpu_HIP) )
CALLHIP( hipFree(wbuf3s_gpu_HIP) )
CALLHIP( hipFree(wbuf4s_gpu_HIP) )
CALLHIP( hipFree(wbuf5s_gpu_HIP) )
CALLHIP( hipFree(wbuf6s_gpu_HIP) )
CALLHIP( hipFree(wbuf1r_gpu_HIP) )
CALLHIP( hipFree(wbuf2r_gpu_HIP) )
CALLHIP( hipFree(wbuf3r_gpu_HIP) )
CALLHIP( hipFree(wbuf4r_gpu_HIP) )
CALLHIP( hipFree(wbuf5r_gpu_HIP) )
CALLHIP( hipFree(wbuf6r_gpu_HIP) )

CALLHIP( hipFree(divbuf1s_gpu_HIP) )
CALLHIP( hipFree(divbuf2s_gpu_HIP) )
CALLHIP( hipFree(divbuf3s_gpu_HIP) )
CALLHIP( hipFree(divbuf4s_gpu_HIP) )
CALLHIP( hipFree(divbuf5s_gpu_HIP) )
CALLHIP( hipFree(divbuf6s_gpu_HIP) )
CALLHIP( hipFree(divbuf1r_gpu_HIP) )
CALLHIP( hipFree(divbuf2r_gpu_HIP) )
CALLHIP( hipFree(divbuf3r_gpu_HIP) )
CALLHIP( hipFree(divbuf4r_gpu_HIP) )
CALLHIP( hipFree(divbuf5r_gpu_HIP) )
CALLHIP( hipFree(divbuf6r_gpu_HIP) )

CALLHIP( hipFree(ducbuf1s_gpu_HIP) )
CALLHIP( hipFree(ducbuf2s_gpu_HIP) )
CALLHIP( hipFree(ducbuf3s_gpu_HIP) )
CALLHIP( hipFree(ducbuf4s_gpu_HIP) )
CALLHIP( hipFree(ducbuf5s_gpu_HIP) )
CALLHIP( hipFree(ducbuf6s_gpu_HIP) )
CALLHIP( hipFree(ducbuf1r_gpu_HIP) )
CALLHIP( hipFree(ducbuf2r_gpu_HIP) )
CALLHIP( hipFree(ducbuf3r_gpu_HIP) )
CALLHIP( hipFree(ducbuf4r_gpu_HIP) )
CALLHIP( hipFree(ducbuf5r_gpu_HIP) )
CALLHIP( hipFree(ducbuf6r_gpu_HIP) )

end subroutine deallocate_dcu_mem

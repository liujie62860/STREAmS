module mod_streams
 use mpi
#ifdef USE_OMP_HIP
 use iso_c_binding
 use omp_lib
 use hipfort      ! use hipfort
 use hipfort_check
#endif
#ifdef USE_CUDA
  use cudafor
#endif
 use, intrinsic :: iso_fortran_env, only : error_unit
 implicit none
 save
!
 integer, parameter :: singtype = selected_real_kind(6,37)   ! single precision
 integer, parameter :: doubtype = selected_real_kind(15,307) ! double precision
#ifdef SINGLE_PRECISION
 integer, parameter :: mykind    = singtype
 integer, parameter :: mpi_prec = mpi_real4
 real(mykind), parameter :: tol_iter = 0.00001_mykind
#else
 integer, parameter :: mykind    = doubtype
 integer, parameter :: mpi_prec = mpi_real8
 real(mykind), parameter :: tol_iter = 0.000000001_mykind
#endif
!
 integer, parameter :: nv      =  5   ! Physical variables
 integer, parameter :: nsmoo   = 10   ! Number of smoothing iterations for wall-normal mesh
 integer, parameter :: nvmean  = 20
 integer, parameter :: nsolmax = 999999
 integer, parameter :: itmax   = 100000
 real(mykind), parameter :: pi = 4._mykind*atan(1._mykind)
!
! MPI related parameters 
!
 integer, parameter :: ndims = 3
 integer, dimension(:), allocatable  :: nblocks
 logical, dimension(:), allocatable  :: pbc
 integer, dimension(mpi_status_size) :: istatus
!
 integer, dimension(:), allocatable :: ncoords
 integer :: mp_cart,mp_cartx,mp_carty,mp_cartz
 integer :: nrank,nproc,nrank_x, nrank_y, nrank_z
 integer :: ileftx,irightx,ilefty,irighty,ileftz,irightz
 integer :: iermpi, iercuda
!
 integer :: nxmax
 integer :: nymax,nymaxwr
 integer :: nzmax
 integer :: nx
 integer :: ny
 integer :: nz
 integer :: ng   ! Number of ghost nodes
 integer :: ngdf ! Number of ghost nodes for digital filtering
 integer :: io_type
!
 integer :: enable_plot3d, enable_vtk
!
! Useful code variables
!
 real(mykind), parameter :: gamma  = 1.4_mykind
 real(mykind), parameter :: pr     = 0.72_mykind
 real(mykind), parameter :: gm1    = gamma-1._mykind
 real(mykind), parameter :: gm     = 1._mykind/gm1
 real(mykind), parameter :: ggmopr = gamma*gm/pr
 real(mykind), parameter :: vtexp  = 3._mykind/4._mykind
 real(mykind), parameter :: rfac   = 0.89_mykind !pr**(1._mykind/3._mykind)
 real(mykind) :: rm,re,taw,sqgmr,s2tinf,retauinflow,trat
 real(mykind) :: rtrms
 real(mykind), dimension(0:nsolmax) :: tsol, tsol_restart
 real(mykind) :: dtsave, dtsave_restart
 integer :: iflow
 integer :: idiski, ndim
 integer :: istore, istore_restart 
 integer :: iorder,iweno
 integer :: visc_type
 real(mykind) :: tresduc
 real(mykind), dimension(:,:), allocatable :: dcoe
 real(mykind), dimension(:,:), allocatable :: dcoe_gpu
 real(mykind), dimension(:), allocatable  :: winf,winf1
 real(mykind), dimension(:), allocatable  :: winf_gpu,winf1_gpu
 real(mykind) :: rho0,t0,p0,u0,v0,w0,s0
 real(mykind) :: dftscaling
 real(mykind) :: ximp,thetas,deflec,tanhsfac,xsh
 real(mykind) :: pgradf
 integer :: ivis
 logical :: masterproc
 logical :: dfupdated
!
 character(3) :: chx,chy,chz
 character(6) :: stat_io
!
! Vector of conservative variables and fluxes
 real(mykind), dimension(:,:,:,:), allocatable :: w,fl,fln
 real(mykind), dimension(:,:,:,:), allocatable :: w_order
 real(mykind), dimension(:,:,:,:), allocatable :: w_gpu,fl_gpu,fln_gpu

 real(mykind), dimension(:,:,:,:), allocatable :: fhat_trans_gpu, fl_trans_gpu
 real(mykind), dimension(:,:,:,:), allocatable :: wv_gpu, wv_trans_gpu
 real(mykind), dimension(:,:,:), allocatable :: temperature_trans_gpu

 real(mykind), dimension(:,:,:), allocatable :: temperature
 real(mykind), dimension(:,:,:), allocatable :: temperature_gpu
 logical, dimension(:,:,:), allocatable :: ducros,ducros_gpu
! 
! RK data
 real(mykind), dimension(3) :: gamvec,rhovec
 real(mykind) :: dtglobal,cfl,dtmin,alpdt,telaps,telaps0,alpdtold
 integer :: icyc,ncyc,ncyc0,nstep,nprint
!
! Coordinates and metric related quantities 
 real(mykind) :: rlx,rly,rlz,rlywr,dyp_target
 real(mykind), dimension(:), allocatable :: x
 real(mykind), dimension(:), allocatable :: y,yn
 real(mykind), dimension(:), allocatable :: z
 real(mykind), dimension(:), allocatable :: x_gpu
 real(mykind), dimension(:), allocatable :: y_gpu,yn_gpu
 real(mykind), dimension(:), allocatable :: z_gpu
 real(mykind), dimension(:), allocatable :: xg_gpu
 real(mykind), dimension(:), allocatable :: xg,yg,zg
 real(mykind), dimension(:), allocatable :: dcsidx,dcsidx2,dcsidxs
 real(mykind), dimension(:), allocatable :: detady,detady2,detadys
 real(mykind), dimension(:), allocatable :: dzitdz,dzitdz2,dzitdzs
 real(mykind), dimension(:), allocatable :: dcsidx_gpu,dcsidx2_gpu,dcsidxs_gpu
 real(mykind), dimension(:), allocatable :: detady_gpu,detady2_gpu,detadys_gpu
 real(mykind), dimension(:), allocatable :: dzitdz_gpu,dzitdz2_gpu,dzitdzs_gpu
 real(mykind), dimension(:), allocatable :: dxg,dyg,dzg
!
 real(mykind) :: dpdx,rhobulk,ubulk,tbulk
!
! BC data
 integer, dimension(:), allocatable :: ibc,ibcnr
 integer, dimension(:), allocatable :: ibcnr_gpu
 integer, parameter :: nfmax = 64
 integer :: rand_start 
 real(mykind), dimension(3) :: xlen_df
 real(mykind), dimension(:,:,:), allocatable :: vf_df,vf_df_gpu
 real(mykind), dimension(:,:,:), allocatable :: rf,rf_gpu,rfy,rfy_gpu
 real(mykind), dimension(:,:,:), allocatable :: bx_df,by_df,bz_df
 real(mykind), dimension(:,:,:), allocatable :: by_df_gpu,bz_df_gpu
 real(mykind), dimension(:,:,:), allocatable :: amat_df,amat_df_gpu
!
! Statistical quantities
 integer :: istat,itav,nstat,nstatloc
 real(mykind), dimension(:,:,:), allocatable :: w_av,w_avzg
 real(mykind), dimension(:,:  ), allocatable :: w_av_1d,w_avxzg
 real(mykind), dimension(:), allocatable     ::  xstat
 integer     , dimension(:), allocatable     :: ixstat
 integer     , dimension(:), allocatable     :: igxstat
!
! Mean field for initialization
 real(mykind), dimension(:,:,:), allocatable :: wmean,wmean_gpu

 real(mykind),dimension(:), allocatable :: coeff_deriv1
 real(mykind),dimension(:), allocatable :: coeff_deriv1_gpu
 real(mykind),dimension(:), allocatable :: coeff_clap
 real(mykind),dimension(:), allocatable :: coeff_clap_gpu

 real(mykind), dimension(:,:,:,:), allocatable :: wbuf1s_gpu, wbuf2s_gpu, wbuf3s_gpu, wbuf4s_gpu, wbuf5s_gpu, wbuf6s_gpu
 real(mykind), dimension(:,:,:,:), allocatable :: wbuf1r_gpu, wbuf2r_gpu, wbuf3r_gpu, wbuf4r_gpu, wbuf5r_gpu, wbuf6r_gpu
 real(mykind), dimension(:,:,:), allocatable :: divbuf1s_gpu, divbuf2s_gpu, divbuf3s_gpu, divbuf4s_gpu, divbuf5s_gpu, divbuf6s_gpu
 real(mykind), dimension(:,:,:), allocatable :: divbuf1r_gpu, divbuf2r_gpu, divbuf3r_gpu, divbuf4r_gpu, divbuf5r_gpu, divbuf6r_gpu
 logical, dimension(:,:,:), allocatable :: ducbuf1s_gpu, ducbuf2s_gpu, ducbuf3s_gpu, ducbuf4s_gpu, ducbuf5s_gpu, ducbuf6s_gpu
 logical, dimension(:,:,:), allocatable :: ducbuf1r_gpu, ducbuf2r_gpu, ducbuf3r_gpu, ducbuf4r_gpu, ducbuf5r_gpu, ducbuf6r_gpu

 real(mykind), dimension(:,:,:,:), allocatable :: wbuf1s, wbuf2s, wbuf3s, wbuf4s, wbuf5s, wbuf6s
 real(mykind), dimension(:,:,:,:), allocatable :: wbuf1r, wbuf2r, wbuf3r, wbuf4r, wbuf5r, wbuf6r
 real(mykind), dimension(:,:,:), allocatable :: divbuf1s, divbuf2s, divbuf3s, divbuf4s, divbuf5s, divbuf6s
 real(mykind), dimension(:,:,:), allocatable :: divbuf1r, divbuf2r, divbuf3r, divbuf4r, divbuf5r, divbuf6r
 logical, dimension(:,:,:), allocatable :: ducbuf1s, ducbuf2s, ducbuf3s, ducbuf4s, ducbuf5s, ducbuf6s
 logical, dimension(:,:,:), allocatable :: ducbuf1r, ducbuf2r, ducbuf3r, ducbuf4r, ducbuf5r, ducbuf6r

 real(mykind),dimension(:,:), allocatable :: wallpfield,wallpfield_gpu
 real(mykind),dimension(:,:,:), allocatable :: slicexy,slicexy_gpu
 real(mykind),dimension(:,:,:,:), allocatable :: fhat,fhat_gpu

 real(mykind), allocatable, dimension(:,:,:) :: vf_df_old
 real(mykind), allocatable, dimension(:,:,:) :: uf
 real(mykind), allocatable, dimension(:,:) :: evmax_mat_yz
 real(mykind), allocatable, dimension(:) :: evmax_mat_y
 real(mykind), allocatable, dimension(:) :: bulk5g_gpu
 real(mykind), allocatable, dimension(:,:) :: rtrms_ib_gpu
 real(mykind), allocatable, dimension(:) :: rtrms_ib_1d_gpu
!
 real(mykind), dimension(:,:,:,:), allocatable :: gplus_x,gminus_x
 real(mykind), dimension(:,:,:,:), allocatable :: gplus_y,gminus_y
 real(mykind), dimension(:,:,:,:), allocatable :: gplus_z,gminus_z
!
#ifdef USE_OMP_HIP
 integer :: local_comm, mydev
 integer :: omp_info
 integer(c_int) :: dev
 integer(c_size_t) :: dev_off, mykindSize, w_order_csize, fl_csize, temperature_csize, indx_csize, indy_csize, indz_csize
 target :: w_gpu,wv_gpu,wv_trans_gpu,temperature_gpu,temperature_trans_gpu,fl_gpu,fln_gpu,fl_trans_gpu,fhat_gpu, &
           fhat_trans_gpu, dcoe_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu
 target :: wbuf1s_gpu, wbuf2s_gpu, wbuf3s_gpu, wbuf4s_gpu, wbuf5s_gpu, wbuf6s_gpu
 target :: wbuf1r_gpu, wbuf2r_gpu, wbuf3r_gpu, wbuf4r_gpu, wbuf5r_gpu, wbuf6r_gpu

 real(mykind),pointer,dimension(:,:,:,:) :: wbuf1s_gpu_HIP, wbuf2s_gpu_HIP, wbuf3s_gpu_HIP, wbuf4s_gpu_HIP, wbuf5s_gpu_HIP, wbuf6s_gpu_HIP
 real(mykind),pointer,dimension(:,:,:,:) :: wbuf1r_gpu_HIP, wbuf2r_gpu_HIP, wbuf3r_gpu_HIP, wbuf4r_gpu_HIP, wbuf5r_gpu_HIP, wbuf6r_gpu_HIP
 real(mykind),pointer,dimension(:,:,:) :: divbuf1s_gpu_HIP, divbuf2s_gpu_HIP, divbuf3s_gpu_HIP, divbuf4s_gpu_HIP, divbuf5s_gpu_HIP, divbuf6s_gpu_HIP
 real(mykind),pointer,dimension(:,:,:) :: divbuf1r_gpu_HIP, divbuf2r_gpu_HIP, divbuf3r_gpu_HIP, divbuf4r_gpu_HIP, divbuf5r_gpu_HIP, divbuf6r_gpu_HIP

 target :: divbuf1s_gpu, divbuf2s_gpu, divbuf3s_gpu, divbuf4s_gpu, divbuf5s_gpu, divbuf6s_gpu
 target :: divbuf1r_gpu, divbuf2r_gpu, divbuf3r_gpu, divbuf4r_gpu, divbuf5r_gpu, divbuf6r_gpu
 target :: ducbuf1s_gpu, ducbuf2s_gpu, ducbuf3s_gpu, ducbuf4s_gpu, ducbuf5s_gpu, ducbuf6s_gpu
 target :: ducbuf1r_gpu, ducbuf2r_gpu, ducbuf3r_gpu, ducbuf4r_gpu, ducbuf5r_gpu, ducbuf6r_gpu
 target :: wbuf1s, wbuf2s, wbuf3s, wbuf4s, wbuf5s, wbuf6s
 target :: wbuf1r, wbuf2r, wbuf3r, wbuf4r, wbuf5r, wbuf6r
 target :: divbuf1s, divbuf2s, divbuf3s, divbuf4s, divbuf5s, divbuf6s
 target :: divbuf1r, divbuf2r, divbuf3r, divbuf4r, divbuf5r, divbuf6r
 target :: ducbuf1s, ducbuf2s, ducbuf3s, ducbuf4s, ducbuf5s, ducbuf6s
 target :: ducbuf1r, ducbuf2r, ducbuf3r, ducbuf4r, ducbuf5r, ducbuf6r

 type(c_ptr) :: hipStream, stream2
 type(c_ptr) :: w_gpu_ptr, wv_trans_gpu_ptr, wv_gpu_ptr
 type(c_ptr) :: dcoe_gpu_ptr
 type(c_ptr) :: dcsidx_gpu_ptr, detady_gpu_ptr, dzitdz_gpu_ptr
 type(c_ptr) :: fl_trans_gpu_ptr, fl_gpu_ptr, fln_gpu_ptr
 type(c_ptr) :: fhat_trans_gpu_ptr, fhat_gpu_ptr
 type(c_ptr) :: temperature_trans_gpu_ptr, temperature_gpu_ptr
#endif

#ifdef USE_CUDA
 attributes(device) :: fhat_trans_gpu, fl_trans_gpu
 attributes(device) :: temperature_trans_gpu
 attributes(device) :: wv_gpu, wv_trans_gpu

 integer :: local_comm, mydev
 attributes(device) :: w_gpu,fl_gpu,fln_gpu
 attributes(device) :: temperature_gpu,ducros_gpu
 attributes(device) :: dcsidx_gpu,dcsidx2_gpu,dcsidxs_gpu
 attributes(device) :: detady_gpu,detady2_gpu,detadys_gpu
 attributes(device) :: dzitdz_gpu,dzitdz2_gpu,dzitdzs_gpu
 attributes(device) :: x_gpu,y_gpu,yn_gpu,z_gpu
 attributes(device) :: xg_gpu
 attributes(device) :: coeff_deriv1_gpu
 attributes(device) :: coeff_clap_gpu
 attributes(device) :: ibcnr_gpu
 attributes(device) :: dcoe_gpu
 attributes(device) :: wmean_gpu
 attributes(device) :: winf_gpu,winf1_gpu
 attributes(device) :: rf_gpu,rfy_gpu
 attributes(device) :: vf_df_gpu
 attributes(device) :: by_df_gpu
 attributes(device) :: bz_df_gpu
 attributes(device) :: amat_df_gpu
 attributes(device) :: fhat_gpu
 attributes(device) :: wbuf1s_gpu, wbuf2s_gpu, wbuf3s_gpu, wbuf4s_gpu, wbuf5s_gpu, wbuf6s_gpu
 attributes(device) :: wbuf1r_gpu, wbuf2r_gpu, wbuf3r_gpu, wbuf4r_gpu, wbuf5r_gpu, wbuf6r_gpu
 attributes(device) :: divbuf1s_gpu, divbuf2s_gpu, divbuf3s_gpu, divbuf4s_gpu, divbuf5s_gpu, divbuf6s_gpu
 attributes(device) :: divbuf1r_gpu, divbuf2r_gpu, divbuf3r_gpu, divbuf4r_gpu, divbuf5r_gpu, divbuf6r_gpu
 attributes(device) :: ducbuf1s_gpu, ducbuf2s_gpu, ducbuf3s_gpu, ducbuf4s_gpu, ducbuf5s_gpu, ducbuf6s_gpu
 attributes(device) :: ducbuf1r_gpu, ducbuf2r_gpu, ducbuf3r_gpu, ducbuf4r_gpu, ducbuf5r_gpu, ducbuf6r_gpu

 attributes(pinned) :: wbuf1s, wbuf2s, wbuf3s, wbuf4s, wbuf5s, wbuf6s
 attributes(pinned) :: wbuf1r, wbuf2r, wbuf3r, wbuf4r, wbuf5r, wbuf6r
 attributes(pinned) :: divbuf1s, divbuf2s, divbuf3s, divbuf4s, divbuf5s, divbuf6s
 attributes(pinned) :: divbuf1r, divbuf2r, divbuf3r, divbuf4r, divbuf5r, divbuf6r
 attributes(pinned) :: ducbuf1s, ducbuf2s, ducbuf3s, ducbuf4s, ducbuf5s, ducbuf6s
 attributes(pinned) :: ducbuf1r, ducbuf2r, ducbuf3r, ducbuf4r, ducbuf5r, ducbuf6r

 integer(kind=cuda_stream_kind) :: stream1, stream2
 attributes(device) :: vf_df_old,uf
 attributes(device) :: evmax_mat_yz,evmax_mat_y
 attributes(device) :: bulk5g_gpu
 attributes(device) :: rtrms_ib_gpu,rtrms_ib_1d_gpu
 attributes(device) :: wallpfield_gpu
 attributes(device) :: slicexy_gpu

 attributes(device) :: gplus_x, gminus_x
 attributes(device) :: gplus_y, gminus_y
 attributes(device) :: gplus_z, gminus_z
#endif

#define USE_OMP
#ifdef USE_OMP
 CONTAINS
   subroutine set_device_gpu(myrank)
     use omp_lib

     implicit none
     integer, intent(in) :: myrank
     logical :: init_omp

     init_omp=omp_in_parallel()
     call omp_set_default_device(myrank)

   end subroutine set_device_gpu
#endif

end module mod_streams

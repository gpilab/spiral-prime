#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // For std::vector, etc.
#include <pybind11/complex.h>      
#include <pybind11/operators.h>    // For operator overloads if needed (e.g., for structs)

#include "GPIArray/GPIArray.hpp"     // Your custom Array class

namespace py = pybind11;
using namespace GPIArray; // Use GPIArray namespace
int kern_stage1_mltbnd(Array<double> &crdin, Array<std::complex<double> > **kerntmp, Array<double> **kernradtmp,
                Array<std::complex<double> > &kphase, Array<std::complex<double> > &tephase, Array<double> &fmap_residual,
                Array<double> &fmap_mean, Array<double> &peaks, Array<long> &mindex, 
                Array<double> &crdmag,
                double b0, double dwell, double gamma,
                int side, double freqinc,
                double te1, double te2, double te3)
  {
    int i,j,k,nb;
    int i2;
    int krad;
    int kernwidth, kernmid;
    double dtheta_dk;
    
    //auto begin = std::chrono::high_resolution_clock::now();
    
    for (i = 0; i < crdmag.size(0); i++)
      crdmag(i) = std::sqrt(crdin(0,i)*crdin(0,i) + crdin(1,i)*crdin(1,i));

  //************************************************
  // find  extreme cases for residual fmap_residual
  // then find the dimensions for the kernel table
  //************************************************

    double fmin = fmap_residual(0,0,0,0);
    double fmax = fmap_residual(0,0,0,0);
    double fmaxmag;
    double tau;

    for (i = 0; i < fmap_residual.size(0); i++) {
      for (j = 0; j < fmap_residual.size(1); j++) {
        for (k = 0; k < fmap_residual.size(2); k++) {
          for (nb = 0; nb < fmap_residual.size(3); nb++) {
            if (fmap_residual(i,j,k,nb) > fmax)  fmax = fmap_residual(i,j,k,nb);
            if (fmap_residual(i,j,k,nb) < fmin)  fmin = fmap_residual(i,j,k,nb);
      } } } }

    fmax = fmax;
    fmin = fmin;
    fmaxmag = std::max(std::abs(fmax),std::abs(fmin));


    // Now calculate the slope of the phase over the last 5% of the samples
    // We assume this is the maximum slope, and the maximum radius of the blurring
    // kernels is equal to the multiple of 2Pi for this slope

    // points at 95% and 100% of coord array 
    int i100 = crdmag.size(0)-1;
    int i95 = ((95*crdmag.size(0))/100)-1;
    if (i95 >= i100) i95 = i100-1; // should never happen but just to be safe
    tau = dwell*double(i100);

    // worst case dtheta from off-resonance
    // find this and normalize by "dk" to find worst dtheta/dk
    // this worst case, divided by 2pi, is the max radius of the kernels
    double dtheta_f = 2.*M_PI*fmaxmag*dwell*double(i100-i95);
    dtheta_dk = std::abs(dtheta_f)/(crdmag(i100)-crdmag(i95));
    krad = std::ceil(dtheta_dk/(2.*M_PI)); // largest possible k-space radius for blurring kernel
    
    // kernwidth will be the width of ktmp(), so that the broadest PSF in the FFT fits
    // kernmid is the center point of ktmp and the width of kern()
    kernwidth = 2*(krad+side+2);
    kernmid = krad+side+2;

    // Now find the rest of the kernel table dimensions, in the frequency dimension
    // min and max values for frequency indices, relative to f=0
    int minindex_f = std::floor(fmin/freqinc);
    int maxindex_f =  std::ceil(fmax/freqinc);

    // necessary because we use m1 = m0+1 below for linear interpolation of kernel indices
    maxindex_f++;

    // we are going to send these back to the main routine via mindex for proper indexing into the kernel table
    mindex(0) = minindex_f;
    mindex(1) = maxindex_f;

    // allocate kernel table, and assign to temp arrays for sending back to python wrapper via PYFI
    Array<std::complex<double> > *kern = new Array<std::complex<double> > (kernmid,kernmid,1+maxindex_f-minindex_f);
    *kerntmp=kern;

    Array<double> *kernrad = new Array<double> (1+maxindex_f-minindex_f);
    *kernradtmp=kernrad;

    (*kern) = (std::complex<double>)(0.);

    //**********************************************
    // Here we Populate the kernel kern
    //**********************************************
    double tstart[3];
    double kn[100000], rad, rk, kdel;
    int l,m,drad;
    long   r2;
    double theta_offres;
    double alpha, alphanorm;
    double k_m, k_b;
    double knorm;

    std::complex<double> kernval;
    std::complex<double> kv;

    // "kern()" is made to only store a quadrant of the kernels, so is just kernmid wide
    // Build temporary array ktmp for processing which are the full diameter of kernels in xy space
    // other arrays are also just for quadrant
    Array<std::complex<double> > ktmp(kernwidth, kernwidth, (*kern).size(2));
    ktmp.fill(std::complex<double>(0.));
    Array<double>  tau2d(kernmid, kernmid);

    tau2d = 0.;

    //-----------------------
    // 1. some initial calculations
    //-----------------------
    tstart[0] = te1;
    tstart[1] = te2;
    tstart[2] = te3;

    drad = (double)(kernwidth)*crdmag(i100); // data radius
    knorm = (double)(kernwidth);

    /* normalize k-space coordinates */
    for (i=0; i<(int)crdmag.size(0); i++){
      kn[i] = crdmag(i)*knorm;// kn goes from 0 to drad (i.e. data radius)
    }

    //-----------------------
    // 2. Populate tau2d array to use in larger kernel array below
    //-----------------------

    for (i=0; i< kernmid; i++) {
      i2 = i*i;
      for (j=i; j<kernmid; j++) {
        rk = std::sqrt((double)(i2 + j*j)); // rk is radius

        // find k that gives us kn[k] = rk
        if (rk > kn[0]) {
          for (k=0; (rk > kn[k] && k < i100); k++);
          k--;
          }
        else
          k=0;

        /* interpolate between sampled kr (kernel & traj) */
        if (k >= i100-1)
          kdel = 0;
        else if  (kn[k+1] > kn[k])
          kdel = (rk-kn[k])/(kn[k+1]-kn[k]);
        else
          kdel = 0;

        tau2d(i,j) = dwell*((double)(k) + kdel);
      } } // i j
      
    //-----------------------
    // 3. Calculate the desired r-space radius of each kernel
    //-----------------------
    for (l=0; l < (*kern).size(2); l++) {
      dtheta_f = 2.*M_PI*(double)(l+minindex_f)*freqinc * dwell*double(i100-i95); // outer phase slope from offres
      dtheta_dk = std::abs(dtheta_f)/(crdmag(i100)-crdmag(i95)); // worst possible dtheta/dk for this l
      (*kernrad)(l) = std::ceil(dtheta_dk / (2.*M_PI)) + side;
      }

    //-----------------------
    // 4. Now actually populate the kernel array kern()
    //-----------------------

    // 8-fold symmetry
    for (l=0; l < (*kern).size(2); l++) {
      for (i=0; i<kernmid; i++) {
        for (j=i; j<kernmid; j++) {
          theta_offres = 2.*M_PI * tau2d(i,j) * (double)(l+minindex_f)*freqinc;
          kernval = std::polar(1.,theta_offres);

          ktmp(kernmid+i,kernmid+j,l) = kernval;
          ktmp(kernmid-i,kernmid+j,l) = kernval;
          ktmp(kernmid+i,kernmid-j,l) = kernval;
          ktmp(kernmid-i,kernmid-j,l) = kernval;
          ktmp(kernmid+j,kernmid+i,l) = kernval;
          ktmp(kernmid-j,kernmid+i,l) = kernval;
          ktmp(kernmid+j,kernmid-i,l) = kernval;
          ktmp(kernmid-j,kernmid-i,l) = kernval;
          } // j
        theta_offres = 2.*M_PI * tau * (double)(l+minindex_f)*freqinc;
        kernval = std::polar(1.,theta_offres);
    
        ktmp(kernmid+i,0,l) = kernval;
        ktmp(kernmid-i,0,l) = kernval;
        ktmp(0,kernmid+i,l) = kernval;
        ktmp(0,kernmid-i,l) = kernval;
        } // i
      ktmp(0,0,l) = kernval;
      } // l

    for(l = 0; l < ktmp.size(2); l++) {
      auto ktmp_slice = ktmp.slice(S::all(), S::all(), S(l)).copy();
      FFTW::fft2(ktmp_slice, ktmp_slice, FFTW_BACKWARD);
      ktmp.slice(S::all(), S::all(), S(l)) = ktmp_slice;
    }
    

    if (side == 0)
      alphanorm = 1.;
    else
      alphanorm = M_PI/double(side);

    /* taper the kernel in image space (only the side part) */
    for (l=0; l < (*kern).size(2); l++) {
      for (i=0; i<kernmid; i++) {
        i2 = i*i;
        for (j=i; j<kernmid; j++) {
          rad = std::sqrt((double)(i2+j*j));

          if (rad <= (*kernrad)(l)-side) {
            kernval = ktmp(kernmid+i,kernmid+j,l);
            (*kern)(i,j,l) = kernval;
            (*kern)(j,i,l) = kernval;
            }
          else if (rad < (*kernrad)(l)) { // hanning window over "side" pixels at edge of kernel
            alpha = (rad+side-(*kernrad)(l))*alphanorm;
            kernval = ktmp(kernmid+i,kernmid+j,l)*0.5*(1.0+std::cos(alpha));
            (*kern)(i,j,l) = kernval;
            (*kern)(j,i,l) = kernval;
            }
      } } } // i j l

    /***************************/
    /* Next section for kphase */
    /***************************/
    double tauval, hann, omega, Nh;
    long ih0, ih02, ih;

    kphase = (std::complex<double>)(1.); /* TCC: padding one on the outside of sampling circle*/

    ih0 = kphase.size(0)/2;
    ih02 = ih0*ih0;
    ih = (double)kphase.size(0)*crdmag(i100);
    Nh = (double)std::max(long(1),ih0- ih);
    alphanorm = M_PI/double(Nh);

    /* normalize k-space coordinates */
    knorm = (double)(ih0)/0.5;
    for (i=0; i<(int)crdmag.size(0); i++)
      kn[i] = crdmag(i)*knorm;// kn goes from 0 to ih = kernel width/2

    /* work through an octant
    * ASSUME that crdmag increases monotonically! */

    for (i=0; i<ih0; i++) {
      i2 = i*i;
      for (j=i; j<ih0; j++) {
        r2 = i2 + j*j;
        rk = std::sqrt((double)r2);
        rad = std::min(double(ih),rk);// rad goes from 0 to ih
        // find k that gives us kn[k] = rk
        if (rad>kn[0]) {
          for (k=0; (rk>kn[k] && k < i100); k++);
            k--;
          }
        else
          k=0;

        /* interpolate between sampled kr (kernel & traj) */
        if (k >= i100-1)
          kdel = 0;
        else if (kn[k+1] > kn[k])
          kdel = (rk-kn[k])/(kn[k+1]-kn[k]);
        else
          kdel = 0;

        tauval = dwell*((double)(k) + kdel);

        for (k=0; k<kphase.size(2); k++) { // slice
          for (nb=0; nb<kphase.size(5); nb++) { //freq band
            // Mean Freq - this is for demodulation of the input data before CG loop
            omega = 2.*M_PI*fmap_mean(k,nb);

            for (m=0; m<kphase.size(3); m++) { // TE
              kv = std::polar(1.,omega*(tauval+tstart[m]));

              /* 8-fold symmetry! */
              kphase(ih0+i,ih0+j,k,m,0,nb) = kv;
              kphase(ih0+i,ih0-j,k,m,0,nb) = kv;
              kphase(ih0-i,ih0+j,k,m,0,nb) = kv;
              kphase(ih0-i,ih0-j,k,m,0,nb) = kv;
              kphase(ih0+j,ih0+i,k,m,0,nb) = kv;
              kphase(ih0+j,ih0-i,k,m,0,nb) = kv;
              kphase(ih0-j,ih0+i,k,m,0,nb) = kv;
              kphase(ih0-j,ih0-i,k,m,0,nb) = kv;

              // FAT
              kernval  = 0.;
              for (int sp=0; sp<(int)peaks.size(1); ++sp)
                kernval  += std::polar(peaks(0, sp),2.*M_PI*(peaks(1, sp)) *(tstart[m]+tauval) );
              /* JV: putting mean freq demodulation into fat kphase as well, updating blurk and deblurk */
              kv *= kernval;

            /* 8-fold symmetry! */
              kphase(ih0+i,ih0+j,k,m,1,nb) = kv;
              kphase(ih0+i,ih0-j,k,m,1,nb) = kv;
              kphase(ih0-i,ih0+j,k,m,1,nb) = kv;
              kphase(ih0-i,ih0-j,k,m,1,nb) = kv;
              kphase(ih0+j,ih0+i,k,m,1,nb) = kv;
              kphase(ih0+j,ih0-i,k,m,1,nb) = kv;
              kphase(ih0-j,ih0+i,k,m,1,nb) = kv;
              kphase(ih0-j,ih0-i,k,m,1,nb) = kv;
          } } } // m nb k
      } } // i j

    // Fill in edges of array
    tauval = dwell*(double)(i100);
    for (k=0; k<kphase.size(2); k++) { // slice
      for (nb=0; nb<kphase.size(5); nb++) { //freq band
        // Mean Freq - this is for demodulation of the input data before CG loop
        omega = 2.*M_PI*fmap_mean(k,nb);
        for (m=0; m<kphase.size(3); m++) { // TE
          // Water
          kv = std::polar(1.,omega*(tauval+tstart[m]));
          kphase(0,0,k,m,0,nb) = kv;
          for (i=0; i<ih0; i++) {
            kphase(ih0+i,0,k,m,0,nb) = kv;
            kphase(ih0-i,0,k,m,0,nb) = kv;
            kphase(0,ih0+i,k,m,0,nb) = kv;
            kphase(0,ih0-i,k,m,0,nb) = kv;
            }
          // FAT
          kernval  = 0.;
          for (int sp=0; sp<(int)peaks.size(1); ++sp)
            kernval  += std::polar(peaks(0, sp),2.*M_PI*(peaks(1, sp)) *(tstart[m]+tauval) );
          kv *= kernval;
          kphase(0,0,k,m,1,nb) = kv;
          for (i=0; i<ih0; i++) {
            kphase(ih0+i,0,k,m,1,nb) = kv;
            kphase(ih0-i,0,k,m,1,nb) = kv;
            kphase(0,ih0+i,k,m,1,nb) = kv;
            kphase(0,ih0-i,k,m,1,nb) = kv;
            }
      } } } // k nb m

    /***************************/
    /* Next section for tephase */
    /* TE-dependent phase only */
    /***************************/

    for (i=0; i<tephase.size(0); i++) {
      for (j=0; j<tephase.size(1); j++) {
        for (k=0; k<tephase.size(2); k++) {
          for (m=0; m<tephase.size(3); m++) { // TE
            for (nb=0; nb<tephase.size(4); nb++) { // freq band
              tephase(i,j,k,m,nb) = std::polar(1.,2.*M_PI*tstart[m]*fmap_residual(i,j,k,nb));
      } } } } } // i j k m nb
      
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    // std::cout << "Kern stage1 took " << elapsed.count() << "ms \n";
    // fflush(stdout);  
    return(0);
} // kern_stage1_mltbnd

PYBIND11_MODULE(KernelBuilder, m) {
    m.doc() = "Pybind11 binding for MRIRecon::DeblurFX C++ class.";

    // Bind kern_stage1_mltbnd function
    m.def("generate_kernels",
      [](const Array<double> &crdin, 
         const Array<double> &fmap_residual,
         const Array<double> &fmap_mean, 
         const Array<double> &peaks, 
         const Array<double> &echo_times,
         double b0, double dwell, double gamma,
         int side, double freqinc) 
      {
        // Prepare crdmag array
        Array<double> crdmag(crdin.size(1));

        // Prepare kernel and kernel radius pointers
        Array<std::complex<double> > *kern_table_ptr = nullptr;
        Array<double> *kernrad_ptr = nullptr;

        // Prepare kphase and tephase arrays
        Array<std::complex<double> > kphase(
            fmap_residual.size(0), fmap_residual.size(1), fmap_residual.size(2), 
            echo_times.size(0), 2, fmap_residual.size(3));
        Array<std::complex<double> > tephase(
            fmap_residual.size(0), fmap_residual.size(1), fmap_residual.size(2), 
            echo_times.size(0), fmap_residual.size(3));
        Array<long> mindex(2);

        // Extract echo times
        double te1 = echo_times(0);
        double te2 = echo_times.size(0) > 1 ? echo_times(1) : te1;
        double te3 = echo_times.size(0) > 2 ? echo_times(2) : te2;

        kern_stage1_mltbnd(
            const_cast<Array<double>&>(crdin), &kern_table_ptr, &kernrad_ptr,
            kphase, tephase, 
            const_cast<Array<double>&>(fmap_residual), 
            const_cast<Array<double>&>(fmap_mean), 
            const_cast<Array<double>&>(peaks), 
            mindex, crdmag,
            b0, dwell, gamma, side, freqinc, te1, te2, te3
        );

        // Copy results to return values
        Array<std::complex<double> > kern_table = *kern_table_ptr;
        Array<double> kernrad = *kernrad_ptr;

        // Clean up
        delete kern_table_ptr;
        delete kernrad_ptr;

        return py::make_tuple(kern_table, kernrad, kphase, tephase, mindex);
      },
      py::arg("crdin"),
      py::arg("fmap_residual"),
      py::arg("fmap_mean"),
      py::arg("peaks"),
      py::arg("echo_times"),
      py::arg("b0"),
      py::arg("dwell"),
      py::arg("gamma"),
      py::arg("side"),
      py::arg("freqinc"),
      "Multi-band kernel stage 1 function"
    );
}
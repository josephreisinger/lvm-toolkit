/*
   Copyright 2010 Joseph Reisinger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
// GNU samplib, adapted from Hal Daume's HBC, updated with God knows what else
//
// Notes:
//
// (Joseph Reisinger) I replaced the underlying random number generator here
// with something a little more sexy. I'm also slowly converting the vector
// operations to use ublas
#ifndef STATS_H_
#define STATS_H_

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace boost::numeric;

// The random seed
DECLARE_int32(random_seed);

double digamma(double x);
double trigamma(double x);
double invdigamma(double x);

void advnst(long k);
double genbet(double aa,double bb);
double genchi(double df);
double genexp(double av);
double genf(double dfn, double dfd);
double gengam(double a,double r);
void genmn(double *parm,double *x,double *work);
void genmul(long n,double *p,long ncat,long *ix);
double gennch(double df,double xnonc);
double gennf(double dfn, double dfd, double xnonc);
double gennor(double av,double sd);
double genunf(double low,double high);
void getsd(long *iseed1,long *iseed2);
void gscgn(long getset,long *g);
long ignbin(long n,double pp);
long ignnbn(long n,double p);
long ignpoi(double mu);
void initgn(long isdtyp);
long mltmod(long a,long s,long m);
void phrtsd(char* phrase,long* seed1,long* seed2);
double ranf(void);
void setall(long iseed1,long iseed2);
void setant(long qvalue);
void setgmn(double *meanv,double *covm,long p,double *parm);
void setsd(long iseed1,long iseed2);
double sexpo(void);
double sgamma(double a);
double snorm(void);


// These are adapted from Hal Daume's HBC:

template <typename vec_t>
void normalizeLog(vec_t& x);

template <typename vec_t>
inline int sample_multinomial(const vec_t& d);
template <typename vec_t>
int sample_log_multinomial(vec_t& d);

inline double sample_normal();
inline double sample_gaussian(double mean, double si2);
ublas::vector<double> sample_gaussian_vector(double mean, double si2, unsigned dim);
inline double sample_uniform();
template <typename vec_t>
ublas::vector<double> sample_vmf(const vec_t& mu, double kappa);
template <typename vec_t>
ublas::vector<double> sample_spherical_gaussian(const vec_t& mean, double si2);
ublas::vector<double> sample_sym_dirichlet(double alpha, unsigned dim);

template <typename vec_t>
double logp_sym_dirichlet(const vec_t& value, double alpha);
double logp_gamma(double x, double a, double b);

double approx_log_iv(double nu, double z);

// TODO(jsr): are these implemented somewhere in BLAS?
// Return the matrix reflecting unit vector C{p} to unit vector C{q}.
ublas::matrix<double> reflection_matrix(const ublas::vector<double>& p, const ublas::vector<double>& q);

// Return the matrix rotating unit vector C{f} to unit vector C{t}.
ublas::matrix<double> rotation_matrix(const ublas::vector<double>& f, const ublas::vector<double>& t);

#include "stats.cc" // include implementation here for template stuff


#endif  //  STATS_H_

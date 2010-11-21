// This implements the multinomial sampler for hBernoulliLDA, which constrains the
// topics to be a tree with fixed depth.


#ifndef SAMPLE_BERN_NCRP_H_
#define SAMPLE_BERN_NCRP_H_

//#include "crp-base.h"

class hBernoulliLDA : public hBernoulliLDABase {
 public:
  hBernoulliLDA(unsigned L, double gamma);
  ~hBernoulliLDA() { /* TODO: free memory! */ }

  string current_state();
 private:
  void resample_posterior();
  void resample_posterior_z_for(unsigned d);

  double compute_log_likelihood();
};

#endif  // SAMPLE_BERN_NCRP_H_

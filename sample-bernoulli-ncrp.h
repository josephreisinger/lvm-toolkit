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

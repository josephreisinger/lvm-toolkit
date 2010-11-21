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
// This implements (I think) the GEM distributed hLDA, which has an infinitely
// deep tree, where each node can have (possibly) infinitely many branches.
// There are some issues with implementing this sampler that I (and others)
// have raised on the Princeton topic models list, but so far no one has come
// up with any answers. In any case the sampler /seems/ to work just fine.


#ifndef SAMPLE_GEM_NCRP_H_
#define SAMPLE_GEM_NCRP_H_

#include <string>

#include "ncrp-base.h"

// The GEM(m, \pi) distribution hyperparameter m, controls the "proportion of
// general words relative to specific words"
DECLARE_double(gem_m);

// The GEM(m, \pi) hyperparameter \pi: reflects how strictly we expect the
// documents to adhere to the m proportions.
DECLARE_double(gem_pi);

class GEMNCRP : public NCRPBase {
 public:
  GEMNCRP(double m, double pi);
  ~GEMNCRP() { /* TODO: free memory! */ }

  string current_state();
 private:
  void resample_posterior();
  void resample_posterior_z_for(unsigned d, bool remove);

  double compute_log_likelihood();

 private:
  double _gem_m;
  double _pi;

  unsigned _maxL;
};

#endif  // SAMPLE_GEM_NCRP_H_

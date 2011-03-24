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
// Samples from the Nested Chinese Restaurant Process using multinomials. Input
// data is assumed to be a newline-delimited list of documents, each of which is
// composed of some number of tokens.
//
// This version uses a fixed-depth tree.

#include <math.h>
#include <time.h>

#include "ncrp-base.h"
#include "sample-mult-ncrp.h"

// Performs a single document's level assignment resample step
void FixedDepthNCRP::resample_posterior_z_for(unsigned d, bool remove) {
    VLOG(1) << "resample posterior z for " << d;

    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];

        if (remove) {
            // Remove this document and word from the counts
            _c[d][_z[d][n]]->remove_no_ndsum(w,d);
        }

        vector<double> lp_z_dn;

        unsigned start = FLAGS_ncrp_skip_root ? 1 : 0;
        for (int l = start; l < _L; l++) {
            // check that ["doesnt exist"]->0
            DCHECK(_c[d][l]->nw.find(w) != _c[d][l]->nw.end() || _c[d][l]->nw[w] == 0);
            DCHECK(_c[d][l]->nd.find(d) != _c[d][l]->nd.end() || _c[d][l]->nd[d] == 0);

            lp_z_dn.push_back(log(_eta[w] + _c[d][l]->nw[w]) -
                    log(_eta_sum + _c[d][l]->nwsum) +
                    log(_alpha[l] + _c[d][l]->nd[d]) -
                    log(_alpha_sum + _nd[d]-1));
        }

        // Update the assignment
        // _z[d][n] = SAFE_sample_unnormalized_log_multinomial(&lp_z_dn) + start;
        _z[d][n] = sample_unnormalized_log_multinomial(&lp_z_dn) + start;

        // Update the counts

        // Check to see that the default dictionary insertion works like we
        // expect
        DCHECK(_c[d][_z[d][n]]->nw.find(w) != _c[d][_z[d][n]]->nw.end()
                || _c[d][_z[d][n]]->nw[w] == 0);
        DCHECK(_c[d][_z[d][n]]->nd.find(d) != _c[d][_z[d][n]]->nd.end()
                || _c[d][_z[d][n]]->nd[d] == 0);

        _c[d][_z[d][n]]->add_no_ndsum(w,d);
    }
}

// Resamples the level allocation variables z_{d,n} given the path assignments
// c and the path assignments given the level allocations
void FixedDepthNCRP::resample_posterior() {
  CHECK_GT(_lV, 0);
  CHECK_GT(_lD, 0);
  CHECK_GT(_L, 0);

  if (FLAGS_ncrp_update_hyperparameters) {
    // TODO:: for now this is basically assuming a uniform distribution
    // over hyperparameters (bad!) with a truncated gaussian as the proposal
    // distribution
    // double old_ll = _ll;
    // double old_eta = _eta;
    // double old_alpha = _alpha;
    // double old_gamma = _gamma;
    // _eta += sample_gaussian() / 1000.0;
    // _alpha += sample_gaussian() / 1.0;
    // _gamma += sample_gaussian() / 1000.0;
    // _eta = max(0.00000001, _eta);
    // _alpha = max(0.00001, _alpha);
    // _gamma = max(0.00001, _gamma);

    // double new_ll = compute_log_likelihood();
    // double k = log(sample_uniform());

    // if (k < new_ll - old_ll) {
    //   _ll = new_ll;
    // } else {
    //   _eta = old_eta;
    //   _alpha = old_alpha;
    //   _gamma = old_gamma;
    //   _ll  = old_ll;
    // }
  }
  // Interleaved version
  for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
      unsigned d = d_itr->first;

      if (FLAGS_ncrp_depth > 1 && FLAGS_ncrp_max_branches != 1) {
          resample_posterior_c_for(d);
      }
      // BAD BAD: skipping check
      // DCHECK(tree_is_consistent());
      for (int z = 0; z < FLAGS_ncrp_z_per_iteration; z++) {
          resample_posterior_z_for(d, true);
          // DCHECK(tree_is_consistent());
      }
  }

  print_summary();
}


double FixedDepthNCRP::compute_log_likelihood() {
    // VLOG(1) << "compute log likelihood " ;
    // Compute the log likelihood for the tree
    double log_lik = 0;
    _unique_nodes = 0;  // recalculate the tree size
    // Compute the log likelihood of the tree
    deque<CRP*> node_queue;
    node_queue.push_back(_ncrp_root);

    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();

        _unique_nodes += 1;

        if (current->tables.size() > 0) {
            for (int i = 0; i < current->tables.size(); i++) {
                CHECK_GT(current->tables[i]->ndsum, 0);
                if (FLAGS_ncrp_m_dependent_gamma) {
                    log_lik += log(current->tables[i]->ndsum) - log(current->ndsum * (_gamma + 1) - 1);
                } else {
                    log_lik += log(current->tables[i]->ndsum) - log(current->ndsum+_gamma-1);
                }
            }

            node_queue.insert(node_queue.end(), current->tables.begin(),
                    current->tables.end());
        }
    }

    // Compute the log likelihood of the level assignments (correctly?)
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        double lndsumd = log(_nd[d]+_alpha_sum);
        for (int n = 0; n < _D[d].size(); n++) {
            // likelihood of drawing this word
            unsigned w = _D[d][n];
            log_lik += log(_c[d][_z[d][n]]->nw[w]+_eta[w]) -
                log(_c[d][_z[d][n]]->nwsum+_eta_sum);
            // likelihood of the topic?
            log_lik += log(_c[d][_z[d][n]]->nd[d]+_alpha[_z[d][n]]) - lndsumd;
        }
    }
    return log_lik;
}

string FixedDepthNCRP::current_state() {
  _filename = FLAGS_ncrp_datafile;
  _filename += StringPrintf("-L%d-gamma%.2f-alpha%.2f-eta%.2f-eds%.2f-zpi%d-best.data",
                            _L, _gamma, _alpha_sum / (double)_alpha.size(),
                            _eta_sum / (double)_eta.size(),
                            FLAGS_ncrp_eta_depth_scale,
                            FLAGS_ncrp_z_per_iteration);

  return StringPrintf(
      "ll = %f (%f at %d) %d alpha = %f eta = %f gamma = %f L = %d",
      _ll, _best_ll, _best_iter, _unique_nodes,
      _alpha_sum / (double)_alpha.size(),
      _eta_sum / (double)_eta.size(), _gamma, _L);
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    FixedDepthNCRP h = FixedDepthNCRP();
    h.load_data(FLAGS_ncrp_datafile);

    h.run();
}

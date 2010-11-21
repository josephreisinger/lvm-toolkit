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

#include "crp-base.h"
#include "sample-mult-ncrp.h"


hLDA::hLDA(unsigned L, double gamma)
    : hLDABase(L, gamma) { }

// Performs a single document's level assignment resample step
void hLDA::resample_posterior_z_for(unsigned d) {
  for (int n = 0; n < _D[d].size(); n++) {
    unsigned w = _D[d][n];
    // Remove this document and word from the counts
    _c[d][_z[d][n]]->nw[w] -= 1;  // # of words in topic z equal to w
    _c[d][_z[d][n]]->nd[d] -= 1;  // # of words in doc d with topic z
    _c[d][_z[d][n]]->nwsum -= 1;  // # of words in topic z
    _ndsum[d]             -= 1;  // # of words in doc d

    CHECK_GE(_c[d][_z[d][n]]->nwsum, 0);
    CHECK_GE(_c[d][_z[d][n]]->nw[w], 0);
    //CHECK_GT(_ndsum[d], 0);

    vector<double> lp_z_dn;

    unsigned total_nd = 0;
    for (int l = 0; l < _L; l++) {
      // check that ["doesnt exist"]->0
      DCHECK(_c[d][l]->nw.find(w) != _c[d][l]->nw.end() || _c[d][l]->nw[w] == 0);
      DCHECK(_c[d][l]->nd.find(d) != _c[d][l]->nd.end() || _c[d][l]->nd[d] == 0);
      total_nd += _c[d][l]->nd[d];

      lp_z_dn.push_back(log(_eta[w] + _c[d][l]->nw[w]) -
                        log(_eta_sum + _c[d][l]->nwsum) +
                        log(_alpha[l] + _c[d][l]->nd[d]) -
                        log(_alpha_sum + _ndsum[d]));
    }
    DCHECK_EQ(total_nd, _ndsum[d]);

    // Update the assignment
    _z[d][n] = sample_unnormalized_log_multinomial(&lp_z_dn);

    // Update the counts

    // Check to see that the default dictionary insertion works like we
    // expect
    DCHECK(_c[d][_z[d][n]]->nw.find(w) != _c[d][_z[d][n]]->nw.end()
           || _c[d][_z[d][n]]->nw[w] == 0);
    DCHECK(_c[d][_z[d][n]]->nd.find(d) != _c[d][_z[d][n]]->nd.end()
           || _c[d][_z[d][n]]->nd[d] == 0);

    _c[d][_z[d][n]]->nw[w] += 1;  // number of words in topic z equal to w
    _c[d][_z[d][n]]->nd[d] += 1;  // number of words in doc d with topic z
    _c[d][_z[d][n]]->nwsum += 1;  // number of words in topic z
    _ndsum[d]              += 1;  // number of words in doc d
  }

}

// Resamples the level allocation variables z_{d,n} given the path assignments
// c and the path assignments given the level allocations
void hLDA::resample_posterior() {
  CHECK_GT(_lV, 0);
  CHECK_GT(_lD, 0);
  CHECK_GT(_L, 0);

  if (FLAGS_crp_update_hyperparameters) {
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

      if (FLAGS_depth > 1 && FLAGS_crp_max_branches != 1) {
          resample_posterior_c_for(d);
      }
      // BAD BAD: skipping check
      // DCHECK(tree_is_consistent());
      for (int z = 0; z < FLAGS_crp_z_per_iteration; z++) {
          resample_posterior_z_for(d);
          // DCHECK(tree_is_consistent());
      }
  }
}


double hLDA::compute_log_likelihood() {
  // Compute the log likelihood for the tree
  double log_lik = 0;
  _unique_nodes = 0;  // recalculate the tree size
  // Compute the log likelihood of the tree
  deque<CRP*> node_queue;
  node_queue.push_back(_crp_root);

  while (!node_queue.empty()) {
    CRP* current = node_queue.front();
    node_queue.pop_front();

    _unique_nodes += 1;

    if (current->tables.size() > 0) {
      for (int i = 0; i < current->tables.size(); i++) {
        CHECK_GT(current->tables[i]->m, 0);
        if (FLAGS_crp_m_dependent_gamma) {
          log_lik += log(current->tables[i]->m) - log(current->m * (_gamma + 1) - 1);
        } else {
          log_lik += log(current->tables[i]->m) - log(current->m+_gamma-1);
        }
      }

      node_queue.insert(node_queue.end(), current->tables.begin(),
                        current->tables.end());
    }
  }

  // Compute the log likelihood of the level assignments (correctly?)
  for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
      unsigned d = d_itr->first;

    double lndsumd = log(_ndsum[d]+_alpha_sum);
    for (int v = 0; v < _D[d].size(); v++) {
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

string hLDA::current_state() {
  _filename = FLAGS_crp_datafile;
  _filename += StringPrintf("-L%d-gamma%f-alpha%f-eta%f-zpi%d-best.data",
                            _L, _gamma, _alpha_sum / (double)_alpha.size(),
                            _eta_sum / (double)_eta.size(),
                            FLAGS_crp_z_per_iteration);

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

    hLDA h = hLDA(FLAGS_depth, FLAGS_crp_gamma);
    h.load_data(FLAGS_crp_datafile);
    h.initialize();

    h.run(FLAGS_gibbs_iterations);
}

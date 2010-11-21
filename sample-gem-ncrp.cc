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

#include "ncrp-base.h"
#include "sample-gem-ncrp.h"

// The GEM(m, \pi) distribution hyperparameter m, controls the "proportion of
// general words relative to specific words"
DEFINE_double(gem_m,
              0.1,
              "m reflects the proportion of general words to specific words");

// The GEM(m, \pi) hyperparameter \pi: reflects how strictly we expect the
// documents to adhere to the m proportions.
DEFINE_double(gem_pi,
              0.1,
              "reflects our confidence in the setting m");

GEMNCRP::GEMNCRP(double m, double pi)
    : _gem_m(m), _pi(pi), _maxL(0) {
  _L = 3;  // for now we need an initial depth (just for the first data pt)
  _maxL = _L;

  CHECK_GE(_gem_m, 0.0);
  CHECK_LE(_gem_m, 1.0);
}


string GEMNCRP::current_state() {
  // HACK: put this in here for now since it needs to get updated whenever
  // maxL changes
  _filename = FLAGS_ncrp_datafile;
  _filename += StringPrintf("-L%d-m%f-pi%f-gamma%f-eta%f-zpi%d-best.data",
                            _maxL, _gem_m, _pi, _gamma,
                            _eta_sum / (double)_eta.size(),
                            FLAGS_ncrp_z_per_iteration);

  return StringPrintf(
      "ll = %f (%f at %d) %d m = %f pi = %f eta = %f gamma = %f L = %d",
      _ll, _best_ll, _best_iter, _unique_nodes, _gem_m, _pi,
      _eta_sum / (double)_eta.size(), _gamma,
      _maxL);
}

// Performs a single document's level assignment resample step There is some
// significant subtly to this compared to the fixed depth version. First, we no
// longer have the guarantee that all documents are attached to leaves. Some
// documents may now stop at interior nodes of our current tree. This is ok,
// since the it just means that we haven't assigned any words to lower levels,
// and hence we haven't needed to actually assert what the path is. Anyway,
// since the level assignments can effectively change length here, we need to
// get more child nodes from the nCRP on the fly.
void GEMNCRP::resample_posterior_z_for(unsigned d, bool remove) {
  // CHECK_EQ(_L, -1); // HACK to make sure we're not using _L
  CHECK(false) << "implement remove";

    CHECK(!FLAGS_ncrp_skip_root);
  for (int n = 0; n < _D[d].size(); n++) {  // loop over every word
    unsigned w = _D[d][n];
    // Compute the new level assignment #
    _c[d][_z[d][n]]->nw[w] -= 1;  // number of words in topic z equal to w
    _c[d][_z[d][n]]->nd[d] -= 1;  // number of words in doc d with topic z
    _c[d][_z[d][n]]->nwsum -= 1;  // number of words in topic z
    _nd[d] -= 1;  // number of words in doc d


    CHECK_GE(_c[d][_z[d][n]]->nwsum, 0);
    CHECK_GE(_c[d][_z[d][n]]->nw[w], 0);
    CHECK_GE(_nd[d], 0);
    CHECK_GT(_c[d][_z[d][n]]->ndsum, 0);

    // ndsum_above[k] is #[z_{d,-n} >= k]
    vector<unsigned> ndsum_above;
    ndsum_above.resize(_c[d].size());
    ndsum_above[_c[d].size()-1] = _c[d].back()->nd[d];
    // LOG(INFO) << "ndsum_above[" << _c[d].size()-1 << "] = "
    //           << ndsum_above.back();
    for (int l = _c[d].size()-2; l >= 0; l--) {
      // TODO:: optimize this
      ndsum_above[l] = _c[d][l]->nd[d] + ndsum_above[l+1];
      // LOG(INFO) << "ndsum_above[" << l << "] = " << ndsum_above[l];
    }

    // Here we assign probabilities to all the "finite" options, e.g. all
    // the levels up to the current maximum level for this document. TODO::
    // this can be optimized quite extensively
    vector<double> lposterior_z_dn;
    double lp_z_dn_sum = 0;
    double V_j_sum = 0;
    unsigned total_nd = 0;
    for (int l = 0; l < _c[d].size(); l++) {
      // check that ["doesnt exist"]->0
      // DCHECK(_c[d][l]->nw.find(w) != _c[d][l]->nw.end() || _c[d][l]->nw[w] == 0);
      // DCHECK(_c[d][l]->nd.find(d) != _c[d][l]->nd.end() || _c[d][l]->nd[d] == 0);
      total_nd += _c[d][l]->nd[d];

      double lp_w_dn = log(_eta[w] + _c[d][l]->nw[w]) -
          log(_eta_sum + _c[d][l]->nwsum);
      double lp_z_dn = log(_pi*(1-_gem_m) + _c[d][l]->nd[d]) -
          log(_pi + ndsum_above[l]) + V_j_sum;

      lposterior_z_dn.push_back(lp_w_dn + lp_z_dn);

      if (l < _c[d].size()-1) {
        V_j_sum += log(_gem_m*_pi + ndsum_above[l+1]) -
            log(_pi + ndsum_above[l]);
      }
      lp_z_dn_sum = addLog(lp_z_dn_sum, lp_z_dn);
    }
    // DCHECK_EQ(total_nd, _nd[d]);


    // If the "new" entry is sampled, we have to determine actually at what
    // level to attach the word. This is done by repeatedly sampling from a
    // binomial, that results in sampling from the GEM.
    // XXX: this is from the earlier version of the paper...
    unsigned new_max_level = _c[d].size()+1;
    while (sample_uniform() < _gem_m) {
      new_max_level += 1;
    }

    // The next big block computes lp_w_dn for the new level assignment
    // (hopefully without taking too much computation).
    double lp_w_dn = 0;
    CRP* new_leaf = _c[d].back();

    // if there are things attached below here
    if (_c[d].back()->tables.size() > 0) {
      // then sample from them

      // We now need to have selected a branch to depth new_max_level
      // Theoretically, we've already done this, since the GEM distribution and
      // nCRP draws are infinite. However, obviously, this is not the case.
      // Instead, we need to proceed as if we alread know down what branch to
      // extend the level allocations. We do this by post-hoc selecting a
      // subtree path starting at the old end of c[d] and augmenting it to get
      // a path to new_max_level

      // Keep track of the removed counts for computing the likelihood of
      // the data
      // HACK: we only remove one at a time, so this can be optimized....
      LevelWordToCountMap nw_removed;
      LevelToCountMap     nwsum_removed;
      nw_removed[_z[d][n]][w] += 1;
      nwsum_removed[_z[d][n]] += 1;

      vector<double> lp_c_d;  // log-probability of this branch c_d
      vector<CRP*> c_d;  // the actual branch c_d
      calculate_path_probabilities_for_subtree(_c[d].back(), d, new_max_level,
                                               nw_removed, nwsum_removed,
                                               &lp_c_d, &c_d);

      // Choose a new leaf node
      int index = sample_unnormalized_log_multinomial(&lp_c_d);

      new_leaf = c_d[index];  // keep a pointer around for later

      // If we choose to create a new branch at a level less than our
      // desired level, then it can have no words already added, hence the
      // word probability is the default.
      if (c_d[index]->level < new_max_level) {
        // then there are no words below here
        lp_w_dn = log(_eta[w]) - log(_eta_sum);
      } else {
        // TODO: this is slow
        // replay back to level level
        CRP* current = c_d[index];
        while (current->level > new_max_level-1) {
          CHECK(false);  // shouldn't get here
          current = current->prev[0];
        }
        lp_w_dn = log(_eta[w] + current->nw[w]) - log(_eta_sum + current->nwsum);
      }

    } else {  // then there are no words below here
      lp_w_dn = log(_eta[w]) - log(_eta_sum);
    }

    // Add the probability of sampling a new level, defined as the above
    lposterior_z_dn.push_back(log(1.0-exp(lp_z_dn_sum))+lp_w_dn);
    // LOG(INFO) << lposterior_z_dn[lposterior_z_dn.size()-1];

    // Update the assignment
    _z[d][n] = sample_unnormalized_log_multinomial(&lposterior_z_dn);


    CHECK_LE(_z[d][n], _c[d].size());

    // Take care of the m assignment if we shrink
    // TODO: this is probably inefficient
    if (_z[d][n] < _c[d].size()) {
      // detach this document from the lower nodes, if nd[d] is zero
      for (unsigned l = _c[d].size()-1; l > _z[d][n]; l--) {
        if (_c[d][l]->nd[d] == 0) {
          CHECK_GT(_c[d][l]->ndsum, 0);
          _c[d][l]->ndsum -= 1;
          _c[d].pop_back();
          CHECK(_c[d].size() == l);
        } else {
          break;  // break off early to allow nd=1 -> nd=0 -> nd=1
        }
      }
    } else if (_z[d][n] == _c[d].size()) {  // sampled the new entry
      _z[d][n] = new_max_level-1;

      // Support the new level by adding to the CRP tree
      unsigned old_size = _c[d].size();
      graft_path_at(new_leaf, &_c[d], new_max_level);

      // Add to all the document counts
      for (int l = old_size; l < _c[d].size(); l++) {
        _c[d][l]->ndsum += 1;
      }
    }


    // Update the maximum depth if necessary
    if (_z[d][n] > _maxL) {
      _maxL = _z[d][n];
    }

    // Update the counts

    // Check to see that the default dictionary insertion works like we
    // expect
    // DCHECK(_c[d][_z[d][n]]->nw.find(w) != _c[d][_z[d][n]]->nw.end() || _c[d][_z[d][n]]->nw[w] == 0);
    // DCHECK(_c[d][_z[d][n]]->nd.find(d) != _c[d][_z[d][n]]->nd.end() || _c[d][_z[d][n]]->nd[d] == 0);

    _c[d][_z[d][n]]->nw[w] += 1;  // number of words in topic z equal to w
    _c[d][_z[d][n]]->nd[d] += 1;  // number of words in doc d with topic z
    _c[d][_z[d][n]]->nwsum += 1;  // number of words in topic z
    _nd[d] += 1;  // number of words in doc d

    CHECK_GT(_c[d][_z[d][n]]->ndsum, 0);

    // TODO:: delete excess leaves If we reassigned the levels we might
    // have caused some of the lower nodes in the tree to become empty. If so,
    // then we should remove them. TODO:: how to do this in a way that
    // updates all the things pointng to that node (without having to keep
    // track of that?) This is probably just purely an efficiency thing, so we
    // only need to worry if memory gets to be an issue
  }
}
// Resamples the level allocation variables z_{d,n} given the path assignments
// c and the path assignments given the level allocations
void GEMNCRP::resample_posterior() {
  CHECK_GT(_lV, 0);
  CHECK_GT(_lD, 0);
  CHECK_GT(_L, 0);

  // Interleaved version
  _maxL = 0;
  for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
      unsigned d = d_itr->first;

    // LOG(INFO) <<  "  resampling document " <<  d;
    resample_posterior_c_for(d);
    // DCHECK(tree_is_consistent());
    for (int z = 0; z < FLAGS_ncrp_z_per_iteration; z++) {
      resample_posterior_z_for(d, true);
      // DCHECK(tree_is_consistent());
    }
  }
}

double GEMNCRP::compute_log_likelihood() {
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

    // Should never have words attached but no documents
    CHECK(!(current->ndsum == 0 && current->nwsum > 0));

    if (current->tables.size() > 0) {
      for (int i = 0; i < current->tables.size(); i++) {
        if (current->tables[i]->ndsum > 0) {  // TODO: how to delete nodes?
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

    // TODO: inefficient?
    // ndsum_above[k] is #[z_{d,-n} >= k]
    vector<unsigned> ndsum_above;
    for (int l = 0; l < _c[d].size(); l++) {
      // TODO: optimize this
      ndsum_above.push_back(0);
      // count the total words attached for this document below here
      for (int ll = l; ll < _c[d].size(); ll++) {
        ndsum_above[l] += _c[d][ll]->nd[d];
      }
      // LOG(INFO) << "ndsum_above[" << l << "] = " << ndsum_above[l];
    }

    for (int n = 0; n < _D[d].size(); n++) {
      // likelihood of drawing this word
      unsigned w = _D[d][n];
      log_lik += log(_c[d][_z[d][n]]->nw[w]+_eta[w]) -
          log(_c[d][_z[d][n]]->nwsum+_eta_sum);

      // likelihood of the topic?
      // TODO: this is heinously inefficient
      double V_j_sum = 0;
      for (int l = 0; l < _z[d][n]; l++) {
        if (l < _c[d].size()-1) {
          V_j_sum += log(_gem_m*_pi + ndsum_above[l+1]) -
              log(_pi + ndsum_above[l]);
        }
      }

      log_lik += log((1-_gem_m)*_pi + _c[d][_z[d][n]]->nd[d]) -
          log(_pi + ndsum_above[_z[d][n]]) + V_j_sum;
    }
  }
  return log_lik;
}

int main(int argc, char **argv) {
  GEMNCRP h = GEMNCRP(FLAGS_gem_m, FLAGS_gem_pi);
  h.load_data(FLAGS_ncrp_datafile);

  h.run();
}

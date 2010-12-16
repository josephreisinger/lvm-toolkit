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
// Like sample-fixed-ncrp but reads in the topic structure in a much more
// flexible way (assumes a file containing doc<tab>topic<tab>topic).

#ifndef SAMPLE_PRECOMPUTED_FIXED_NCRP_H_
#define SAMPLE_PRECOMPUTED_FIXED_NCRP_H_

#include <string>
#include <vector>

#include "ncrp-base.h"
#include "sample-fixed-ncrp.h"
#include "sample-precomputed-fixed-ncrp.h"

// Topic assignments file, if we've precomputed the topic assignments
DECLARE_string(topic_assignments_file);

// Number of additional topics to use over the labeled ones
DECLARE_int32(additional_noise_topics);

// Should we cull topics that only have one document?
DECLARE_bool(cull_unique_topics);

// This version differs from the normal GEM sampler in that the tree structure
// is fixed a priori. Hence there is no resampling of c, the path allocations.
class NCRPPrecomputedFixed : public GEMNCRPFixed {
    public:
        NCRPPrecomputedFixed(double m, double pi) : GEMNCRPFixed(m, pi) { }

        // Load the tree structure
        void load_precomputed_tree_structure(const string& filename);

        // Allocate all the documents at once (called for non-streaming)
        void batch_allocation();

        // Allocate a single document; can be called during load for streaming
        void allocate_document(unsigned d);

        // Write out a static dictionary required for decoding Gibbs samples
        void write_dictionary();

        // Write out the Gibbs sample
        void write_data(string prefix);

    protected:
        void add_crp_node(const string& name);

    protected:
          google::dense_hash_map<string, CRP*> _node_to_crp;

          unsigned _total;  // number of total documents on load
          unsigned _missing;  // number of missing documents on load
};

#endif  // SAMPLE_PRECOMPUTED_FIXED_NCRP_H_

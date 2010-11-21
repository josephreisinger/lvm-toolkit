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
#include "ncrp-base.h"
#include "sample-fixed-ncrp.h"
#include "sample-precomputed-fixed-ncrp.h"

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    NCRPPrecomputedFixed h = NCRPPrecomputedFixed(FLAGS_gem_m, FLAGS_gem_pi);
    h.load_data(FLAGS_ncrp_datafile);
    h.load_precomputed_tree_structure(FLAGS_topic_assignments_file);

    h.run();
}

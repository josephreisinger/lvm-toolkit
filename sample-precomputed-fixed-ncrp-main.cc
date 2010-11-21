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

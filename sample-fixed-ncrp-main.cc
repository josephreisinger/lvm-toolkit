#include "ncrp-base.h"
#include "sample-fixed-ncrp.h"

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    GEMNCRPFixed h = GEMNCRPFixed(FLAGS_gem_m, FLAGS_gem_pi);
    h.load_data(FLAGS_ncrp_datafile);
    h.load_tree_structure(FLAGS_tree_structure_file);

    h.run();
}

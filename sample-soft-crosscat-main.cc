#include "sample-soft-crosscat.h"

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    SoftCrossCatMM h = SoftCrossCatMM();
    h.load_data(FLAGS_mm_datafile);
    h.run();
}

#include "sample-crosscat-mm.h"

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    CrossCatMM h = CrossCatMM();
    h.load_data(FLAGS_mm_datafile);
    h.initialize();
    h.run();
}

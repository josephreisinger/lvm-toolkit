#include "ncrp-base.h"
#include "sample-clustered-lda.h"

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    ClusteredLDA h = ClusteredLDA();
    h.initialize();

    h.run();
}

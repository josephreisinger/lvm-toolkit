# XXXXXXXXXX this has to change for 64/32
#INCLUDES = -I include/ -I /projects/nn/joeraii/local_libraries_mastodon/include/
# INCLUDES = -I include/ -I /projects/nn/joeraii/local_libraries/include/
INCLUDES = -I include/ -I /scratch/cluster/joeraii/ncrp/local_libraries/include/
CC = g++
# LDFLAGS = -L/p/lib -L/projects/nn/joeraii/local_libraries_mastodon/lib/ -L/projects/nn/joeraii/local_libraries/lib/ -L/p/lib/
LDFLAGS = -L/p/lib -L/scratch/cluster/joeraii/ncrp/local_libraries/lib/ -L/p/lib/
LIBRARIES = -lglog -lgflags -lpthread
EXECUTABLES = sampleSoftCrossCatMixtureModel sampleMultNCRP sampleGEMNCRP sampleFixedNCRP samplePrecomputedFixedNCRP sampleClusteredLDA sampleMixtureModel sampleCrossCatMixtureModel
OBJECTS = dSFMT.o strutil.o gibbs-base.o ncrp-base.o sample-clustered-lda.o sample-precomputed-fixed-ncrp.o sample-fixed-ncrp.o sample-gem-ncrp.o sample-mult-ncrp.o sample-crosscat-mm.o  sample-soft-crosscat.o
MTFLAGS = -msse2 -DDSFMT_MEXP=521 -DHAVE_SSE2 --param max-inline-insns-single=1800 --param inline-unit-growth=500 --param large-function-growth=900
CFLAGS = -O3  $(MTFLAGS)  -DUSE_MT_RANDOM
COMPILE = $(CC) $(CFLAGS) $(INCLUDES)
FULLCOMPILE = $(COMPILE) $(LDFLAGS) $(LIBRARIES) 

all:	$(OBJECTS) $(EXECUTABLES)

dSFMT.o: dSFMT-src-2.0/dSFMT.c
	$(COMPILE) -c dSFMT-src-2.0/dSFMT.c -o dSFMT.o
strutil.o: strutil.cc
	$(COMPILE) -c strutil.cc -o strutil.o
gibbs-base.o: gibbs-base.cc 
	$(COMPILE) -c gibbs-base.cc -o gibbs-base.o
ncrp-base.o: ncrp-base.cc 
	$(COMPILE) -c ncrp-base.cc -o ncrp-base.o
sample-fixed-ncrp.o: sample-fixed-ncrp.h sample-fixed-ncrp.cc
	$(COMPILE) -c sample-fixed-ncrp.cc -o sample-fixed-ncrp.o
sample-precomputed-fixed-ncrp.o: sample-precomputed-fixed-ncrp.h sample-precomputed-fixed-ncrp.cc
	$(COMPILE) -c sample-precomputed-fixed-ncrp.cc -o sample-precomputed-fixed-ncrp.o
sample-clustered-lda.o: sample-clustered-lda.h sample-clustered-lda.cc
	$(COMPILE) -c sample-clustered-lda.cc -o sample-clustered-lda.o
sample-gem-ncrp.o: sample-gem-ncrp.cc
	$(COMPILE) -c sample-gem-ncrp.cc -o sample-gem-ncrp.o
sample-mult-ncrp.o: sample-mult-ncrp.cc
	$(COMPILE) -c sample-mult-ncrp.cc -o sample-mult-ncrp.o
sample-crosscat-mm.o: sample-crosscat-mm.cc
	$(COMPILE) -c sample-crosscat-mm.cc -o sample-crosscat-mm.o
sample-soft-crosscat.o: sample-soft-crosscat.cc
	$(COMPILE) -c sample-soft-crosscat.cc -o sample-soft-crosscat.o

sampleMultNCRP: strutil.o dSFMT.o ncrp-base.o gibbs-base.o sample-mult-ncrp.cc sample-mult-ncrp.o
	$(FULLCOMPILE) strutil.o dSFMT.o sample-mult-ncrp.o ncrp-base.o gibbs-base.o -o sampleMultNCRP
sampleGEMNCRP: strutil.o dSFMT.o ncrp-base.o gibbs-base.o sample-gem-ncrp.cc sample-gem-ncrp.o
	$(FULLCOMPILE) strutil.o dSFMT.o sample-gem-ncrp.o ncrp-base.o gibbs-base.o -o sampleGEMNCRP
sampleFixedNCRP: strutil.o dSFMT.o ncrp-base.o gibbs-base.o sample-gem-ncrp.cc sample-fixed-ncrp.o
	$(FULLCOMPILE) sample-fixed-ncrp-main.cc strutil.o dSFMT.o sample-fixed-ncrp.o ncrp-base.o gibbs-base.o -o sampleFixedNCRP
samplePrecomputedFixedNCRP: strutil.o dSFMT.o ncrp-base.o gibbs-base.o sample-gem-ncrp.cc sample-precomputed-fixed-ncrp.o sample-fixed-ncrp.o
	$(FULLCOMPILE) sample-precomputed-fixed-ncrp-main.cc strutil.o dSFMT.o gibbs-base.o sample-fixed-ncrp.o sample-precomputed-fixed-ncrp.o ncrp-base.o -o samplePrecomputedFixedNCRP
sampleClusteredLDA: strutil.o dSFMT.o ncrp-base.o gibbs-base.o sample-clustered-lda.cc 
	$(FULLCOMPILE) sample-clustered-lda-main.cc strutil.o dSFMT.o sample-clustered-lda.o ncrp-base.o gibbs-base.o -o sampleClusteredLDA
sampleCrossCatMixtureModel: strutil.o dSFMT.o gibbs-base.o sample-crosscat-mm.cc 
	$(FULLCOMPILE) sample-crosscat-mm-main.cc strutil.o dSFMT.o sample-crosscat-mm.o gibbs-base.o -o sampleCrossCatMixtureModel
sampleSoftCrossCatMixtureModel: strutil.o dSFMT.o gibbs-base.o sample-soft-crosscat.o sample-soft-crosscat-main.cc
	$(FULLCOMPILE) sample-soft-crosscat-main.cc strutil.o dSFMT.o sample-soft-crosscat.o gibbs-base.o -o sampleSoftCrossCatMixtureModel
sampleNonconjugateDP: strutil.o dSFMT.o gibbs-base.o sample-nonconjugate-dp.cc 
	$(FULLCOMPILE) sample-nonconjugate-dp.cc strutil.o dSFMT.o sample-nonconjugate-dp.o gibbs-base.o -o sampleNonconjugateDP

clean:
	-rm -f *.o *.so *.pyc *~ 

deepclean: clean
	-rm -f $(OBJECTS)

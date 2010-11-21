
sample-mm.o: sample-mm.cc
	$(COMPILE) -c sample-mm.cc -o sample-mm.o
sampleMixtureModel: strutil.o dSFMT.o gibbs-base.o sample-mm.cc 
	$(FULLCOMPILE) sample-mm-main.cc strutil.o dSFMT.o sample-mm.o gibbs-base.o -o sampleMixtureModel
# samplevMFDPMixture: strutil.o dSFMT.o sample-vmf-dp-mixture.cc gibbs-base.o
# 	$(FULLCOMPILE) sample-vmf-dp-mixture.cc strutil.o dSFMT.o gibbs-base.o -o samplevMFDPMixture

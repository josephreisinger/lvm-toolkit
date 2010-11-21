// This implements the multinomial sampler for FixedDepthNCRP, which constrains the
// topics to be a tree with fixed depth.


#ifndef SAMPLE_MULT_NCRP_H_
#define SAMPLE_MULT_NCRP_H_

class FixedDepthNCRP : public NCRPBase {
    public:
        FixedDepthNCRP() { }
        ~FixedDepthNCRP() { /* TODO: free memory! */ }

        string current_state();
    private:
        void resample_posterior();
        void resample_posterior_z_for(unsigned d, bool remove);

        double compute_log_likelihood();
};

#endif  // SAMPLE_MULT_NCRP_H_

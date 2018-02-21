#ifndef NET_H
#define NET_H

#include <string>
#include <vector>
#include <map>


#include "hdf5.h"
#include "H5Cpp.h"
#include "hdf5_hl.h"


#include "layer.h"
#include "common.h"

using namespace std;
using namespace H5;

namespace danknet {



template <typename Dtype>
class Net {
private:
    Phase phase_;
    string name_;
    Dtype                lr_rate_ = 0;
    Dtype                weight_decay_ = 0;
    Dtype                momentum_ = 0;
    Dtype                gamma_ = 0;

    int train_iters = 0;
    int test_iters = 0;
    int step_size_ = 0;

    vector<Layer<Dtype>*> layers_;
    map<string, Blob<Dtype>*> blobs_;

    // vector<Data3d<Dtype>*> bottom_data_;
    // vector<Data3d<Dtype>*> top_data_;

//    if(blobs_.find(l0) != blobs_.end()) {

//    }
public:
    explicit Net() {}
    virtual ~Net() {}

    void AddLayer(Layer<Dtype>* layer);
//    void AddBlob(Blob<Dtype>& blob);

    inline string name() { return name_; }

    inline Phase phase() { return phase_; }
    inline void phase(Phase phase) {
        phase_ = phase;
        for(int i = 0; i < layers_.size(); i++) {
            layers_[i]->phase(phase_);
        }
    }

    inline Dtype lr_rate() { return lr_rate_; }
    inline void lr_rate(Dtype lr_rate) {
        lr_rate_ = lr_rate;
        for(int i = 0; i < layers_.size(); i++) {
            layers_[i]->lr_rate(lr_rate_);
        }
    }

    inline Dtype weight_decay() {return weight_decay_;}
    inline void weight_decay(Dtype weight_decay) {
        weight_decay_ = weight_decay;
        for(int i = 0; i < layers_.size(); i++) {
            layers_[i]->weight_decay(weight_decay_);
        }
    }

    inline Dtype momentum() {return momentum_;}
    inline void momentum(Dtype momentum) {
        momentum_ = momentum;
        for(int i = 0; i < layers_.size(); i++) {
            layers_[i]->momentum(momentum_);
        }
    }

    inline Dtype gamma() {return gamma_;}
    inline void gamma(Dtype gamma) { gamma_ = gamma;}

    inline Dtype step_size() {return step_size_;}
    inline void step_size(Dtype step_size) { step_size_ = step_size; }

    void Forward();
    void Backward();

    void WeightsFromHDF5(string filename);
    void WeightsToHDF5(string filename);
};

} // namespace danknet

#endif // NET_H

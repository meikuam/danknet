#ifndef NET_H
#define NET_H

#include <vector>
#include <map>

#include <string>

#include "layer.h"
#include "common.h"

using namespace std;

namespace danknet {


enum Phase {
    TRAIN,
    TEST
};

/*
 * Root layer has no bottom data. That data feeds to net via "load_batch()" method.
 */

template <typename Dtype>
class Net {
private:
    Phase           phase_;
    string          name_;
    vector<Layer<Dtype>*> layers_;
    map<string, Blob<Dtype>*> blobs_;

    // vector<Data3d<Dtype>*> bottom_data_;
    // vector<Data3d<Dtype>*> top_data_;

//    if(blobs_.find(l0) != blobs_.end()) {

//    }
public:
    explicit Net();
    virtual ~Net() {}

    void AddLayer(Layer<Dtype>* layer);
    void AddBlob(Blob<Dtype>& blob);

    void Compile();

    inline Phase phase() { return phase_; }
    inline string name() { return name_; }

    void Forward();
    void Backward();

    void WeightsFromHDF5(string filename);
    void WeightsToHDF5(string filename);
};

} // namespace danknet

#endif // NET_H

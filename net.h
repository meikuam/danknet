#ifndef NET_H
#define NET_H

#include <vector>
#include <string>

#include "layer.h"

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

    vector<Data2d<Dtype>*> bottom_data_;
    vector<Data2d<Dtype>*> top_data_;
public:
    explicit Net();
    virtual ~Net() {}

    void AddLayer(Layer<Dtype>* layer);
    void Compile();

    inline Phase phase() { return phase_; }
    inline string name() { return name_; }

    const vector<Data2d<Dtype>*>& Forward();
    void Backward();

    void WeightsFromHDF5(string filename);
    void WeightsToHDF5(string filename);
};

} // namespace danknet

#endif // NET_H

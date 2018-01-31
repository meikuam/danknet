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


template <typename Dtype>
class Net {
private:
//    vector<Layer*>    layers_;
    Phase           phase_;
    string          name_;
    vector<Layer<Dtype>*> layers_;
public:
    explicit Net();
    ~Net();

    void Init();
    inline Phase phase() { return phase_; }
    inline string name() { return name_; }

    const vector<Data2d<Dtype>*>& Forward();
    void Backward();

    void WeightsFromHDF5(string filename);
    void WeightsToHDF5(string filename);
};

} // namespace danknet

#endif // NET_H

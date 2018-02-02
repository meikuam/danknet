#include "layer.h"

namespace danknet {

template<class Dtype>
Layer<Dtype>::Layer(string name, vector<string> bottom, vector<string> top) {
    name_ = name;
    for(vector<string>::iterator it = bottom.begin(); it != bottom.end(); it++) {
        bottom_.push_back(*it);
    }
    for(vector<string>::iterator it = top.begin(); it != top.end(); it++) {
        top_.push_back(*it);
    }
}

} // namespace danknet

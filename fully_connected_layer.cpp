#include "fully_connected_layer.h"

namespace danknet {

template<typename Dtype>
vector<Blob<Dtype>*>*
FullyConectedLayer<Dtype>::Forward() {
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
FullyConectedLayer<Dtype>::Backward() {
    return &this->bottom_;
}
} // namespace danknet

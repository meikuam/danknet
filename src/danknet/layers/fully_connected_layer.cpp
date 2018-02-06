#include "fully_connected_layer.h"

namespace danknet {

template<typename Dtype>
FullyConectedLayer<Dtype>::FullyConectedLayer(int units,
                                              string name,
                                              vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {
      units_ = units;

}


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
INSTANTIATE_CLASS(FullyConectedLayer);
} // namespace danknet

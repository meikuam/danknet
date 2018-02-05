#include "convolutional_layer.h"

namespace danknet {

template<typename Dtype>
vector<Blob<Dtype>*>*
ConvolutionalLayer<Dtype>::Forward() {
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
ConvolutionalLayer<Dtype>::Backward() {
    return &this->bottom_;
}

INSTANTIATE_CLASS(ConvolutionalLayer);
} // namespace danknet

#include "pooling_layer.h"


namespace danknet {

template<typename Dtype>
vector<Blob<Dtype>*>*
PoolingLayer<Dtype>::Forward() {
        return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
PoolingLayer<Dtype>::Backward() {
    return &this->bottom_;
}
} // namespace danknet

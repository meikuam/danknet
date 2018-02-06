#include "loss_layer.h"

namespace danknet {


template<typename Dtype>
LossLayer<Dtype>::LossLayer(string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {

}


template<typename Dtype>
vector<Blob<Dtype>*>*
LossLayer<Dtype>::Forward() {
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
LossLayer<Dtype>::Backward() {
    return &this->bottom_;
}
INSTANTIATE_CLASS(LossLayer);
} // namespace danknet

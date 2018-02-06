#include "pooling_layer.h"


namespace danknet {

template<typename Dtype>
PoolingLayer<Dtype>::PoolingLayer(int kernel_w, int kernel_h,
                                  int stride_w, int stride_h,
                                  int pad_w, int pad_h,
                                  string name,
                                  vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {
      kernel_w_ = kernel_w;
      kernel_h_ = kernel_h;
      stride_w_ = stride_w;
      stride_h_ = stride_h;
      pad_w_ = pad_w;
      pad_h_ = pad_h;
  }

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
INSTANTIATE_CLASS(PoolingLayer);
} // namespace danknet

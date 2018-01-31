#include "pooling_layer.h"


namespace danknet {

template<typename Dtype>
void
PoolingLayer<Dtype>::LayerSetUp(const vector<Data2d<Dtype>*>& bottom,
                                const vector<Data2d<Dtype>*>& top) {

}


template<typename Dtype>
void
PoolingLayer<Dtype>::Forward(const vector<Data2d<Dtype>*>& bottom,
                             const vector<Data2d<Dtype>*>& top) {

}

template<typename Dtype>
void
PoolingLayer<Dtype>::Backward(const vector<Data2d<Dtype>*>& bottom,
                              const vector<Data2d<Dtype>*>& top) {

}
} // namespace danknet

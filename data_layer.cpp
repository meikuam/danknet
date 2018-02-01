#include "data_layer.h"


namespace danknet {

template<typename Dtype>
void
DataLayer<Dtype>::LayerSetUp(const vector<Data2d<Dtype>*>& bottom,
                             const vector<Data2d<Dtype>*>& top) {

}


template<typename Dtype>
void
DataLayer<Dtype>::Forward(const vector<Data2d<Dtype>*>& bottom,
                          const vector<Data2d<Dtype>*>& top) {

}

template<typename Dtype>
void
DataLayer<Dtype>::Backward(const vector<Data2d<Dtype>*>& bottom,
                           const vector<Data2d<Dtype>*>& top) {

}
} // namespace danknet

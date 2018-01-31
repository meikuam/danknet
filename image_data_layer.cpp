#include "image_data_layer.h"


namespace danknet {

template<typename Dtype>
void
ImageDataLayer<Dtype>::LayerSetUp(const vector<Data2d<Dtype>*>& bottom,
                                  const vector<Data2d<Dtype>*>& top) {

}


template<typename Dtype>
void
ImageDataLayer<Dtype>::Forward(const vector<Data2d<Dtype>*>& bottom,
                               const vector<Data2d<Dtype>*>& top) {

}

template<typename Dtype>
void
ImageDataLayer<Dtype>::Backward(const vector<Data2d<Dtype>*>& bottom,
                                const vector<Data2d<Dtype>*>& top) {

}
} // namespace danknet

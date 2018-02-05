#include "layer.h"
#include <iostream>


namespace danknet {
template<class Dtype>
Layer<Dtype>::Layer(string name, vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top) {
    name_ = name;
    bottom_ = bottom;
    for(int i = 0; i < bottom.size(); i++) {
        top_.push_back(new Blob<Dtype>(name_ + "_top", bottom_[i]->shape()));
    }
    top = top_;


}

template<class Dtype>
vector<Blob<Dtype>*>* Layer<Dtype>::Forward() {
    Data3d<Dtype>* bottom = bottom_[0]->data(0);
    Data3d<Dtype>* top = top_[0]->data(0);
//    *(top->data(0,0,0)) = *(bottom->data(0,0,0)) + 10;
    return &top_;
}

template<class Dtype>
vector<Blob<Dtype>*>* Layer<Dtype>::Backward() {
    return &bottom_;
}

INSTANTIATE_CLASS(Layer);

} // namespace danknet

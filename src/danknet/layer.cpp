#include "layer.h"

namespace danknet {


template<class Dtype>
Layer<Dtype>::Layer(string name,
                    vector<Blob<Dtype>*>& bottom,
                    vector<Blob<Dtype>*>& top) {
    name_ = name;
//    bottom_ = bottom;
//    for(int i = 0; i < bottom.size(); i++) {
//        top_.push_back(new Blob<Dtype>(name_ + "_top", bottom_[i]->shape()));
//    }
//    top = top_;
}


template<class Dtype>
vector<Blob<Dtype>*>*
Layer<Dtype>::Forward() {
    Data3d<Dtype>* bottom = bottom_[0]->Data(0);
    Data3d<Dtype>* top = top_[0]->Data(0);
    Shape top_shape = top->shape();
    for(int w = 0; w < top_shape.width(); w++)
        for(int h = 0; h < top_shape.height(); h++)
            for(int c = 0; c < top_shape.depth(); c++) {
                *top->data(w, h, c) = *bottom->data(w, h, c)*1.1;
            }
    return &top_;
}


template<class Dtype>
vector<Blob<Dtype>*>*
Layer<Dtype>::Backward() {
    return &bottom_;
}

INSTANTIATE_CLASS(Layer);

} // namespace danknet

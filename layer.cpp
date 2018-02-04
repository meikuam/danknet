#include "layer.h"
namespace danknet {

template<class Dtype>
Layer<Dtype>::Layer(string name, vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top) {
    name_ = name;
    for(int i = 0; i < bottom.size(); i++) {
        bottom_.push_back(bottom[i]);
        top.push_back(new Blob<Dtype>(name_ + bottom[i]->name(), bottom[i]->shape()));
    }
}

//template<class Dtype>
//vector<Blob<Dtype>*>* Layer<Dtype>::Forward() {
//    std::cout<<"forward"<<std::endl;
//}

//template<class Dtype>
//vector<Blob<Dtype>*>* Layer<Dtype>::Backward() {

//}

INSTANTIATE_CLASS(Layer);

} // namespace danknet

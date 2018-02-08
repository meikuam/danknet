#include "softmax_layer.h"

namespace danknet {


template<typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer(string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {

    //-------------copy bottom vector-------------
    this->bottom_ = bottom;

    //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(1, 1, 1/*this->bottom_[0]->shape().depth()*/, this->bottom_[0]->shape().batch())));
    top = this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
SoftmaxLayer<Dtype>::Forward() {
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    //---------------clear batches----------------
    top->setToZero();
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape bottom_shape = bottom_data->shape();
        int K = bottom_shape.depth();
        Dtype* exps = new Dtype[K];
        Dtype sum =  (Dtype)0;
//        for(int k = 0; k < K; k++) {
//            *top_data->data(0, 0, k) = exp(*bottom_data->data(0, 0, k));
//            sum += *top_data->data(0, 0, k);
//        }
//        for(int k = 0; k < K; k++) {
//            *top_data->data(0, 0, k) = *top_data->data(0, 0, k) / sum;
//        }
        for(int k = 0; k < K; k++) {
            exps[k] = exp(*bottom_data->data(0, 0, k));
            sum += exps[k];
        }
        for(int k = 0; k < K; k++) {
            exps[k] = exps[k] / sum;
        }
        int max_num = 0;
        for(int k = 1; k < K; k++) {
            if(exps[k] > exps[max_num]) {
                max_num = k;
            }
        }
        *top_data->data(0, 0, 0) = max_num;
        delete exps;
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
SoftmaxLayer<Dtype>::Backward() {
    return &this->bottom_;
}
INSTANTIATE_CLASS(SoftmaxLayer);
} // namespace danknet

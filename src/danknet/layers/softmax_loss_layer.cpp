#include "softmax_loss_layer.h"
namespace danknet {


template<typename Dtype>
SoftmaxLossLayer<Dtype>::SoftmaxLossLayer(Activation activation, string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {
    //-------------copy bottom vector-------------
    this->bottom_ = bottom;
    //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(1, 1, 1, this->bottom_[0]->shape().batch())));
    top = this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
SoftmaxLossLayer<Dtype>::Forward() {

    Blob<Dtype>* pred = this->bottom_[0];
    Blob<Dtype>* labels = this->bottom_[1];
    Blob<Dtype>* top = this->top_[0];
    //-------------clear top batches--------------
    top->setToZero();
    //-------------------batch--------------------
    for(int batch = 0; batch < pred->batch_size(); batch++) {
        Data3d<Dtype>* pred_data = pred->Data(batch);
        Data3d<Dtype>* labels_data = labels->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);


        Shape pred_shape = pred_data->shape();
        int K = pred_shape.depth();
//        int w = pred_shape.width();
//        int h = pred_shape.height();

        //------------------softmax-------------------
        Dtype* exps = new Dtype[K];
        Dtype sum =  (Dtype)0;
        for(int k = 0; k < K; k++) {
            exps[k] = exp(*pred_data->data(0, 0, k));
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
        int label = 0;
        for(int k = 0; k < K; k++) {
            if(*labels_data->data(0, 0, k) == 1) {
                label = k;
            }
        }
        //loss  = -log(Py)
        *top_data->data(0, 0, 0) = -log(exps[label]);

        delete exps;
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
SoftmaxLossLayer<Dtype>::Backward() {
    Blob<Dtype>* pred = this->bottom_[0];
    Blob<Dtype>* labels = this->bottom_[1];
    for(int batch = 0; batch < pred->batch_size(); batch++) {
        Data3d<Dtype>* pred_data = pred->Data(batch);
        Data3d<Dtype>* labels_data = labels->Data(batch);
        Shape pred_shape = pred_data->shape();
        int K = pred_shape.depth();
        int w = pred_shape.width();
        int h = pred_shape.height();

        //------------------softmax-------------------
        Dtype* exps = new Dtype[K];
        Dtype sum =  (Dtype)0;
        for(int k = 0; k < K; k++) {
            exps[k] = exp(*pred_data->data(0, 0, k));
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
        int label = 0;
        for(int k = 0; k < K; k++) {
            if(*labels_data->data(0, 0, k) == 1) {
                label = k;
            }
        }
        //
        for(int k = 0; k < K; k++) {
            if(k == label) {
                *pred_data->data(0, 0, k) = -exps[k];
            } else {
                *pred_data->data(0, 0, k) = exps[k];
            }
        }
        delete exps;
    }

    return &this->bottom_;
}
INSTANTIATE_CLASS(SoftmaxLossLayer);
} // namespace danknet

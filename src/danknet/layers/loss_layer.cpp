#include "loss_layer.h"
namespace danknet {


template<typename Dtype>
LossLayer<Dtype>::LossLayer(Dtype lr_rate,
                            string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {
    this->lr_rate_ = lr_rate;
    //-------------copy bottom vector-------------
    this->bottom_ = bottom;
    //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(1, 1, 1, this->bottom_[0]->shape().batch())));
    top = this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
LossLayer<Dtype>::Forward() {

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

        //loss  = 0.5 * sum((pred - labels)^2)
        Dtype loss = (Dtype)0;
        for(int k = 0; k < pred_data->shape().depth(); k++) {
            loss +=  (*labels_data->data(0,0,k) - *pred_data->data(0, 0, k)) * (*labels_data->data(0,0,k) - *pred_data->data(0, 0, k));
        }
        loss /= (Dtype)2;

        *top_data->data(0, 0, 0) = loss;
    }

    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
LossLayer<Dtype>::Backward() {
    Blob<Dtype>* pred = this->bottom_[0];
    Blob<Dtype>* labels = this->bottom_[1];
    for(int batch = 0; batch < pred->batch_size(); batch++) {
        Data3d<Dtype>* pred_data = pred->Data(batch);
        Data3d<Dtype>* labels_data = labels->Data(batch);
        Shape pred_shape = pred_data->shape();
        int K = pred_shape.depth();
        //dE/dyk = - (pred_data(k) - labels(k))
        for(int k = 0; k < K; k++) {
            *pred_data->data(0, 0, k) =  - (*labels_data->data(0,0,k) - *pred_data->data(0, 0, k)) * this->lr_rate_;
        }
    }

    return &this->bottom_;
}
INSTANTIATE_CLASS(LossLayer);
} // namespace danknet

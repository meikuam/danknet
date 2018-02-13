#include "loss_layer.h"
#include <iostream>
using namespace std;
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
    //---------------clear batches----------------
    top->setToZero();
    //-------------------batch--------------------
    for(int batch = 0; batch < pred->batch_size(); batch++) {
        Data3d<Dtype>* pred_data = pred->Data(batch);
        Data3d<Dtype>* labels_data = labels->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);

        //softmax
        Shape pred_shape = pred_data->shape();
        int K = pred_shape.depth();
        Dtype* exps = new Dtype[K];
        Dtype sum =  (Dtype)0;
        for(int k = 0; k < K; k++) {
            exps[k] = exp(*pred_data->data(0, 0, k));
            sum += exps[k];
        }
        for(int k = 0; k < K; k++) {
            exps[k] = exps[k] / sum;
        }
//        int max_num = 0;
//        for(int k = 1; k < K; k++) {
//            if(exps[k] > exps[max_num]) {
//                max_num = k;
//            }
//        }
//        for(int k = 0; k < K; k++) {
//            if(k == max_num)
//                exps[k] = 1;
//            else
//                exps[k] = 0;
//        }
        //loss  = 0.5 * sum((pred - labels)^2)
        Dtype loss = (Dtype)0;
        for(int k = 0; k < K; k++) {
            loss +=  (*labels_data->data(0,0,k) - exps[k]) * (*labels_data->data(0,0,k) - exps[k]);
        }
        loss /= (Dtype)2;

        *top_data->data(0, 0, 0) = loss;
        delete exps;
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

        //softmax
        Shape pred_shape = pred_data->shape();
        int K = pred_shape.depth();
        Dtype* exps = new Dtype[K];
        Dtype sum =  (Dtype)0;
        cout<<"pred_data: ";
        for(int k = 0; k < K; k++) {
            cout<<*pred_data->data(0, 0, k)<<" ";
            exps[k] = exp(*pred_data->data(0, 0, k));
            sum += exps[k];
        }
        cout<<endl<<"exps: ";
        for(int k = 0; k < K; k++) {
            exps[k] = exps[k] / sum;
            cout<<exps[k]<<" ";
        }

        cout<<endl<<"labels: ";
        for(int k = 0; k < K; k++) {
            cout<<*labels_data->data(0,0,k)<<" ";
        }

        cout<<endl<<"errors: ";
        //dE/dyk = - (pred_data(k) - labels(k))
        for(int k = 0; k < K; k++) {
            *pred_data->data(0, 0, k) =  - (exps[k] - *labels_data->data(0,0,k)) * this->lr_rate_;
            cout<<*pred_data->data(0, 0, k)<<" ";
        }
        cout<<endl;
//        cout<<pred->name();
        delete exps;
    }

    return &this->bottom_;
}
INSTANTIATE_CLASS(LossLayer);
} // namespace danknet

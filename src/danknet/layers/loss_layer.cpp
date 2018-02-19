#include "loss_layer.h"
namespace danknet {


template<typename Dtype>
LossLayer<Dtype>::LossLayer(string name,
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

        Shape pred_shape = pred_data->shape();
        int K = pred_shape.depth();
        int w = pred_shape.width();
        int h = pred_shape.height();

        for(int x = 0; x < w; x++) {
            for(int y = 0; y < h; y++) {
                for(int k = 0; k < K; k++) {
                    loss +=  (*labels_data->data(x, y, k) - *pred_data->data(x, y, k)) * (*labels_data->data(x, y, k) - *pred_data->data(x, y, k));
                }
            }
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
        int w = pred_shape.width();
        int h = pred_shape.height();

        //dE/dyk = - (labels(k) - pred_data(k))
        for(int x = 0; x < w; x++) {
            for(int y = 0; y < h; y++) {
                for(int k = 0; k < K; k++) {
                    *pred_data->data(x, y, k) = - (*labels_data->data(x, y, k) - *pred_data->data(x, y, k));// * this->lr_rate_;
                }
            }
        }
    }

    for(int batch = 0; batch < pred->batch_size(); batch++) {
        Data3d<Dtype>* pred_data = pred->Data(batch);
        for(int x = 0; x < pred->width(); x++) {
            for(int y = 0; y < pred->height(); y++) {
                for(int k = 0; k < pred->depth(); k++) {
                    *pred_data->data(x, y, k) /= pred->batch_size();
                }
            }
        }
    }

    return &this->bottom_;
}
INSTANTIATE_CLASS(LossLayer);
} // namespace danknet

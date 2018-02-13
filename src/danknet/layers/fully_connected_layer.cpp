#include "fully_connected_layer.h"

namespace danknet {

template<typename Dtype>
FullyConectedLayer<Dtype>::FullyConectedLayer(int units,
                                              Dtype lr_rate,
                                              string name,
                                              vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top),
      distribution(-0.1,0.1),
      generator(std::chrono::system_clock::now().time_since_epoch().count())
{
      units_ = units;
      this->lr_rate_= lr_rate;
      //-----------------Blob<Dtype>*---------------

      //-------------copy bottom vector-------------
      this->bottom_ = bottom;

      //---------------create weights_--------------
      Shape bottom_shape = bottom[0]->shape();
      this->weights_ = new Blob<Dtype>(this->name_ + "_weights", Shape(bottom_shape.width(), bottom_shape.height(), bottom_shape.depth(), units_));
      this->weights_diff_ = new Blob<Dtype>(this->name_ + "_weights_diff", Shape(bottom_shape.width(), bottom_shape.height(), bottom_shape.depth(), units_));
      initWeights();

      //-------------create top vector--------------
      this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(1, 1, units_, this->bottom_[0]->shape().batch())));
      top = this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
FullyConectedLayer<Dtype>::Forward() {
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    Blob<Dtype>* weights = this->weights_;
    //---------------clear batches----------------
    top->setToZero();
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape bottom_shape = bottom_data->shape();
        Shape top_shape = top_data->shape();
        Shape weights_shape = this->weights_->shape();

    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
FullyConectedLayer<Dtype>::Backward() {
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    Blob<Dtype>* weights = this->weights_;
    Blob<Dtype>* weights_diff = this->weights_diff_;

    weights_diff->setToZero();
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape bottom_shape = bottom_data->shape();
        Shape top_shape = top_data->shape();
        Shape weights_shape = weights_diff->shape();

    }
    return &this->bottom_;
}


template<typename Dtype>
void
FullyConectedLayer<Dtype>::initWeights() {
    Blob<Dtype>* weights = this->weights_;
    Shape weights_shape = this->weights_->shape();

    for(int k = 0; k < weights_shape.batch(); k++) {
       for(int c = 0; c < weights_shape.depth(); c++) {
           for(int x = 0; x < weights_shape.width(); x++) {
               for(int y = 0; y < weights_shape.height(); y++) {
                   *weights->data(k, x, y, c) = (Dtype)(distribution(generator));
               }
           }
       }
    }
}

INSTANTIATE_CLASS(FullyConectedLayer);
} // namespace danknet

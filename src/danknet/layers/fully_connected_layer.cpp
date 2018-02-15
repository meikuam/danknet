#include "fully_connected_layer.h"

namespace danknet {

template<typename Dtype>
FullyConnectedLayer<Dtype>::FullyConnectedLayer(int units,
                                                string name,
                                                vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top),
      distribution(-1, 1),
      generator(std::chrono::system_clock::now().time_since_epoch().count())
{
      units_ = units;
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
FullyConnectedLayer<Dtype>::Forward() {
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
        int bottom_units = bottom_shape.width() * bottom_shape.height() * bottom_shape.depth();
        for(int top_unit = 0; top_unit < top_shape.depth(); top_unit++) {
            Data3d<Dtype>* weights_data = weights->Data(top_unit);
            Dtype* bottom_ptr = bottom_data->data();
            Dtype* weights_ptr = weights_data->data();
            Dtype* top_ptr = top_data->data();

            for(int bottom_unit = 0; bottom_unit < bottom_units; bottom_unit++) {
                top_ptr[top_unit] += bottom_ptr[bottom_unit] * weights_ptr[bottom_unit];
            }
            //-------------ReLU activation----------------
            if(top_ptr[top_unit] <= 0) {
                top_ptr[top_unit] = 0;
            }
        }

    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
FullyConnectedLayer<Dtype>::Backward() {
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
        int bottom_units = bottom_shape.width() * bottom_shape.height() * bottom_shape.depth();


        // calc weights diffs

        for(int top_unit = 0; top_unit < top_shape.depth(); top_unit++) {
            Dtype* bottom_ptr = bottom_data->data();
            Dtype* weights_diff_ptr = weights_diff->data(top_unit);
            Dtype* top_ptr = top_data->data();

            for(int bottom_unit = 0; bottom_unit < bottom_units; bottom_unit++) {
                weights_diff_ptr[bottom_unit] += top_ptr[top_unit] * bottom_ptr[bottom_unit];
            }
        }
        // calc error
        bottom_data->setToZero();
        for(int top_unit = 0; top_unit < top_shape.depth(); top_unit++) {
            Dtype* bottom_ptr = bottom_data->data();
            Dtype* weights_ptr = weights->data(top_unit);
            Dtype* top_ptr = top_data->data();

            for(int bottom_unit = 0; bottom_unit < bottom_units; bottom_unit++) {
                bottom_ptr[bottom_unit] += weights_ptr[bottom_unit] * top_ptr[top_unit];
            }
        }
        //-------------ReLU derivation----------------
            Dtype* bottom_ptr = bottom_data->data();
            for(int bottom_unit = 0; bottom_unit < bottom_units; bottom_unit++) {
                if(bottom_ptr[bottom_unit] <= 0 || isnan(bottom_ptr[bottom_unit])) {
                    bottom_ptr[bottom_unit] = 0;
                }
            }
        // update weights
        for(int top_unit = 0; top_unit < top_shape.depth(); top_unit++) {
            Dtype* weights_diff_ptr = weights_diff->data(top_unit);
            Dtype* weights_ptr = weights->data(top_unit);

            for(int bottom_unit = 0; bottom_unit < bottom_units; bottom_unit++) {
                if(!isnan(weights_diff_ptr[bottom_unit])) {
                    weights_ptr[bottom_unit] -= (weights_diff_ptr[bottom_unit] + weights_ptr[bottom_unit] * this->weight_decay_) *this->lr_rate_;
                }
            }
        }

    }
    return &this->bottom_;
}


template<typename Dtype>
void
FullyConnectedLayer<Dtype>::initWeights() {
    Blob<Dtype>* weights = this->weights_;
    Shape weights_shape = this->weights_->shape();
    int max_num =  weights_shape.width() * weights_shape.height() * weights_shape.depth() * weights_shape.batch();
    for(int k = 0; k < weights_shape.batch(); k++) {
       for(int c = 0; c < weights_shape.depth(); c++) {
           for(int x = 0; x < weights_shape.width(); x++) {
               for(int y = 0; y < weights_shape.height(); y++) {
                   *weights->data(k, x, y, c) = (Dtype)(distribution(generator)) * sqrt(4.0 / max_num);
               }
           }
       }
    }
}

INSTANTIATE_CLASS(FullyConnectedLayer);
} // namespace danknet

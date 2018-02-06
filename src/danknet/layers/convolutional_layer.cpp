#include "convolutional_layer.h"

namespace danknet {

template<typename Dtype>
ConvolutionalLayer<Dtype>::ConvolutionalLayer(int kernel_w, int kernel_h,
                            int depth, int kernels,
                            int stride_w, int stride_h,
                            int pad_w, int pad_h,
                            string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
      : Layer<Dtype>(name, bottom, top) {
      kernel_w_     = kernel_w;
      kernel_h_     = kernel_h;
      kernels_      = kernels;
      depth_        = depth;
      stride_w_     = stride_w;
      stride_h_     = stride_h;
      pad_w_        = pad_w;
      pad_h_        = pad_h;

      //-----------------Blob<Dtype>*---------------
      //---------------create weights_--------------
      this->weights_ = new Blob<Dtype>(this->name_ + "_weights", Shape(kernel_w_, kernel_h_, depth_, kernels_));
      //-------------copy bottom vector-------------
      this->bottom_ = bottom;

      //-------------create top vector--------------
      //(input_dim + 2 * pad - kernel_size) / stride + 1;
      int out_w = (this->bottom_[0]->shape().width() + 2 * pad_w_ - kernel_w_) / stride_w_;
      int out_h = (this->bottom_[0]->shape().height() + 2 * pad_h_ - kernel_h_) / stride_h_;
      this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(out_w, out_h, kernels_, this->bottom_[0]->shape().batch())));
      top = this->top_;
}

template<typename Dtype>
vector<Blob<Dtype>*>*
ConvolutionalLayer<Dtype>::Forward() {
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    Blob<Dtype>* weights = this->weights_;
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape bottom_shape = bottom_data->shape();
        Shape top_shape = top_data->shape();
        Shape weights_shape = this->weights_->shape();
        //-------------------kernel-------------------
        for(int kernel = 0; kernel < weights_shape.batch(); kernel++) {


            for(int depth = 0; depth < weights_shape.depth(); depth++) {
                for(int out_w = 0; out_w < top_shape.width(); out_w++) {
                    for(int out_h = 0; out_h < top_shape.height(); out_h++) {

                    }
                }
            }
        }
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
ConvolutionalLayer<Dtype>::Backward() {
    return &this->bottom_;
}

INSTANTIATE_CLASS(ConvolutionalLayer);
} // namespace danknet

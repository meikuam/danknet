#include "convolutional_layer.h"

namespace danknet {

template<typename Dtype>
ConvolutionalLayer<Dtype>::ConvolutionalLayer(int kernel_w, int kernel_h,
                            int depth, int kernels,
                            int stride_w, int stride_h,
                            int pad_w, int pad_h,
                            string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
      : Layer<Dtype>(name, bottom, top),
        distribution(-0.05,0.05),
        generator(std::chrono::system_clock::now().time_since_epoch().count())
{
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
      initWeights();

      //-------------copy bottom vector-------------
      this->bottom_ = bottom;

      //-------------create top vector--------------
      //(input_dim + 2 * pad - kernel_size) / stride + 1;
      int out_w = (this->bottom_[0]->shape().width() + /*2 **/ pad_w_ - kernel_w_) / stride_w_ + 1;
      int out_h = (this->bottom_[0]->shape().height() + /*2 **/ pad_h_ - kernel_h_) / stride_h_ + 1;
      this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(out_w, out_h, kernels_, this->bottom_[0]->shape().batch())));
      top = this->top_;
}

template<typename Dtype>
vector<Blob<Dtype>*>*
ConvolutionalLayer<Dtype>::Forward() {
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
        //-------------------kernel-------------------
        for(int kernel = 0; kernel < top_shape.depth(); kernel++) {
            Data3d<Dtype>* weights_data = weights->Data(kernel);
            for(int depth = 0; depth < bottom_shape.depth(); depth++) {
                for(int out_x = 0, inp_x = 0/*- pad_w_*/; out_x < top_shape.width(); out_x++,  inp_x += stride_w_ ) {
                    for(int out_y = 0, inp_y = 0/*- pad_h_*/; out_y < top_shape.height(); out_y++, inp_y += stride_h_) {
                        //--------convolution (correlation)-----------
                        for(int x = 0; x < weights_shape.width(); x++) {
                            for(int y = 0; y < weights_shape.height(); y++) {
                                *top_data->data(out_x, out_y, kernel) += *bottom_data->data(inp_x + x, inp_y + y, depth) * *weights_data->data(x, y, depth) ;
                            }
                        }
                    }
                }
            }
            for(int out_x = 0; out_x < top_shape.width(); out_x++) {
                for(int out_y = 0; out_y < top_shape.height(); out_y++) {
                    //-------------ReLU activation----------------
                    if(*top_data->data(out_x, out_y, kernel) < 0) {
                        *top_data->data(out_x, out_y, kernel) = 0;
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
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    Blob<Dtype>* weights = this->weights_;

    // 1)
    // dh^-1 * X + weights
    // weights = top^-1 * bottom + weights
    // 2)
    // X = weights^-1 * dh
    // bottom = weights^-1 * top


    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape bottom_shape = bottom_data->shape();
        Shape top_shape = top_data->shape();
        Shape weights_shape = this->weights_->shape();
//        //-------------------kernel-------------------
        for(int kernel = 0; kernel < top_shape.depth(); kernel++) {
            Data3d<Dtype>* weights_data = weights->Data(kernel);
            for(int depth = 0; depth < bottom_shape.depth(); depth++) {
                for(int out_x = 0, inp_x = 0/*- pad_w_*/; out_x < weights_shape.width(); out_x++,  inp_x += stride_w_ ) {
                    for(int out_y = 0, inp_y = 0/*- pad_h_*/; out_y < weights_shape.height(); out_y++, inp_y += stride_h_) {
                        for(int x = 0; x < top_shape.width(); x++) {
                            for(int y = 0; y < top_shape.height(); y++) {
                                *weights_data->data(out_x, out_y, depth) += *top_data->data(x, y, kernel) * *bottom_data->data(inp_x + x, inp_y + y, depth)*0.1;
                            }
                        }
                    }
                }
            }
        }
        bottom_data->setToZero();
        for(int kernel = 0; kernel < top_shape.depth(); kernel++) {
            Data3d<Dtype>* weights_data = weights->Data(kernel);
            for(int depth = 0; depth < bottom_shape.depth(); depth++) {
                for(int out_x = 0, inp_x = 0/*- pad_w_*/; out_x < top_shape.width(); out_x++,  inp_x += stride_w_ ) {
                    for(int out_y = 0, inp_y = 0/*- pad_h_*/; out_y < top_shape.height(); out_y++, inp_y += stride_h_) {

                        for(int x = 0; x < weights_shape.width(); x++) {
                            for(int y = 0; y < weights_shape.height(); y++) {
                                *bottom_data->data(inp_x + x, inp_y + y, depth) += *top_data->data(out_x, out_y, kernel) * *weights_data->data(x, y, depth);
                            }
                        }
                    }
                }
            }
        }
    }
    return &this->bottom_;
}

template<typename Dtype>
void
ConvolutionalLayer<Dtype>::initWeights() {
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

INSTANTIATE_CLASS(ConvolutionalLayer);
} // namespace danknet

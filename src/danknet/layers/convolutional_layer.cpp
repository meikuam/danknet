#include "convolutional_layer.h"

namespace danknet {

template<typename Dtype>
ConvolutionalLayer<Dtype>::ConvolutionalLayer(int kernel_w, int kernel_h,
                            int depth, int kernels,
                            int stride_w, int stride_h,
                            int pad_w, int pad_h,
                            Activation activation,
                            string name,
                            vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
      : Layer<Dtype>(name, bottom, top),
        distribution(0, 1),
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

      activation_   = activation;

      //-----------------Blob<Dtype>*---------------
      //---------------create weights_--------------
      this->weights_ = new Blob<Dtype>(this->name_ + "_weights", Shape(kernel_w_, kernel_h_, depth_, kernels_));
      this->weights_diff_ = new Blob<Dtype>(this->name_ + "_weights_diff", Shape(kernel_w_, kernel_h_, depth_, kernels_));

      //-------------copy bottom vector-------------
      this->bottom_ = bottom;

      //-------------create top vector--------------
      //(input_dim + 2 * pad - kernel_size) / stride + 1;
      int top_w = (this->bottom_[0]->shape().width() + /*2 **/ pad_w_ - kernel_w_) / stride_w_ + 1;
      int top_h = (this->bottom_[0]->shape().height() + /*2 **/ pad_h_ - kernel_h_) / stride_h_ + 1;
      this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(top_w, top_h, kernels_, this->bottom_[0]->shape().batch())));
      top = this->top_;

      initWeights();
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
                for(int top_x = 0, bottom_x = 0/*- pad_w_*/; top_x < top_shape.width(); top_x++,  bottom_x += stride_w_ ) {
                    for(int top_y = 0, bottom_y = 0/*- pad_h_*/; top_y < top_shape.height(); top_y++, bottom_y += stride_h_) {
                        //--------convolution (correlation)-----------
                        for(int x = 0; x < weights_shape.width(); x++) {
                            for(int y = 0; y < weights_shape.height(); y++) {
                                *top_data->data(top_x, top_y, kernel) += *bottom_data->data(bottom_x + x, bottom_y + y, depth) * *weights_data->data(x, y, depth) ;
                            }
                        }
                    }
                }
            }
            for(int top_x = 0; top_x < top_shape.width(); top_x++) {
                for(int top_y = 0; top_y < top_shape.height(); top_y++) {
                    //----------------activation------------------
                    *top_data->data(top_x, top_y, kernel) = act_func(*top_data->data(top_x, top_y, kernel), activation_);
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
    Blob<Dtype>* weights_diff = this->weights_diff_;
    Blob<Dtype>* weights_diff_new = new Blob<Dtype>("weights_diff_new", weights_diff->shape());
    weights_diff_new ->setToZero();

    // 1)
    // dh^-1 * X
    // weights_diff = top^-1 * bottom
    // 2)
    // X = weights^-1 * dh
    // bottom = weights^-1 * top

//    weights_diff->setToZero();

    Shape weights_shape = weights_diff->shape();
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* bottom_data_error = new Data3d<Dtype>(bottom_data->shape());
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape bottom_shape = bottom_data->shape();
        Shape top_shape = top_data->shape();
        // weights_diff = top * bottom

        //-------------------kernel-------------------
        // calc weights diffs
        for(int kernel = 0; kernel < top_shape.depth(); kernel++) {
            Data3d<Dtype>* weights_diff_new_data = weights_diff_new->Data(kernel);
            for(int depth = 0; depth < bottom_shape.depth(); depth++) {
                for(int top_x = 0, bottom_x = 0/*- pad_w_*/; top_x < top_shape.width(); top_x++,  bottom_x += stride_w_ ) {
                    for(int top_y = 0, bottom_y = 0/*- pad_h_*/; top_y < top_shape.height(); top_y++, bottom_y += stride_h_) {
                        for(int x = 0; x < weights_shape.width(); x++) {
                            for(int y = 0; y < weights_shape.height(); y++) {
                                *weights_diff_new_data->data(x, y, depth) += *top_data->data(top_x, top_y, kernel) * *bottom_data->data(bottom_x + x, bottom_y + y, depth);
                            }
                        }
                    }
                }
            }
        }
        // calc error
        // we are don't admire ativation function
        bottom_data_error->setToZero();
        for(int kernel = 0; kernel < top_shape.depth(); kernel++) {
            Data3d<Dtype>* weights_data = weights->Data(kernel);
            for(int depth = 0; depth < bottom_shape.depth(); depth++) {
                for(int top_x = 0, bottom_x = 0/*- pad_w_*/; top_x < top_shape.width(); top_x++,  bottom_x += stride_w_ ) {
                    for(int top_y = 0, bottom_y = 0/*- pad_h_*/; top_y < top_shape.height(); top_y++, bottom_y += stride_h_) {

                        for(int x = 0; x < weights_shape.width(); x++) {
                            for(int y = 0; y < weights_shape.height(); y++) {
                                *bottom_data_error->data(bottom_x + x, bottom_y + y, depth) += *weights_data->data(x, y, depth) * *top_data->data(top_x, top_y, kernel);
                            }
                        }
                    }
                }
            }
        }
        //-------------ReLU derivation----------------
        for(int depth = 0; depth < bottom_shape.depth(); depth++) {
            for(int bottom_x = 0; bottom_x < bottom_shape.width(); bottom_x ++) {
                for(int bottom_y = 0; bottom_y < bottom_shape.height(); bottom_y ++) {
                    if(isnan(*bottom_data_error->data(bottom_x, bottom_y , depth))) {
                        *bottom_data->data(bottom_x, bottom_y , depth) = 0;
                    } else {
                        *bottom_data->data(bottom_x, bottom_y , depth) = derivate_act_func(*bottom_data->data(bottom_x, bottom_y , depth), activation_) * *bottom_data_error->data(bottom_x, bottom_y, depth);
                    }
                }
            }
        }
        delete bottom_data_error;
    }

    // update weights
    for(int kernel = 0; kernel < weights_shape.batch(); kernel++) {
        Data3d<Dtype>* weights_data = weights->Data(kernel);
        Data3d<Dtype>* weights_diff_data = weights_diff->Data(kernel);
        Data3d<Dtype>* weights_diff_new_data = weights_diff_new->Data(kernel);

        for(int depth = 0; depth < weights_shape.depth(); depth++) {
            for(int x = 0; x < weights_shape.width(); x++) {
                for(int y = 0; y < weights_shape.height(); y++) {
                    if(!isnan(*weights_diff_new_data->data(x, y, depth))) {
                        *weights_diff_data->data(x, y, depth) = this->momentum_ * *weights_diff_data->data(x, y, depth) - this->lr_rate_ * (*weights_diff_new_data->data(x, y, depth) / bottom->batch_size());
//                        *weights_diff_data->data(x, y, depth) = this->momentum_ * *weights_diff_data->data(x, y, depth) - this->lr_rate_ * *weights_diff_new_data->data(x, y, depth);

                        *weights_data->data(x, y, depth) += *weights_diff_data->data(x, y, depth) + this->lr_rate_ * this->weight_decay_ *  *weights_data->data(x, y, depth);
//                        *weights_data->data(x,y, depth) -= (*weights_diff_data->data(x, y, depth) + *weights_data->data(x, y, depth) * this->weight_decay_) * this->lr_rate_;
 //                            *weights_data->data(x,y, depth) -= *weights_diff_data->data(x, y, depth) * this->lr_rate_;
                        if(isnan(*weights_data->data(x, y, depth))) {
                            *weights_data->data(x, y, depth) = distribution(generator) * 0.01;
                        }
                    }
                }
            }
        }
    }
    delete weights_diff_new;
    return &this->bottom_;
}


// init weights as recomended at: http://cs231n.github.io/neural-networks-2/#init
template<typename Dtype>
void
ConvolutionalLayer<Dtype>::initWeights() {
    Blob<Dtype>* weights = this->weights_;
    Shape weights_shape = this->weights_->shape();
//    Shape top_shape = this->top_[0]->shape();

    Shape bottom_shape = this->bottom_[0]->shape();
    Shape top_shape = this->top_[0]->shape();

    int max_num =  weights_shape.width() * weights_shape.height() * weights_shape.depth(); // * weights_shape.batch(); // + top_shape.width() * top_shape.height() * top_shape.depth();

//    int max_num = bottom_shape.width() * bottom_shape.height() * bottom_shape.depth() + top_shape.width() * top_shape.height() * top_shape.depth();
    for(int k = 0; k < weights_shape.batch(); k++) {
       for(int c = 0; c < weights_shape.depth(); c++) {
           for(int x = 0; x < weights_shape.width(); x++) {
               for(int y = 0; y < weights_shape.height(); y++) {
//                   *weights->data(k, x, y, c) = (Dtype)(distribution(generator)) * 0.01;
//                   for(int y = 0; y < weights_shape.height(); y++) {
//                       if(activation_ == ReLU || activation_ == leakyReLU) {
                           *weights->data(k, x, y, c) = (Dtype)(distribution(generator)) * sqrt( 2.0 / max_num);
//                       } else {
//                           *weights->data(k, x, y, c) = (Dtype)(distribution(generator)) / sqrt(max_num);
//                       }
//                   }
               }
           }
       }
    }
}

INSTANTIATE_CLASS(ConvolutionalLayer);
} // namespace danknet

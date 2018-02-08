#include "pooling_layer.h"


namespace danknet {

template<typename Dtype>
PoolingLayer<Dtype>::PoolingLayer(int kernel_w, int kernel_h,
                                  int stride_w, int stride_h,
                                  int pad_w, int pad_h,
                                  string name,
                                  vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top)
    : Layer<Dtype>(name, bottom, top) {
      kernel_w_ = kernel_w;
      kernel_h_ = kernel_h;
      stride_w_ = stride_w;
      stride_h_ = stride_h;
      pad_w_ = pad_w;
      pad_h_ = pad_h;

      //-------------copy bottom vector-------------
      this->bottom_ = bottom;

      //-------------create top vector--------------
      //(input_dim + 2 * pad - kernel_size) / stride + 1;
      int out_w = (this->bottom_[0]->shape().width() + /*2 **/ pad_w_ - kernel_w_) / stride_w_ + 1;
      int out_h = (this->bottom_[0]->shape().height() + /*2 **/ pad_h_ - kernel_h_) / stride_h_ + 1;
      this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(out_w, out_h, this->bottom_[0]->shape().depth(), this->bottom_[0]->shape().batch())));
      top = this->top_;

  }

template<typename Dtype>
vector<Blob<Dtype>*>*
PoolingLayer<Dtype>::Forward() {
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape top_shape = top_data->shape();
        for(int depth = 0; depth < top_shape.depth(); depth++) {
            for(int out_x = 0, inp_x = 0/*- pad_w_*/; out_x < top_shape.width(); out_x++,  inp_x += stride_w_ ) {
                for(int out_y = 0, inp_y = 0/*- pad_h_*/; out_y < top_shape.height(); out_y++, inp_y += stride_h_) {
                    Dtype val = numeric_limits<Dtype>::min();
                    //--------max pooling-----------
                    for(int x = 0; x < kernel_w_; x++) {
                        for(int y = 0; y < kernel_h_; y++) {
                            if(*bottom_data->data(inp_x + x, inp_y + y, depth) > val) {
                                val = *bottom_data->data(inp_x + x, inp_y + y, depth);
                            }
                        }
                    }
                    *top_data->data(out_x, out_y, depth) = val;
                }
            }
        }
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
PoolingLayer<Dtype>::Backward() {
    Blob<Dtype>* bottom = this->bottom_[0];
    Blob<Dtype>* top = this->top_[0];
    //-------------------batch--------------------
    for(int batch = 0; batch < bottom->batch_size(); batch++) {
        Data3d<Dtype>* bottom_data = bottom->Data(batch);
        Data3d<Dtype>* top_data = top->Data(batch);
        Shape top_shape = top_data->shape();
        for(int depth = 0; depth < top_shape.depth(); depth++) {
            for(int out_x = 0, inp_x = 0/*- pad_w_*/; out_x < top_shape.width(); out_x++,  inp_x += stride_w_ ) {
                for(int out_y = 0, inp_y = 0/*- pad_h_*/; out_y < top_shape.height(); out_y++, inp_y += stride_h_) {
                    Dtype val = numeric_limits<Dtype>::min();
                    //--------max pooling-----------
                    for(int x = 0; x < kernel_w_; x++) {
                        for(int y = 0; y < kernel_h_; y++) {
                            if(*bottom_data->data(inp_x + x, inp_y + y, depth) > val) {
                                val = *bottom_data->data(inp_x + x, inp_y + y, depth);
                            }
                        }
                    }
                    for(int x = 0; x < kernel_w_; x++) {
                        for(int y = 0; y < kernel_h_; y++) {
                            *bottom_data->data(inp_x + x, inp_y + y, depth) = *top_data->data(inp_x + x, inp_y + y, depth);
//                            if(*bottom_data->data(inp_x + x, inp_y + y, depth) != val) {
//                                *bottom_data->data(inp_x + x, inp_y + y, depth) = 0;
//                            } else {
//                            }
                        }
                    }
                }
            }
        }
    }
    return &this->bottom_;
}
INSTANTIATE_CLASS(PoolingLayer);
} // namespace danknet

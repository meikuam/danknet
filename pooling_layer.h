#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

namespace danknet {

//Pooling layer

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(int kernel_w, int kernel_h,
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
    }

    virtual inline layertype type() const {return Pooling_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
};

} // namespace danknet
#endif // POOLING_LAYER_H

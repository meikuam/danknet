#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

namespace danknet {

//Pooling layer

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(int kernel_w, int kernel_h,
                        int depth,
                        int stride_w, int stride_h,
                        int pad_w, int pad_h,
                        string name,
                        vector<string> bottom, vector<string> top)
      : Layer<Dtype>(name, bottom, top) {
        kernel_w_ = kernel_w;
        kernel_h_ = kernel_h;
        depth_ = depth;
        stride_w_ = stride_w;
        stride_h_ = stride_h;
        pad_w_ = pad_w;
        pad_h_ = pad_h;
    }

    virtual inline layertype type() const {return Pooling_Layer; }

    virtual void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int depth_;
//    int height_, width_;
};

} // namespace danknet
#endif // POOLING_LAYER_H

#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

namespace danknet {

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer()
      : Layer<Dtype>() {}

    virtual inline layertype type() const {return Pooling_Layer; }

    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int channels_;
    int height_, width_;
};

} // namespace danknet
#endif // POOLING_LAYER_H

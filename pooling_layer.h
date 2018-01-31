#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

namespace danknet {

template <typename Btype, typename Ttype>
class PoolingLayer : public Layer<Btype, Ttype> {
 public:
  explicit PoolingLayer()
      : Layer<Btype, Ttype>() {}

    virtual inline layertype type() const {return Pooling; }

    virtual void Forward(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);
    virtual void Backward(const vector<Data2d<Btype>*>& top, const vector<Data2d<Ttype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int channels_;
    int height_, width_;
};

} // namespace danknet
#endif // POOLING_LAYER_H

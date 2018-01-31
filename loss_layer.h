#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "layer.h"

namespace danknet {

template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer()
      : Layer<Dtype>() {}

    virtual inline layertype type() const {return Loss_Layer; }

    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);

private:
};

} // namespace danknet
#endif // LOSS_LAYER_H

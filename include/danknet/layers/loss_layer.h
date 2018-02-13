#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "layer.h"

namespace danknet {

//Loss layer

template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(Dtype lr_rate,
                     string name,
                     vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline layertype type() const {return Loss_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();

private:
};

} // namespace danknet
#endif // LOSS_LAYER_H

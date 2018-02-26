#ifndef SOFTMAX_LOSS_LAYER_H
#define SOFTMAX_LOSS_LAYER_H

#include "layer.h"

namespace danknet {

//Softmax Loss layer

template <typename Dtype>
class SoftmaxLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLossLayer(Activation activation,
                     string name,
                     vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline Layertype type() const {return Softmax_Loss_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();

private:
    Activation activation_;
};

} // namespace danknet
#endif // SOFTMAX_LOSS_LAYER_H

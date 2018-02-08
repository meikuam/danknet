#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

namespace danknet {

//Softmax layer

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(string name,
                     vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline layertype type() const {return Softmax_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();

private:
};

} // namespace danknet
#endif // SOFTMAX_LAYER_H

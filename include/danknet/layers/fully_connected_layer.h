#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"

namespace danknet {

//Fully connected layer (inner product layer)

template <typename Dtype>
class FullyConectedLayer : public Layer<Dtype> {
 public:
  explicit FullyConectedLayer(int units,
                              string name,
                              vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline layertype type() const {return Fully_Connected_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();

private:
    int units_;
};

} // namespace danknet

#endif // FULLY_CONNECTED_LAYER_H

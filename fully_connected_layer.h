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
                              vector<string> bottom, vector<string> top)
      : Layer<Dtype>(name, bottom, top) {
        units_ = units;
    }

    virtual inline layertype type() const {return Fully_Connected_Layer; }

    virtual void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

private:
    int units_;
};

} // namespace danknet

#endif // FULLY_CONNECTED_LAYER_H

#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"
#include <random>
#include <chrono>

namespace danknet {

//Fully connected layer (inner product layer)

template <typename Dtype>
class FullyConnectedLayer : public Layer<Dtype> {
 public:
  explicit FullyConnectedLayer(int units,
                              Dtype lr_rate,
                              string name,
                              vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline layertype type() const {return Fully_Connected_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
    void initWeights();
private:
    int units_;

    Blob<Dtype>*         weights_diff_;

    std::default_random_engine generator;
    std::uniform_real_distribution<Dtype> distribution;
};

} // namespace danknet

#endif // FULLY_CONNECTED_LAYER_H

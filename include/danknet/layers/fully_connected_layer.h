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
                               Activation activation,
                               string name,
                               vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline Layertype type() const {return Fully_Connected_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
    void initWeights();
private:
    int units_;
    Activation activation_;

    Blob<Dtype>*         weights_diff_;

    std::default_random_engine generator;
//    std::uniform_real_distribution<Dtype> distribution;
    std::normal_distribution<Dtype> distribution;
};

} // namespace danknet

#endif // FULLY_CONNECTED_LAYER_H

#ifndef KOHONEN_LAYER_H
#define KOHONEN_LAYER_H

#include "layer.h"
#include <random>
#include <chrono>

namespace danknet {

//Kohonen layer (inner product layer)

template <typename Dtype>
class KohonenLayer : public Layer<Dtype> {
 public:
  explicit KohonenLayer(string name,
                        vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline Layertype type() const {return Kohonen_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
    void initWeights();
private:
    int width_,
        height_,
        depth_;

    Blob<Dtype>*         weights_diff_;

    std::default_random_engine generator;
    std::normal_distribution<Dtype> distribution;
};

} // namespace danknet

#endif // KOHONEN_LAYER_H

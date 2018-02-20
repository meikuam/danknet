#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H
#include "layer.h"

#include <random>
#include <chrono>

namespace danknet {

//Convolutional layer
// This layer have one bottom blob
template <typename Dtype>
class ConvolutionalLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionalLayer(int kernel_w, int kernel_h,
                              int depth, int kernels,
                              int stride_w, int stride_h,
                              int pad_w, int pad_h,
                              Activation activation,
                              string name,
                              vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual inline Layertype type() const {return Convolutional_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
    void initWeights();
private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int kernels_, depth_;

    Activation activation_;

    Blob<Dtype>*         weights_diff_;

    std::default_random_engine generator;
//    std::uniform_real_distribution<Dtype> distribution;
    std::normal_distribution<Dtype> distribution;
};

} // namespace danknet
#endif // CONVOLUTIONAL_LAYER_H

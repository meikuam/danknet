#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H
#include "layer.h"

namespace danknet {

template <typename Dtype>
class ConvolutionalLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionalLayer()
      : Layer<Dtype>() {}

    virtual inline layertype type() const {return Convolutional_Layer; }
    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
//    int pad_h_, pad_w_;
    int depth_;
    int height_, width_;
};

} // namespace danknet
#endif // CONVOLUTIONAL_LAYER_H

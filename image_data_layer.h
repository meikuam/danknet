#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H
#include "layer.h"


namespace danknet {


template <typename Dtype>
class ImageDataLayer : public Layer<Dtype> {
 public:
  explicit ImageDataLayer()
      : Layer<Dtype>() {}


    virtual inline layertype type() const {return Image_Data_Layer; }
    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);

private:

};

} // namespace danknet
#endif // IMAGE_DATA_LAYER_H

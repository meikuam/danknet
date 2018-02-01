#ifndef DATA_LAYER_H
#define DATA_LAYER_H
#include "layer.h"


namespace danknet {


template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(int width, int height, int depth, string name, vector<string> top, vector<string> bottom)
      : Layer<Dtype>(name, top, bottom) {}


    virtual inline layertype type() const {return Image_Data_Layer; }
    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);

private:
    int width_, height_;
    int depth_;
};

} // namespace danknet
#endif // DATA_LAYER_H

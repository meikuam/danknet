#ifndef DATA_LAYER_H
#define DATA_LAYER_H
#include "layer.h"


namespace danknet {

//Layer that stores some data and feed bottom to top

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(int width, int height,
                     int depth,
                     string name,
                     vector<string> bottom, vector<string> top)
      : Layer<Dtype>(name, bottom, top) {
        width_ = width;
        height_ = height;
        depth_ = depth;
    }


    virtual inline layertype type() const {return Data_Layer; }
    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
//    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

//    virtual void LayerSetUp(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);

    void load_batch(vector<Data2d<Dtype>*>& batch);

private:
    int width_, height_;
    int depth_;
    vector<Data2d<Dtype>*> batch_;
};

} // namespace danknet
#endif // DATA_LAYER_H

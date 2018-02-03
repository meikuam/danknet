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


    virtual void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    void load_batch(vector<Blob<Dtype>*>& batch);

private:
    int width_, height_;
    int depth_;
    vector<Data3d<Dtype>*> batch_;
};

} // namespace danknet
#endif // DATA_LAYER_H

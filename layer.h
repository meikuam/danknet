#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>

#include "data2d.h"

using namespace std;


namespace danknet {


enum layertype {
    BaseType,
    Convolutional_Layer,
    Fully_Connected_Layer,
    Image_Data_Layer,
    Pooling_Layer,
    Loss_Layer
};

template <typename Dtype>
class Layer {
public:
    explicit Layer();

    virtual ~Layer() {}

    string          name_;

    virtual inline layertype type() const {return BaseType; }
    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(vector<Data2d<Dtype>*>& bottom, vector<Data2d<Dtype>*>& top);
};

} // namespace danknet
#endif // LAYER_H

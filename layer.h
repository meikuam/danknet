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
    Data_Layer,
    Pooling_Layer,
    Loss_Layer
};

template <typename Dtype>
class Layer {
public:
    explicit Layer(string name, vector<string> bottom, vector<string> top);

    virtual ~Layer() {}


    virtual inline layertype type() const {return BaseType; }
    virtual void Forward(const vector<Data2d<Dtype>*>& bottom, const vector<Data2d<Dtype>*>& top);
    virtual void Backward(const vector<Data2d<Dtype>*>& top, const vector<Data2d<Dtype>*>& bottom);

    virtual void LayerSetUp(vector<Data2d<Dtype>*>& bottom, vector<Data2d<Dtype>*>& top);

    inline vector<string> get_bottom_names() {
        return bottom_;
    }
    inline vector<string> get_top_names() {
        return top_;
    }

private:
    string          name_;
    vector<string>  top_;
    vector<string>  bottom_;

    Data2d<Dtype>* weights_;

};

} // namespace danknet
#endif // LAYER_H

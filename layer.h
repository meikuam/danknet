#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>

#include "data.h"
#include "common.h"

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
private:
    string          name_;

    vector<Blob<Dtype>*> bottom_;
    vector<Blob<Dtype>*> top_;
    vector<Blob<Dtype>*> weights_;
public:
    explicit Layer(string name, vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual ~Layer() {}

    virtual inline layertype type() const {return BaseType; }
    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();


    inline vector<Blob<Dtype>*>* get_bottom() { return &bottom_; }
    inline vector<Blob<Dtype>*>* get_top() { return &top_; }
    inline vector<Blob<Dtype>*>* get_weights(){ return &weights_;}



};

} // namespace danknet
#endif // LAYER_H

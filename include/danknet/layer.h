#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>

#include "data.h"
#include "common.h"

using namespace std;


namespace danknet {


// -----------------Layer---------------------
// Layer have bottom data as input and top
// data as output. top data is created in
// constructor of the Layer.

//------------vector<Blob<Dtype>*>------------
// vector of Blobs is a container for
// references to multiple input data.
// For example LossLayer that have
// two input data Blobs (predicted data,
// label).

//-----------------Blob<Dtype>*---------------
// Blob is a container for batches of
// references to Data3d objects.


enum layertype {
    BaseType,
    Convolutional_Layer,
    Fully_Connected_Layer,
    Pooling_Layer,
    Loss_Layer,
    Softmax_Layer
};


template <typename Dtype>
class Layer {
protected:
    string name_;

//    vector<Layer<Dtype>*>   top_layers_,
//                            bottom_layers_;

    vector<Blob<Dtype>*> top_;
    vector<Blob<Dtype>*> bottom_;
    Blob<Dtype>*         weights_;

    Dtype                lr_rate_;

public:
    explicit Layer(string name, vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

    virtual ~Layer() {}

    virtual inline layertype type() const {return BaseType; }
    virtual inline string name() {return name_; }
    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();

    inline vector<Blob<Dtype>*>* top() { return &top_; }
    inline vector<Blob<Dtype>*>* bottom() { return &bottom_; }
    inline Blob<Dtype>*          weights(){ return weights_;}
    inline void setLrRate(Dtype lr_rate) { lr_rate_ = lr_rate; }

//    inline bool has_top_layers() { return top_layers_.size() > 0 ? true : false; }
//    inline bool has_bottom_layers() { return top_layers_.size() > 0 ? true : false; }

//    inline vector<Layer<Dtype>*>* top_layers() { return &top_layers_; }
//    inline vector<Layer<Dtype>*>* bottom_layers() { return &bottom_layers_; }

//    inline void add_top_layer(Layer<Dtype>* top) { top_layers_.push_back(top); }
//    inline void add_bottom_layer(Layer<Dtype>* bottom) { bottom_layers_.push_back(bottom); }
};

} // namespace danknet
#endif // LAYER_H

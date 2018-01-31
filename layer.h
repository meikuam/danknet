#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>

#include "data2d.h"

using namespace std;


namespace danknet {

enum Phase {
    TRAIN,
    TEST
};

enum layertype {
    BaseType,
    Convolutional,
    Fully_Connected,
    Image_Data,
    Pooling
};

template <typename Btype, typename Ttype>
class Layer {
public:
    explicit Layer();

    virtual ~Layer() {}

    Phase           phase_;
    string          name_;

    virtual inline layertype type() const {return BaseType; }
    virtual void Forward(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);
    virtual void Backward(const vector<Data2d<Btype>*>& top, const vector<Data2d<Ttype>*>& bottom);

    virtual void LayerSetUp(vector<Data2d<Btype>*>& bottom, vector<Data2d<Ttype>*>& top);
};

} // namespace danknet
#endif // LAYER_H

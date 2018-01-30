#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>

#include "data2d.h"

using namespace std;
enum phase {
    TRAIN,
    TEST
};

template <class Btype, class Ttype>
class Layer {
public:
    explicit Layer();

    phase           phase_;
    string          name_;
    virtual void Forward(vector<Data2d*>& bottom, vector<Data2d*>& top);
    virtual void Backward(vector<Data2d*>& top, vector<Data2d*>& bottom);

    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
};

#endif // LAYER_H

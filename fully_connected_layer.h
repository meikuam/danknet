#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"

namespace danknet {

template <typename Btype, typename Ttype>
class FullyConectedLayer : public Layer<Btype, Ttype> {
 public:
  explicit FullyConectedLayer()
      : Layer<Btype, Ttype>() {}

    virtual inline layertype type() const {return Fully_Connected; }
    virtual void Forward(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);
    virtual void Backward(const vector<Data2d<Btype>*>& top, const vector<Data2d<Ttype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);

private:

};

} // namespace danknet

#endif // FULLY_CONNECTED_LAYER_H

#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H
#include "layer.h"


namespace danknet {


template <typename Btype, typename Ttype>
class ImageDataLayer : public Layer<Btype, Ttype> {
 public:
  explicit ImageDataLayer()
      : Layer<Btype, Ttype>() {}


    virtual inline layertype type() const {return Image_Data; }
    virtual void Forward(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);
    virtual void Backward(const vector<Data2d<Btype>*>& top, const vector<Data2d<Ttype>*>& bottom);

    virtual void LayerSetUp(const vector<Data2d<Btype>*>& bottom, const vector<Data2d<Ttype>*>& top);

private:

};

} // namespace danknet
#endif // IMAGE_DATA_LAYER_H

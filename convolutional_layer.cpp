#include "convolutional_layer.h"

namespace danknet {

template<typename Btype, typename Ttype>
void
ConvolutionalLayer<Btype, Ttype>::LayerSetUp(const vector<Data2d<Btype>*>& bottom,
                                             const vector<Data2d<Ttype>*>& top) {

}


template<typename Btype, typename Ttype>
void
ConvolutionalLayer<Btype, Ttype>::Forward(const vector<Data2d<Btype>*>& bottom,
                                             const vector<Data2d<Ttype>*>& top) {

}

template<typename Btype, typename Ttype>
void
ConvolutionalLayer<Btype, Ttype>::Backward(const vector<Data2d<Btype>*>& bottom,
                                             const vector<Data2d<Ttype>*>& top) {

}
} // namespace danknet

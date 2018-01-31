#include "pooling_layer.h"


namespace danknet {

template<typename Btype, typename Ttype>
void
PoolingLayer<Btype, Ttype>::LayerSetUp(const vector<Data2d<Btype>*>& bottom,
                                       const vector<Data2d<Ttype>*>& top) {

}


template<typename Btype, typename Ttype>
void
PoolingLayer<Btype, Ttype>::Forward(const vector<Data2d<Btype>*>& bottom,
                                    const vector<Data2d<Ttype>*>& top) {

}

template<typename Btype, typename Ttype>
void
PoolingLayer<Btype, Ttype>::Backward(const vector<Data2d<Btype>*>& bottom,
                                     const vector<Data2d<Ttype>*>& top) {

}
} // namespace danknet

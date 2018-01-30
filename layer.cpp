#include "layer.h"


template<class Btype, class Ttype>
Layer<Btype, Ttype>::Layer() {
    phase_ = TRAIN;
    name = "base_layer";
}

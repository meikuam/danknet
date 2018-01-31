#include "layer.h"

namespace danknet {

template<class Btype, class Ttype>
Layer<Btype, Ttype>::Layer() {
    phase_ = TRAIN;
    name_ = "base_layer";
}

} // namespace danknet

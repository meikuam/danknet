#include "net.h"

namespace danknet {

template<typename Dtype>
Net<Dtype>::Net() {
    Init();
}


template<typename Dtype>
const vector<Data2d<Dtype>*>& Net<Dtype>::Forward() {

}

template<typename Dtype>
void Net<Dtype>::Backward() {

}

template<typename Dtype>
void Net<Dtype>::WeightsFromHDF5(string filename) {

}

template<typename Dtype>
void Net<Dtype>::WeightsToHDF5(string filename) {

}

} // namespace danknet

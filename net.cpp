#include "net.h"

namespace danknet {

template<typename Dtype>
Net<Dtype>::Net() {
//    Init();
}


template<typename Dtype>
void Net<Dtype>::Forward(const vector<Data3d<danknet::Dtype> *> &bottom, const vector<Data3d<danknet::Dtype> *> &top) {
    for(int i = 0; i < layers_.size(); i++) {
        layers_[i].Forward(bottom_data_[i], top_data_[i]);
    }
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

template<typename Dtype>
void Net<Dtype>::AddLayer(Layer<Dtype>* layer) {
    //TODO: bottom_data_ / top_data_ creation
    layers_.push_back(layer);
}
} // namespace danknet

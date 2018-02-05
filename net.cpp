#include "net.h"

namespace danknet {


template<typename Dtype>
void Net<Dtype>::Forward() {
    for(int i = 0; i < layers_.size(); i++) {
        layers_[i]->Forward();
    }
}

template<typename Dtype>
void Net<Dtype>::Backward() {
    for(int i = layers_.size() - 1; i >= 0; i--) {
        layers_[i]->Backward();
    }
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


//template<typename Dtype>
//void Net<Dtype>::AddBlob(Blob<Dtype>& blob) {

//}

INSTANTIATE_CLASS(Net);
} // namespace danknet

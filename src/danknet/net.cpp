#include "net.h"
#include <iostream>
using namespace std;
namespace danknet {


template<typename Dtype>
void Net<Dtype>::Forward() {
    //don't step on snek
    //TODO: this is solver operation
    switch (phase_) {
    case TRAIN:
        train_iters++;
        if(train_iters % step_size_ == 0) {
            lr_rate(lr_rate_ * gamma_);
        }
        break;
    case TEST:
        test_iters++;
        break;
    }

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
    H5File hdf5file(filename, H5F_ACC_RDONLY);
    int numObjects = hdf5file.getNumObjs();
//    cout<<"WeightsFromHDF5: numObjects: "<<numObjects<<endl;

//    cout<<hdf5file.getObjnameByIdx(0)<<endl;
    for(int i = 0; i < numObjects; i++) {
        string name = hdf5file.getObjnameByIdx(i);
        for(int l = 0; l < layers_.size(); l++) {
            switch (layers_[l]->type()) {
            case Convolutional_Layer:
            {
                if(layers_[l]->weights()->name() == name) {
                    DataSet dataset = hdf5file.openDataSet(name);
                    Blob<Dtype>* layer_weights = layers_[l]->weights();
                    Shape weights_shape = layer_weights->shape();

                    Dtype* weights = new Dtype[weights_shape.width() * weights_shape.height() * weights_shape.depth() * weights_shape.batch() ];
                    dataset.read((void*)weights, dataset.getDataType());
//                    cout<<"weights: "<< name<<endl;

                    for(int k = 0; k < weights_shape.batch(); k++) {
                        for(int c = 0; c < weights_shape.depth(); c++) {
                            for(int x = 0; x < weights_shape.width(); x++) {
                                for(int y = 0; y < weights_shape.height(); y++) {
                                    *layer_weights->data(k, x, y, c) = weights[k * weights_shape.depth() * weights_shape.width() *  weights_shape.height() +
                                                                               c * weights_shape.width() *  weights_shape.height() +
                                                                               x * weights_shape.height() +
                                                                               y];
                                    //                                         [out_fm][in_fm][k_size_x][k_size_y];
                                }
                            }
                        }
                    }
                    dataset.close();
                }
                break;
            }
            case Kohonen_Layer:
            case Fully_Connected_Layer:
            {
                if(layers_[l]->weights()->name() == name) {
                    DataSet dataset = hdf5file.openDataSet(name);
                    Blob<Dtype>* layer_weights = layers_[l]->weights();
                    Shape weights_shape = layer_weights->shape();

                    Dtype* weights = new Dtype[weights_shape.width() * weights_shape.height() * weights_shape.depth() * weights_shape.batch() ];
                    dataset.read((void*)weights, dataset.getDataType());
                    for(int k = 0; k < weights_shape.batch(); k++) {
                        for(int c = 0; c < weights_shape.depth(); c++) {
                            for(int x = 0; x < weights_shape.width(); x++) {
                                for(int y = 0; y < weights_shape.height(); y++) {
                                    *layer_weights->data(k, x, y, c) = weights[k * weights_shape.depth() * weights_shape.width() *  weights_shape.height() +
                                                                               c * weights_shape.width() *  weights_shape.height() +
                                                                               x * weights_shape.height() +
                                                                               y];
                                    //                                         [out_fm][in_fm][k_size_x][k_size_y];
                                }
                            }
                        }
                    }
                    dataset.close();
                }
                break;
            }
            }
        }
    }
    hdf5file.close();
}

template<typename Dtype>
void Net<Dtype>::WeightsToHDF5(string filename) {
    H5File hdf5file(filename, H5F_ACC_TRUNC);

    for(int l = 0; l < layers_.size(); l++) {
        switch (layers_[l]->type()) {
        case Convolutional_Layer:
        {
            Blob<Dtype>* layer_weights = layers_[l]->weights();
            Shape weights_shape = layer_weights->shape();
            hsize_t dims[4] = {(hsize_t)weights_shape.batch(),
                               (hsize_t)weights_shape.depth(),
                               (hsize_t)weights_shape.width(),
                               (hsize_t)weights_shape.height()};

            DataSet dataset = hdf5file.createDataSet(layer_weights->name(),
                                                     PredType::NATIVE_DOUBLE,
                                                     DataSpace(4, dims));

            Dtype* weights = new Dtype[weights_shape.width() * weights_shape.height() * weights_shape.depth() * weights_shape.batch() ];
            for(int k = 0; k < weights_shape.batch(); k++) {
                for(int c = 0; c < weights_shape.depth(); c++) {
                    for(int x = 0; x < weights_shape.width(); x++) {
                        for(int y = 0; y < weights_shape.height(); y++) {
                            weights[k * weights_shape.depth() * weights_shape.width() *  weights_shape.height() +
                                    c * weights_shape.width() *  weights_shape.height() +
                                    x * weights_shape.height() +
                                    y] = *layer_weights->data(k, x, y, c);
                            //                                [out_fm][in_fm][k_size_x][k_size_y];
                        }
                    }
                }
            }

            dataset.write(weights, dataset.getDataType());
            dataset.close();
            break;
        }
        case Kohonen_Layer:
        case Fully_Connected_Layer:
        {
            Blob<Dtype>* layer_weights = layers_[l]->weights();
            Shape weights_shape = layer_weights->shape();
            hsize_t *dims;
//            switch (weights_shape.dims()) {
//            case 1:
//                dims = new hsize_t[1] {(hsize_t)weights_shape.batch()};
//                break;
//            case 2:
//                dims = new hsize_t[2] {(hsize_t)weights_shape.batch(),
//                                       (hsize_t)weights_shape.depth()};
//                break;
//            case 3:
//                dims = new hsize_t[3] {(hsize_t)weights_shape.batch(),
//                                       (hsize_t)weights_shape.depth(),
//                                       (hsize_t)weights_shape.width()};
//                break;
//            case 4:
//                dims = new hsize_t[4] {(hsize_t)weights_shape.batch(),
//                                       (hsize_t)weights_shape.depth(),
//                                       (hsize_t)weights_shape.width(),
//                                       (hsize_t)weights_shape.height()};
//                break;
//            }

            dims = new hsize_t[4] {(hsize_t)weights_shape.batch(),
                                   (hsize_t)weights_shape.depth(),
                                   (hsize_t)weights_shape.width(),
                                   (hsize_t)weights_shape.height()};

            DataSet dataset = hdf5file.createDataSet(layer_weights->name(),
                                                     PredType::NATIVE_DOUBLE,
                                                     DataSpace(4, dims));

            Dtype* weights = new Dtype[weights_shape.width() * weights_shape.height() * weights_shape.depth() * weights_shape.batch() ];
            for(int k = 0; k < weights_shape.batch(); k++) {
                for(int c = 0; c < weights_shape.depth(); c++) {
                    for(int x = 0; x < weights_shape.width(); x++) {
                        for(int y = 0; y < weights_shape.height(); y++) {
                            weights[k * weights_shape.depth() * weights_shape.width() *  weights_shape.height() +
                                    c * weights_shape.width() *  weights_shape.height() +
                                    x * weights_shape.height() +
                                    y] = *layer_weights->data(k, x, y, c);
                            //                                [out_fm][in_fm][k_size_x][k_size_y];
                        }
                    }
                }
            }
            dataset.write(weights, dataset.getDataType());
            dataset.close();
            break;
        }
        }
    }
    hdf5file.close();
}

template<typename Dtype>
void Net<Dtype>::AddLayer(Layer<Dtype>* layer) {
    layers_.push_back(layer);

}


//template<typename Dtype>
//void Net<Dtype>::AddBlob(Blob<Dtype>& blob) {

//}

INSTANTIATE_CLASS(Net);
} // namespace danknet

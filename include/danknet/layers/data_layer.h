#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H
#include "layer.h"

#include <random>
#include <chrono>
#include <fstream>

#include <iostream>
using namespace std;
namespace danknet {

//Convolutional layer
// This layer have one bottom blob
template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(int depth,
                          int batches,
                          int labels,
                          string train_path,
                          string test_path,
                          string name,
                          vector<Blob<Dtype>*>& top);

    virtual inline Layertype type() const {return Data_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
private:
    string  train_path_,
            test_path_;

    vector<Dtype> train_data_;
    vector<Dtype> test_data_;

    int width_,
        height_,
        depth_,
        labels_,
        batches_;

    int current_train_item_,
        current_test_item_;

    int train_len_,
        test_len_;

};

} // namespace danknet
#endif // IMAGE_DATA_LAYER_H

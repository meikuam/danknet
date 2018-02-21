#ifndef DATA_LAYER_H
#define DATA_LAYER_H
#include "layer.h"

#include <fstream>

#include <iostream>
using namespace std;
namespace danknet {

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(int data_depth,
                     int label_depth,
                     int batches,
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

    int data_depth_,
        label_depth_,
        batches_;

    int current_train_item_,
        current_test_item_;

};

} // namespace danknet
#endif // DATA_LAYER_H

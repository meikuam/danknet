#ifndef TEXT_DATA_LAYER_H
#define TEXT_DATA_LAYER_H
#include "layer.h"

#include <fstream>

#include <iostream>
using namespace std;
namespace danknet {

template <typename Dtype>
class TextDataLayer : public Layer<Dtype> {
 public:
  explicit TextDataLayer(int data_depth,
                     int label_depth,
                     int batches,
                     string train_path,
                     string test_path,
                     string name,
                     vector<Blob<Dtype>*>& top);

    virtual inline Layertype type() const {return Text_Data_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
private:
    string  train_path_,
            test_path_;

    vector<Dtype> train_data_;
    vector<Dtype> test_data_;

    vector<int> train_labels_;
    vector<int> test_labels_;

    int data_depth_,
        label_depth_,
        batches_;

    int current_train_item_,
        current_test_item_;

    int current_train_label_,
        current_test_label_;

};

} // namespace danknet
#endif // TEXT_DATA_LAYER_H

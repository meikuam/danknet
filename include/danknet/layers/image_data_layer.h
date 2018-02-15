#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H
#include "layer.h"
#include <QImage>

#include <random>
#include <chrono>
#include <fstream>

#include <iostream>
using namespace std;
namespace danknet {

//Convolutional layer
// This layer have one bottom blob
template <typename Dtype>
class ImageDataLayer : public Layer<Dtype> {
 public:
  explicit ImageDataLayer(int width, int height, int depth,
                          int batches,
                          int labels,
                          string train_path,
                          string test_path,
                          string name,
                          vector<Blob<Dtype>*>& top);

    virtual inline layertype type() const {return Image_Data_Layer; }

    virtual vector<Blob<Dtype>*>* Forward();
    virtual vector<Blob<Dtype>*>* Backward();
private:
    string  train_path_,
            test_path_;

    vector<QString> train_data_;
    vector<QString> test_data_;
    vector<int> train_labels_;
    vector<int> test_labels_;

    int width_,
        height_,
        depth_,
        labels_,
        batches_;

    int current_train_image_,
        current_test_image_;

    int train_images_,
        test_images_;

//    std::default_random_engine generator;
//    std::uniform_real_distribution<Dtype> distribution;
};

} // namespace danknet
#endif // IMAGE_DATA_LAYER_H

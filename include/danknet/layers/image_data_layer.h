#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H
#include "layer.h"
#include <QImage>

#include <random>
#include <chrono>

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

    vector<string> train_data_;
    vector<string> test_data_;
    vector<int> train_labels_;
    vector<int> test_test_;

    int width_,
        height_,
        depth_,
        labels_,
        batches_;

    int current_image;

    int train_images,
        test_images;

//    std::default_random_engine generator;
//    std::uniform_real_distribution<Dtype> distribution;
};

} // namespace danknet
#endif // IMAGE_DATA_LAYER_H

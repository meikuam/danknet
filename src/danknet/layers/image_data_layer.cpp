#include "image_data_layer.h"

namespace danknet {

template<typename Dtype>
ImageDataLayer<Dtype>::ImageDataLayer(int width, int height, int depth,
                                      int batches,
                                      int labels,
                                      string train_path,
                                      string test_path,
                                      string name,
                                      vector<Blob<Dtype>*>& top)
      : Layer<Dtype>(name, *(new vector<Blob<Dtype>*>()), top)//,
//        distribution(-0.1,0.1),
//        generator(std::chrono::system_clock::now().time_since_epoch().count())
{
    width_ = width;
    height_ = height;
    depth_ = depth;
    labels_ = labels;
    batches_ = batches;

    train_path_ = train_path;
    test_path_ = test_path;



      //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(width_, height_, depth_, batches_)));
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_labels", Shape(1, 1, labels_, batches_)));
    top = this->top_;
}

template<typename Dtype>
vector<Blob<Dtype>*>*
ImageDataLayer<Dtype>::Forward() {
    Blob<Dtype>* top_data = this->top_[0];
    Blob<Dtype>* top_labels = this->top_[1];

    //-------------------batch--------------------
    for(int batch = 0; batch < top_data->batch_size(); batch++) {

    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
ImageDataLayer<Dtype>::Backward() {
    return &this->bottom_;
}

INSTANTIATE_CLASS(ImageDataLayer);
} // namespace danknet

#include "data_layer.h"

namespace danknet {

template<typename Dtype>
DataLayer<Dtype>::DataLayer(int depth,
                            int batches,
                            int labels,
                            string train_path,
                            string test_path,
                            string name,
                            vector<Blob<Dtype>*>& top)
      : Layer<Dtype>(name, *(new vector<Blob<Dtype>*>()), top)
{
    depth_ = depth;
    labels_ = labels;
    batches_ = batches;

    train_path_ = train_path;
    test_path_ = test_path;

    //read train data
    train_data_.clear();
    train_len_ = 0;

    ifstream train_file(train_path_);
    string line;
    QString qline;
    while(getline(train_file, line)) {
        qline = QString::fromStdString(line);
        train_data_.push_back((Dtype)(qline.toDouble()));
        train_len_++;
    }
    train_file.close();

    current_train_item_ = 0;

    //read test data
    test_data_.clear();
    test_len_ = 0;

    ifstream test_file(test_path_);
    while(getline(test_file, line)) {
        qline = QString::fromStdString(line);
        test_data_.push_back((Dtype)(qline.toDouble()));
        test_len_++;
    }
    test_file.close();
    current_test_item_ = 0;

    //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(1, 1, depth_, batches_)));
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_labels", Shape(1, 1, labels_, batches_)));
    top = this->top_;


    cout<<"DataLayer: "<<name<<endl;
    cout<<"----------------layer info------------------"<<endl;
    cout<<"depth: "<<depth_<<endl;
    cout<<"batches: "<<batches_<<endl;
    cout<<"labels: "<<labels_<<endl;
    cout<<"train_path_: "<<train_path_<<endl;
    cout<<"test_path_: "<<test_path_<<endl;
    cout<<"train_len_: "<<train_len_<<endl;
    cout<<"test_len_: "<<test_len_<<endl;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
DataLayer<Dtype>::Forward() {
    Blob<Dtype>* top_data = this->top_[0];
    Blob<Dtype>* top_labels = this->top_[1];
    int label;

    //-------------------batch--------------------
    for(int batch = 0; batch < top_data->batch_size(); batch++) {

        switch (this->phase_) {
        case TRAIN:
//            if(current_train_image_ >= train_images_) {
//                current_train_image_ = 0;
//            }
//            for(int d = 0; d < depth_; d++) {
//                *top_data->data(batch, 0, 0, d) = train_data_[current_train_item_++];
//            }
//            label = train_labels_[current_train_image_];
//            current_train_image_++;
            break;
        case TEST:
//            if(current_test_image_ >= test_images_) {
//                current_test_image_ = 0;
//            }

//            test_labels_[current_test_image_];
//            test_data_[current_test_image_];
//            current_test_image_++;
            break;
        }
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
DataLayer<Dtype>::Backward() {
//    return &this->bottom_;
}

INSTANTIATE_CLASS(DataLayer);
} // namespace danknet

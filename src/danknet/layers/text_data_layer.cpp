#include "text_data_layer.h"

namespace danknet {

template<typename Dtype>
TextDataLayer<Dtype>::TextDataLayer(int data_depth,
                                    int label_depth,
                                    int batches,
                                    string train_path,
                                    string test_path,
                                    string name,
                                    vector<Blob<Dtype>*>& top)
      : Layer<Dtype>(name, *(new vector<Blob<Dtype>*>()), top)
{
    //layer params
    data_depth_ = data_depth;
    label_depth_ = label_depth;
    batches_ = batches;

    train_path_ = train_path;
    test_path_ = test_path;

    //read train data
    train_data_.clear();
    train_labels_.clear();

    ifstream train_file(train_path_);
    string line;
    QString qline;
    while(getline(train_file, line)) {
        qline = QString::fromStdString(line);
        auto parts = qline.split(' ');

        for(int i = 0; i < data_depth_; i++) {
            train_data_.push_back((Dtype)(parts[i].toDouble()));
        }
        for(int i = data_depth; i < data_depth_ + label_depth; i++) {
            train_labels_.push_back((Dtype)(parts[i].toInt()));
        }
    }
    train_file.close();
    current_train_item_ = 0;
    current_train_label_ = 0;

    //read test data
    test_data_.clear();
    test_labels_.clear();

    ifstream test_file(test_path_);
    while(getline(test_file, line)) {
        qline = QString::fromStdString(line);
        auto parts = qline.split(' ');
        for(int i = 0; i < data_depth_; i++) {
            test_data_.push_back((Dtype)(parts[i].toDouble()));
        }
        for(int i = data_depth; i < data_depth_ + label_depth; i++) {
            test_labels_.push_back((Dtype)(parts[i].toInt()));
        }
    }
    test_file.close();
    current_test_item_ = 0;
    current_test_label_ = 0;

    //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(1, 1, data_depth_, batches_)));
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_labels", Shape(1, 1, label_depth_, batches_)));
    top = this->top_;


    cout<<"DataLayer: "<<name<<endl;
    cout<<"----------------layer info------------------"<<endl;
    cout<<"data_depth_: "<<data_depth_<<endl;
    cout<<"label_depth_: "<<label_depth_<<endl;
    cout<<"batches_: "<<batches_<<endl;
    cout<<"train_path_: "<<train_path_<<endl;
    cout<<"test_path_: "<<test_path_<<endl;
    cout<<"train_len: "<<train_data_.size()<<endl;
    cout<<"test_len: "<<test_data_.size()<<endl;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
TextDataLayer<Dtype>::Forward() {
    Blob<Dtype>* top_data = this->top_[0];
    Blob<Dtype>* top_label = this->top_[1];

    //-------------------batch--------------------
    for(int batch = 0; batch < top_data->batch_size(); batch++) {
        switch (this->phase_) {
        case TRAIN:
        {
            if(current_train_item_ >= (train_data_.size() / data_depth_)) {
                current_train_item_ = 0;
                current_train_label_ = 0;
            }
            for(int d = 0; d < data_depth_; d++) {
                *top_data->data(batch, 0, 0, d) = train_data_[current_train_item_ * data_depth_ + d];
            }
            for(int l = 0; l < label_depth_; l++) {
                *top_label->data(batch, 0, 0, l) = train_labels_[current_train_label_ * label_depth_ + l];
            }
            current_train_item_++;
            current_train_label_++;
            break;
        }
        case TEST:
        {
            if(current_test_item_ >= (test_data_.size() / data_depth_)) {
                current_test_item_ = 0;
                current_test_label_ = 0;
            }
            for(int d = 0; d < data_depth_; d++) {
                *top_data->data(batch, 0, 0, d) = test_data_[current_test_item_ * data_depth_ + d];
            }
            for(int l = 0; l < label_depth_; l++) {
                *top_label->data(batch, 0, 0, l) = test_labels_[current_test_label_ * label_depth_ + l];
            }
            current_test_item_++;
            current_test_label_++;
            break;
        }
        }
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
TextDataLayer<Dtype>::Backward() {
//    return &this->bottom_;
}

INSTANTIATE_CLASS(TextDataLayer);
} // namespace danknet

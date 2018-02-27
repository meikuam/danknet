#include "data_layer.h"

namespace danknet {

template<typename Dtype>
DataLayer<Dtype>::DataLayer(int data_depth,
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

    ifstream train_file(train_path_);
    string line;
    QString qline;
    while(getline(train_file, line)) {
        qline = QString::fromStdString(line);
        train_data_.push_back((Dtype)(qline.toDouble()));
    }
    train_file.close();
    current_train_item_ = 0;

    //read test data
    test_data_.clear();

    ifstream test_file(test_path_);
    while(getline(test_file, line)) {
        qline = QString::fromStdString(line);
        test_data_.push_back((Dtype)(qline.toDouble()));
    }
    test_file.close();
    current_test_item_ = 0;

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
DataLayer<Dtype>::Forward() {
    Blob<Dtype>* top_data = this->top_[0];
    Blob<Dtype>* top_label = this->top_[1];

    //-------------------batch--------------------
    for(int batch = 0; batch < top_data->batch_size(); batch++) {
        switch (this->phase_) {
        case TRAIN:
            if(current_train_item_ + data_depth_ + label_depth_ >= train_data_.size()) {
                current_train_item_ = 0;
            }
            for(int d = 0; d < data_depth_; d++) {
                *top_data->data(batch, 0, 0, d) = train_data_[current_train_item_ + d];
            }
            for(int l = 0; l < label_depth_; l++) {
                *top_label->data(batch, 0, 0, l) = train_data_[current_train_item_ + data_depth_ + l];
            }
            current_train_item_ += label_depth_;
            break;
        case TEST:
            if(current_test_item_ + data_depth_ + label_depth_ >= test_data_.size()) {
                current_test_item_ = 0;
            }
            for(int d = 0; d < data_depth_; d++) {
                *top_data->data(batch, 0, 0, d) = test_data_[current_test_item_ + d];
            }
            for(int l = 0; l < label_depth_; l++) {
                *top_label->data(batch, 0, 0, l) = test_data_[current_test_item_ + data_depth_ + l];
            }
            current_test_item_ += label_depth_;
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

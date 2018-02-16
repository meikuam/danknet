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

    //read train images and labels
    train_data_.clear();
    train_labels_.clear();
    train_images_ = 0;

    ifstream train_file(train_path_);
    string line;
    QString qline;
    while(getline(train_file, line)) {
        qline = QString::fromStdString(line);
        auto parts = qline.split(' ');
        train_data_.push_back(parts[0]);
        train_labels_.push_back(parts[1].toInt());
        train_images_++;
    }
    train_file.close();
    current_train_image_ = 0;

    //read test images and labels
    test_data_.clear();
    test_labels_.clear();
    test_images_ = 0;

    ifstream test_file(test_path_);
    while(getline(test_file, line)) {
        qline = QString::fromStdString(line);
        auto parts = qline.split(' ');
        test_data_.push_back(parts[0]);
        test_labels_.push_back(parts[1].toInt());
        test_images_++;
    }
    test_file.close();
    current_test_image_ = 0;

    //-------------create top vector--------------
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_data", Shape(width_, height_, depth_, batches_)));
    this->top_.push_back(new Blob<Dtype>(this->name_ + "_labels", Shape(1, 1, labels_, batches_)));
    top = this->top_;


    cout<<"ImageDataLayer: "<<name<<endl;
    cout<<"----------------layer info------------------"<<endl;
    cout<<"width: "<<width_<<endl;
    cout<<"height: "<<height_<<endl;
    cout<<"depth: "<<depth_<<endl;
    cout<<"batches: "<<batches_<<endl;
    cout<<"labels: "<<labels_<<endl;
    cout<<"train_path_: "<<train_path_<<endl;
    cout<<"test_path_: "<<test_path_<<endl;
    cout<<"train_images_: "<<train_images_<<endl;
    cout<<"test_images_: "<<test_images_<<endl;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
ImageDataLayer<Dtype>::Forward() {
    Blob<Dtype>* top_data = this->top_[0];
    Blob<Dtype>* top_labels = this->top_[1];
    int label;
    QString path;

//    cout<<"ImageDataLayer: "<<this->name_<<endl;
//    cout<<"------------------Forward-------------------"<<endl;

    //-------------------batch--------------------
    for(int batch = 0; batch < top_data->batch_size(); batch++) {

        switch (this->phase_) {
        case TRAIN:
            if(current_train_image_ >= train_images_) {
                current_train_image_ = 0;
            }

//            cout<<"phase_: TRAIN"<<endl;
//            cout<<"current_train_image_: "<<current_train_image_<<endl;
//            cout<<"label: "<<QString::number(train_labels_[current_train_image_]).toStdString()<<endl;
//            cout<<"path: "<<train_data_[current_train_image_].toStdString()<<endl;

            label = train_labels_[current_train_image_];
            path = train_data_[current_train_image_];
            current_train_image_++;
            break;
        case TEST:
            if(current_test_image_ >= test_images_) {
                current_test_image_ = 0;
            }

//            cout<<"phase_: TEST"<<endl;
//            cout<<"current_test_image_: "<<current_test_image_<<endl;
//            cout<<"label: "<<QString::number(test_labels_[current_test_image_]).toStdString()<<endl;
//            cout<<"path: "<<test_data_[current_test_image_].toStdString()<<endl;

            label = test_labels_[current_test_image_];
            path = test_data_[current_test_image_];
            current_test_image_++;
            break;
        }
        for(int i = 0; i < labels_; i++) {
            *top_labels->data(batch, 0, 0, i) = (i == label) ? 1 : 0;
        }
        *top_data->Data(batch) = QImage(path);
        Shape sh = top_data->shape();

        int num = sh.width() * sh.height() * sh.depth();
        Dtype* dat = top_data->data(0);
        for(int i = 0; i < num; i++) {
            dat[i] = (Dtype) (dat[i] / 255.0);
        }

//        cout<<"top_data width:"<<top_data->width()<<endl;
//        cout<<"top_data height:"<<top_data->height()<<endl;
//        cout<<"top_data depth:"<<top_data->depth()<<endl;
//        cout<<"top_data batch:"<<top_data->batch_size()<<endl;
    }
    return &this->top_;
}


template<typename Dtype>
vector<Blob<Dtype>*>*
ImageDataLayer<Dtype>::Backward() {
//    return &this->bottom_;
}

INSTANTIATE_CLASS(ImageDataLayer);
} // namespace danknet

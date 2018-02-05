#ifndef DATA_H
#define DATA_H

#include <QImage>
#include <string.h>

using namespace std;

//------------------Data3d--------------------
template<typename Dtype>
class Data3d{
    int         w_,
                h_,
                c_;
    bool        nulldata_ = true;
    Dtype*      data_;
public:
    // constructor
    Data3d();
    Data3d(int w, int h, int c);

    // copy constructor
    Data3d(const Data3d &data);
    // assignment operator
    Data3d& operator = (const Data3d& data);
    Dtype* data(int x, int y, int c);

    ~Data3d();

    inline bool         isNull() const      { return nulldata_; }
    inline int          width()             { return w_; }
    inline int          height()            { return h_; }
    inline int          depth()             { return c_; }
};

//------------------Data3d--------------------
//--------------implementation----------------
template<typename Dtype>
Data3d<Dtype>::Data3d() {
    w_ = h_ = c_ = 0;
    nulldata_ = true;
}


template<typename Dtype>
Data3d<Dtype>::Data3d(int w, int h, int c) {
    w_          = w;
    h_          = h;
    c_          = c;
    data_       = new Dtype[c_ * w_ * h_];
    nulldata_   = false;

}


//TODO: different data types are not supported
// copy constructor
template<typename Dtype>
Data3d<Dtype>::Data3d(const Data3d<Dtype> &data) {
    w_          = data.w_;
    h_          = data.h_;
    c_          = data.c_;
    data_       = new Dtype[c_ * w_ * h_];
    memcpy(data_, data.data_, c_ * w_ * h_);
    nulldata_   = data.nulldata_;
}

//TODO: different data types are not supported
// assignment operator
template<typename Dtype>
Data3d<Dtype>& Data3d<Dtype>::operator = (const Data3d<Dtype>& data) {
    w_          = data.w_;
    h_          = data.h_;
    c_          = data.c_;
    data_       = data.data_;
    nulldata_   = data.nulldata_;
}

template<typename Dtype>
Dtype* Data3d<Dtype>::data(int x, int y, int c) {
    // It is assumed that the data is stored as follows:
    //
    //     <- w_ ->
    // ^  abc abc abc
    // h_ abc abc abc
    // v  abc abc abc
    //
    //   abc - c_ elements
    return &(data_[(y * w_ + x) * c_ + c]);
}

template<typename Dtype>
Data3d<Dtype>::~Data3d() {
    nulldata_   = true;
    delete data_;
}

//------------------Shape---------------------
class Shape{
    int     width_,
            height_,
            depth_,
            batch_;

    int     dims_;
public:
    Shape() {
        width_  = 1;
        height_ = 1;
        depth_  = 1;
        batch_  = 1;
        dims_   = 1;
    }

    Shape(int width) {
        width_  = width;
        height_ = 1;
        depth_  = 1;
        batch_  = 1;
        dims_   = 1;
    }
    Shape(int width, int height) {
        width_  = width;
        height_ = height;
        depth_  = 1;
        batch_  = 1;
        dims_   = 2;
    }
    Shape(int width, int height, int depth) {
        width_  = width;
        height_ = height;
        depth_  = depth;
        batch_  = 1;
        dims_   = 3;
    }
    Shape(int width, int height, int depth, int batch) {
        width_  = width;
        height_ = height;
        depth_  = depth;
        batch_  = batch;
        dims_   = 4;
    }
    inline int width()  { return width_; }
    inline int height() { return height_; }
    inline int depth()  { return depth_; }
    inline int batch()  { return batch_; }
    inline int dims()   { return dims_; }
};



//------------------Blob----------------------
//TODO: create convenient implementation
template<typename Dtype>
class Blob{
    string                  name_;
    Shape                   shape_;
    vector<Data3d<Dtype>*>  data_;

public:
    Blob(string name, Shape shape);

    ~Blob();
    inline string           name()          { return name_; }
    inline Shape            shape()         { return shape_; }
    inline int              batch_size()    { return shape_.batch(); }

    inline Data3d<Dtype>*   Data(int i)     { return data_[i]; }

    inline Dtype*           data(int i)     { return data_[i]->data(0,0,0); }
    inline Dtype*           data(int i, int x, int y, int c) {
//        if(i < shape.batch() && x < shape.width() && y < shape.height() && c < shape.depth())
            return data_[i]->data(x, y, c);
    }
};


template<typename Dtype>
Blob<Dtype>::Blob(string name, Shape shape) {
    name_ = name;
    shape_ = shape;
    for(int batch = 0; batch < shape_.batch(); batch++) {
        data_.push_back(new Data3d<Dtype>(shape_.width(), shape_.height(), shape.depth()));
    }
}

template<typename Dtype>
Blob<Dtype>::~Blob() {
    data_.clear();
}




#endif // DATA_H

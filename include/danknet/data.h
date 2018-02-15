#ifndef DATA_H
#define DATA_H

#include <QImage>
#include <string.h>

#include "data2d.h"

using namespace std;

//------------------Shape---------------------
class Shape{
    int     width_,
            height_,
            depth_,
            batch_;

    int     dims_;
public:

    Shape(int width = 1, int height = 1, int depth = 1, int batch = 1) {
        width_  = width;
        height_ = height;
        depth_  = depth;
        batch_  = batch;
        if(batch_ == 1) {
            if(depth_ == 1) {
                if(height_ == 1) {
                    dims_ = 1;
                } else {
                    dims_ = 2;
                }
            } else {
                dims_   = 3;
            }
        } else {
            dims_   = 4;
        }
    }
    inline int width()  { return width_; }
    inline int height() { return height_; }
    inline int depth()  { return depth_; }
    inline int batch()  { return batch_; }

    inline int dims()   { return dims_; }

    inline bool operator==(const Shape& rhs){ return  width_ == rhs.width_ &&
                                                                        height_ == rhs.height_ &&
                                                                        depth_ == rhs.depth_ &&
                                                                        batch_ == rhs.batch_ &&
                                                                        dims_ == rhs.dims_;
                                                              }
    inline bool operator!=(const Shape& rhs){ return !(*this == rhs); }
};


//------------------Data3d--------------------
template<typename Dtype>
class Data3d{
    Shape       shape_;
    Dtype*      data_;
    Dtype       zero_val = 0;
public:
    // constructor
    Data3d();
    Data3d(int w, int h, int c);
    Data3d(Shape shape);

    // copy constructor
    Data3d(const Data3d& data);

    // assignment operator
    // Data3d = Data3d
    Data3d& operator = (const Data3d& data);

    template<typename Btype>
    Data3d& operator = (const Data3d<Btype>& data);

    //Data3d = Data2d
    template<typename Btype>
    Data3d& operator = (const Data2d<Btype>& data);

    //Data3d = QImage
    Data3d& operator = (const QImage& data);

    Dtype* data(int x, int y, int c);
    void setToZero();
    inline Dtype* data()                    { return data_; }

    ~Data3d();

    inline Shape        shape()             { return shape_; }

    inline int          width()             { return shape_.width(); }
    inline int          height()            { return shape_.height(); }
    inline int          depth()             { return shape_.depth(); }
};

//------------------Data3d--------------------
//--------------implementation----------------
//----------------constructor-----------------
template<typename Dtype>
Data3d<Dtype>::Data3d() {
    shape_ = Shape();
}


template<typename Dtype>
Data3d<Dtype>::Data3d(Shape shape) {
    shape_      = Shape(shape.width(), shape.height(), shape.depth());
    data_       = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
}


template<typename Dtype>
Data3d<Dtype>::Data3d(int w, int h, int c) {
    shape_      = Shape(w, h, c);
    data_       = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
}


//--------------copy constructor--------------
//TODO: different data types are not supported
template<typename Dtype>
Data3d<Dtype>::Data3d(const Data3d<Dtype> &data) {
    shape_      = data.shape_;
    data_       = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
    //TODO: check sizeof(Dtype)
    memcpy(data_, data.data_, shape_.width() * shape_.height() * shape_.depth() * sizeof(Dtype));
}


//------------assignment operator-------------
//--------------Data3d = Data3d---------------
template<typename Dtype>
Data3d<Dtype>& Data3d<Dtype>::operator = (const Data3d<Dtype>& data) {
    Shape data_shape = Shape(data.width(), data.height(), data.depth());
    if(shape_ != data_shape) {
        shape_ = data_shape;
        delete data_;
        data_ = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
    }
    shape_      = data.shape_;
    *data_      = *data.data_;
}


template<typename Dtype>
template<typename Btype>
Data3d<Dtype>& Data3d<Dtype>::operator = (const Data3d<Btype>& data) {
    Shape data_shape = Shape(data.width(), data.height(), data.depth());
    if(shape_ != data_shape) {
        shape_ = data_shape;
        delete data_;
        data_ = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
    }
    int w = shape_.width();
    int h = shape_.height();
    int d = shape_.depth();

    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            for(int c = 0; c < d; c++) {
                *data(x, y, c) = (Dtype)(*data(x, y, c));
            }
        }
    }
}


//--------------Data3d = QImage---------------
template<typename Dtype>
Data3d<Dtype>& Data3d<Dtype>::operator = (const QImage& data) {
    Shape data_shape = Shape(data.width(), data.height(), data.depth());
    if(shape_ != data_shape) {
        shape_ = data_shape;
        delete data_;
        data_ = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
    }
    int w = shape_.width();
    int h = shape_.height();
    int d = shape_.depth();
    for(int y = 0; y < h; y++) {
        const uint8_t* data_ptr = data.scanLine(y);
        for(int x = 0; x < w; x++) {
            for(int c = 0; c < d; c++) {
                data_[c * w * h + (y * w + x)] = (Dtype)(data_ptr[x * d + c]);
            }
        }
    }
}


//--------------Data3d = Data2d---------------
template<typename Dtype>
template<typename Btype>
Data3d<Dtype>& Data3d<Dtype>::operator = (const Data2d<Btype>& data) {
    Shape data_shape = Shape(data.width(), data.height(), data.depth());
    if(shape_ != data_shape) {
        shape_ = data_shape;
        //TODO: check memory leak
        data_ = new Dtype[shape_.width() * shape_.height() * shape_.depth()];
    }
    int w = shape_.width();
    int h = shape_.height();
    int d = shape_.depth();

    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            for(int c = 0; c < d; c++) {
                *data(x, y, c) = (Dtype)(*data(x, y, c));
            }
        }
    }
}


template<typename Dtype>
Dtype* Data3d<Dtype>::data(int x, int y, int c) {
    // It is assumed that the data is stored as follows:
    //
    //     <- w_ ->
    // ^  a   a   a
    // h_ a   a   a
    // v  a   a   a
    //     <- w_ ->
    // ^  b   b   b
    // h_ b   b   b
    // v  b   b   b
    //     <- w_ ->
    // ^  c   c   c
    // h_ c   c   c
    // v  c   c   c
    //
    //   abc - c_ elements
    if(x < 0 || x >= shape_.width() ||
       y < 0 || y >= shape_.height() ||
       c < 0 || c >= shape_.depth()) {
        return &zero_val;
    } else {
        return &(data_[c * shape_.width() * shape_.height() + (y * shape_.width() + x)]);
//        return &(data_[(y * shape_.width() + x) * shape_.depth() + c]);
    }
}

template<typename Dtype>
void Data3d<Dtype>::setToZero() {
    memset(data_, 0, shape_.width() * shape_.height() * shape_.depth() * sizeof(Dtype));
}

template<typename Dtype>
Data3d<Dtype>::~Data3d() {
    delete data_;
}


//------------------Blob----------------------
//TODO: create convenient implementation
template<typename Dtype>
class Blob{
    string                  name_;
    Shape                   shape_;
    Data3d<Dtype>**         data_;

public:
    Blob(string name, Shape shape);

    ~Blob();
    inline string           name()          { return name_; }
    inline Shape            shape()         { return shape_; }

    inline int              width()         { return shape_.width(); }
    inline int              height()        { return shape_.height(); }
    inline int              depth()         { return shape_.depth(); }
    inline int              batch_size()    { return shape_.batch(); }

    inline Data3d<Dtype>*   Data(int i)     { return data_[i]; }

    inline Dtype*           data(int i)     { return data_[i]->data(); }
    inline Dtype*           data(int i, int x, int y, int c) {
            return data_[i]->data(x, y, c);
    }
    void setToZero();
};

template<typename Dtype>
void Blob<Dtype>::setToZero() {
    for(int batch = 0; batch < shape_.batch(); batch++) {
        data_[batch]->setToZero();
    }
}

template<typename Dtype>
Blob<Dtype>::Blob(string name, Shape shape) {
    name_       = name;
    shape_      = shape;

    data_       = new Data3d<Dtype>*[shape_.batch()];
    for(int batch = 0; batch < shape_.batch(); batch++) {
        data_[batch] = new Data3d<Dtype>(shape_);
    }
}

template<typename Dtype>
Blob<Dtype>::~Blob() {
    for(int batch = 0; batch < shape_.batch(); batch++) {
        delete data_[batch];
    }
    delete data_;
}




#endif // DATA_H

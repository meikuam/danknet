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
    string      name_;
public:
    // constructor
    Data3d();
    Data3d(int w, int h, int c, string name = "");
    Data3d(QImage* img, string name = "");

    // copy constructor
    Data3d(const Data3d &data);
    // assignment operator
    Data3d& operator = (const Data3d& data);
    Dtype* data(int x, int y, int c);

    ~Data3d();

    inline string       name() {
        return name_;
    }

    inline bool         isNull() const {
        return nulldata_;
    }

    inline int          width() {
        return w_;
    }

    inline int          height() {
        return h_;
    }

    inline int          depth() {
        return c_;
    }
};

//------------------Data3d--------------------
//--------------implementation----------------
template<typename Dtype>
Data3d<Dtype>::Data3d() {
    w_ = h_ = c_ = 0;
    nulldata_ = true;
}


template<typename Dtype>
Data3d<Dtype>::Data3d(int w, int h, int c, string name) {
    w_          = w;
    h_          = h;
    c_          = c;
    data_       = new Dtype[c_ * w_ * h_];
    nulldata_   = false;
    name_       = name;

}

//TODO: different data types are not supported. Need to check sizeof(QImage data)
template<typename Dtype>
Data3d<Dtype>::Data3d(QImage* img, string name) {
    if(img->isNull())
        return;
    w_          = img->width();
    h_          = img->height();
    c_          = img->depth() / (sizeof(Dtype) * 8);
    data_       = new Dtype[c_ * w_ * h_];
    for(int y = 0; y < h_; y++) {
        memcpy(&(data_[y * w_ * c_]), img->scanLine(y), w_ * c_);
    }
    nulldata_   = false;
    name_       = name;
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
    name_       = data.name_;
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
    name_       = data.name_;
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

template<typename Dtype>
class Blob{
    vector<Data3d<Dtype>*> data_;
    string name_;
public:
    Blob(string name, vector<Data3d<Dtype>*>& data_);
    ~Blob();
    inline string name() {
        return name_;
    }
    inline vector<Data3d<Dtype>*>* data() {
        return &data_;
    }
};

template<typename Dtype>
Blob<Dtype>::Blob(string name, vector<Data3d<Dtype>*>& data_) {
    for(vector<Data3d<Dtype>*>::iterator it = data.begin(); it!= data.end(); it++) {
        data_.push_back(*it);
    }
}

template<typename Dtype>
Blob<Dtype>::~Blob() {
    data.clear();
}

class Shape{
    int     dim1_,
            dim2_,
            dim3_,
            dim4_,
            dim5_;
    int     dims_;
public:
    Shape(int dim1) {
        dim1_   = dim1;
        dims    = 1;
    }
    Shape(int dim1, int dim2) {
        dim1_   = dim1;
        dim2_   = dim2;
        dims_   = 2;
    }
    Shape(int dim1, int dim2, int dim3) {
        dim1_   = dim1;
        dim2_   = dim2;
        dim3_   = dim3;
        dims_   = 3;
    }
    Shape(int dim1, int dim2, int dim3, int dim4) {
        dim1_   = dim1;
        dim2_   = dim2;
        dim3_   = dim3;
        dim4_   = dim4;
        dims_   = 4;
    }
    Shape(int dim1, int dim2, int dim3, int dim4, int dim5) {
        dim1_   = dim1;
        dim2_   = dim2;
        dim3_   = dim3;
        dim4_   = dim4;
        dim5_   = dim5;
        dims_   = 5;
    }
    inline int dims(){
        return dims_;
    }
    inline int dim1(){
        return dim1_;
    }
    inline int dim2(){
        return dim2_;
    }
    inline int dim3(){
        return dim3_;
    }
    inline int dim4(){
        return dim4_;
    }
    inline int dim5(){
        return dim5_;
    }

};


#endif // DATA_H

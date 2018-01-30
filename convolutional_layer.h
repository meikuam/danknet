#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

template <typename Btype, typename Ttype>
class ConvolutionalLayer : public Layer<Btype, Ttype> {
 public:
  explicit ConvolutionalLayer()
      : Layer<Ftype, Btype>() {}

    virtual void Forward(vector<Data2d*>& bottom, vector<Data2d*>& top);
    virtual void Backward(vector<Data2d*>& top, vector<Data2d*>& bottom);

    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int channels_;
    int height_, width_;
};

#endif // CONVOLUTIONAL_LAYER_H

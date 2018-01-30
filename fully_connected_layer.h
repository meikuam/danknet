#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

template <typename Btype, typename Ttype>
class FullyConectedLayer : public Layer<Btype, Ttype> {
 public:
  explicit FullyConectedLayer()
      : Layer<Ftype, Btype>() {}

    virtual void Forward(vector<Data2d*>& bottom, vector<Data2d*>& top);
    virtual void Backward(vector<Data2d*>& top, vector<Data2d*>& bottom);

    virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);

private:

};


#endif // FULLY_CONNECTED_LAYER_H

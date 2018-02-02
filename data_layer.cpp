#include "data_layer.h"


namespace danknet {

//template<typename Dtype>
//void
//DataLayer<Dtype>::LayerSetUp(const vector<Data2d<Dtype>*>& bottom,
//                             const vector<Data2d<Dtype>*>& top) {

//}


template<typename Dtype>
void
DataLayer<Dtype>::Forward(const vector<Data2d<Dtype>*>& bottom,
                          const vector<Data2d<Dtype>*>& top) {
    vector<Data2d<Dtype>*>::iterator top_it;

    for(top_it = top.begin(); top_it!= top.end(); top_it++) {
        for(vector<Data2d<Dtype>*>::iterator batch_it = batch_.begin(); batch_it!= batch_.end(); batch_it++) {
            if(batch_it.name()==top_it.name()) {
                *top_it = batch_it;
            }
        }
    }
}

//template<typename Dtype>
//void
//DataLayer<Dtype>::Backward(const vector<Data2d<Dtype>*>& bottom,
//                           const vector<Data2d<Dtype>*>& top) {

//}

template<typename Dtype>
void
DataLayer<Dtype>::load_batch(vector<Data2d<Dtype>*>& batch) {
    batch_.clear();
    for(vector<Data2d<Dtype>*>::iterator it = batch.begin(); it!= batch.end(); it++) {
        batch_.push_back(it);
    }
}

} // namespace danknet

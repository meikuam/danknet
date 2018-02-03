#include "data_layer.h"


namespace danknet {


//template<typename Dtype>
//void
//DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

//}


template<typename Dtype>
void
DataLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//    vector<Data3d<Dtype>*>::iterator top_it;

//    for(top_it = top.begin(); top_it!= top.end(); top_it++) {
//        for(vector<Data3d<Dtype>*>::iterator batch_it = batch_.begin(); batch_it!= batch_.end(); batch_it++) {
//            if(batch_it.name()==top_it.name()) {
//                *top_it = batch_it;
//            }
//        }
//    }
}

//template<typename Dtype>
//void
//DataLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {

//}

template<typename Dtype>
void
DataLayer<Dtype>::load_batch(vector<Blob<Dtype>*>& batch) {
//    batch_.clear();
//    for(vector<Data3d<Dtype>*>::iterator it = batch.begin(); it!= batch.end(); it++) {
//        batch_.push_back(it);
//    }
}

} // namespace danknet

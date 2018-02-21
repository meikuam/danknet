#include <iostream>
#include <unistd.h>
#include <fstream>

#include "danknet.h"

using namespace std;
using namespace danknet;

bool fexists(const char *filename) {
  std::ifstream ifile(filename);
  return (bool)ifile;
}


int main(int argc, char *argv[])
{
    cout<<"start"<<endl;
    QString input_train_path = "/home/hotoru/datasets/dollar/dollar_diff_train.txt";
    QString output_test_path = "dollar_test.txt";

//    int input_depth = 5;
//    int output_depth = 1;
//    int batch_size = 10;
//    //data Blobs
//    vector<Blob<double>*>       input_data0,
//                                fc0_top,
//                                fc1_top,
//                                fc2_top,
//                                fc3_top,
//                                fc4_top,
//                                loss_top;
//    //create Net
//    cout<<"create Net"<<endl;
//    Net<double> pred_net;
//    pred_net.AddLayer(new DataLayer<double>(input_depth, output_depth, batch_size, input_train_path.toStdString(), input_test_path.toStdString(), "input", input_data0));
//    pred_net.AddLayer(new FullyConnectedLayer<double>(1, leakyReLU, "fc0", input_data0, fc0_top));
////    pred_net.AddLayer(new FullyConnectedLayer<double>(1, leakyReLU, "fc1", fc0_top, fc1_top));

//    vector<Blob<double>*> &net_top = fc0_top;
//    net_top.push_back(input_data0[1]);
//    pred_net.AddLayer(new LossLayer<double>("loss", net_top, loss_top));



//    int test_iters = 100;
//    int train_iters = 50000;
//    int iters = 100000;
//    int step_size = 100000;

//    Data3d<double>* data;

//    cout<<"---------------------Forward----------------------"<<endl;
//    for(int k = 0; k < iters; k++) {
//        cout<<"lr_rate: "<<pred_net.lr_rate()<<endl;
//        pred_net.phase(TRAIN);
//        cout<<"phase_: TRAIN"<<endl;

//        double train_loss = 0;
//        for(int i = 0; i < train_iters; i++) {
//            pred_net.Forward();

////            for(int b = 0; b < batch_size; b++) {
////                train_loss += *loss_top[0]->data(b, 0, 0, 0);
////            }
////            train_loss /= batch_size;
////            cout<<k * (train_iters + test_iters) + i<<" avg loss: "<<train_loss<<" net top: "<<*net_top[0]->data(0, 0, 0, 0)<<" label: "<<*input_data0[1]->data(0, 0, 0, 0)<<endl;
////            train_loss = 0;
//            pred_net.Backward();
//        }

//        pred_net.phase(TEST);
//        cout<<"phase_: TEST"<<endl;
//        double test_loss = 0;
//        double loss_ = 0;
//        for(int i = 0; i < test_iters; i++) {
//            pred_net.Forward();
//            test_loss = 0;
//            for(int b = 0; b < batch_size; b++) {
//                test_loss += *loss_top[0]->data(b, 0, 0, 0);
//            }
//            test_loss /= batch_size;
//            cout<<k * (test_iters + train_iters) + train_iters + i<<" avg loss: "<<test_loss<<" net top: "<<*net_top[0]->data(0, 0, 0, 0)<<" label: "<<*input_data0[1]->data(0, 0, 0, 0)<<endl;
//            loss_ += test_loss;
//        }
//        loss_ /= test_iters;
//        cout<<k * (test_iters + train_iters) + train_iters + test_iters<<" avg loss: "<<loss_<<endl;
////        pred_net.WeightsToHDF5("weights/pred_net" + to_string(k) + ".hdf5");
//    }

    return 0;
}

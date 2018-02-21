#include <iostream>
#include <unistd.h>
#include <fstream>

#include <QImage>

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
    //--------------------------------------------
    Net<double> net;


    vector<Blob<double>*>    image_data0,
                            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9,
                            fc2,
                            softmax,
                            loss;

    if(argc < 3)
        return 0;

    cout<<"Add image_data layer"<<endl;
    string train_path(argv[1]);
    string test_path(argv[2]);
    cout<<"train path: "<<train_path<<endl<<"test path: "<<test_path<<endl;


    cout<<"Add ImageDataLayer"<<endl;

    net.AddLayer(new ImageDataLayer<double>(23, 23, 3, 100, 2,
                                           train_path,
                                           test_path,
                                           "input", image_data0));
    net.AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 32, 3, 3, 0, 0, ReLU, "conv0", image_data0, conv0));
    net.AddLayer(new ConvolutionalLayer<double>(3, 3, 32, 256, 2, 2, 0, 0, ReLU, "conv1", conv0, conv1));
    net.AddLayer(new FullyConnectedLayer<double>(2, ReLU, "fc2", conv1, fc2));
    net.AddLayer(new SoftmaxLayer<double>("softmax",fc2, softmax));
    fc2.push_back(image_data0[1]);
    net.AddLayer(new LossLayer<double>("loss", fc2, loss));


    //out_dim = (input_dim + 2 * pad - kernel_size) / stride + 1;

//    net.AddLayer(new ImageDataLayer<double>(5, 5, 3, 100, 2,
//                                           train_path,
//                                           test_path,
//                                           "input", image_data0));
//    net.AddLayer(new ConvolutionalLayer<double>(3, 3, 3, 16, 1, 1, 0, 0, ReLU, "fc1", image_data0, conv0));
//    net.AddLayer(new FullyConnectedLayer<double>(300, ReLU, "fc2", conv0, conv1));
//    net.AddLayer(new FullyConnectedLayer<double>(2, ReLU, "fc3", conv1, fc2));
//    net.AddLayer(new SoftmaxLayer<double>("softmax",fc2, softmax));
//    fc2.push_back(image_data0[1]);
//    net.AddLayer(new LossLayer<double>("loss", fc2, loss));


    cout<<"Forward"<<endl;
    int test_iters = 100;
    int train_iters = 500;
    int iters = 100000;
    int step_size = 1000;

    Data3d<double>* data0;

    double lr_rate = 0.01;
    double weight_decay = 0.00005;
    double momentum = 0.9;
    double gamma = 0.1;

    net.lr_rate(lr_rate);
    net.weight_decay(weight_decay);
    net.momentum(momentum);
    net.gamma(gamma);
    net.step_size(step_size);

//    net.WeightsFromHDF5("net110.hdf5");
//    net.WeightsToHDF5("weights/net.hdf5");

    double fc_data[2];
    for(int k = 0; k < iters; k++) {
        cout<<"lr_rate: "<<net.lr_rate()<<endl;
        net.phase(TRAIN);
        cout<<"phase_: TRAIN"<<endl;
        double train_accuracy = 0;
        double train_loss = 0;
        for(int i = 0; i < train_iters; i++) {
            net.Forward();

            for(int b = 0; b < softmax[0]-> batch_size(); b++) {
                if(((*image_data0[1]->data(b, 0, 0, 0) == 0) ? 1 : 0) == *softmax[0]->data(b, 0, 0, 0)) {
                    train_accuracy++;
                }
                train_loss += *loss[0]->data(b, 0, 0, 0);
                fc_data[0] += *fc2[0]->data(b, 0, 0, 0);
                fc_data[1] += *fc2[0]->data(b, 0, 0, 1);

            }
            fc_data[0] /= 1.0 * softmax[0]-> batch_size();
            fc_data[1] /= 1.0 * softmax[0]-> batch_size();
            train_accuracy /= softmax[0]-> batch_size();
            train_loss /= softmax[0]-> batch_size();
            cout<<k * (train_iters+test_iters) + i<<" avg accuracy: "<<train_accuracy<<" avg loss: "<<train_loss<<" fc2: "<<fc_data[0]<<" "<<fc_data[1]<<endl;
            train_accuracy = 0;
            train_loss = 0;

            fc_data[0] = fc_data[0] = 0;
//            for(int b = 0; b < softmax[0]-> batch_size(); b++) {
//                cout<<k * (train_iters+test_iters) + i<<" sm: "<<*softmax[0]->data(b, 0, 0, 0)<<" l: "<<((*image_data0[1]->data(b, 0, 0, 0) == 0) ? 1 : 0)<<" loss: "<<*loss[0]->data(b, 0, 0, 0)<<" fc2: "<<*fc2[0]->data(b, 0, 0, 0)<<" "<<*fc2[0]->data(b, 0, 0, 1)<<endl;
//            }


//            if(i % 50 == 0) {
//            cout<<"fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;
//            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;
//            cout<<"sm: "<<*softmax[0]->data(0, 0, 0, 0)<<" l: "<<((*image_data0[1]->data(0, 0, 0, 0) == 0) ? 1 : 0)<<endl;
//            }
            net.Backward();
        }

        net.phase(TEST);
        cout<<"phase_: TEST"<<endl;
        double test_accuracy = 0;
        double test_loss = 0;

        double loss_ = 0;
        int accuracy = 0;
        for(int i = 0; i < test_iters; i++) {
            net.Forward();

            for(int b = 0; b < softmax[0]-> batch_size(); b++) {
                if(((*image_data0[1]->data(b, 0, 0, 0) == 0) ? 1 : 0) == *softmax[0]->data(b, 0, 0, 0)) {
                    test_accuracy++;
                }
                test_loss += *loss[0]->data(b, 0, 0, 0);
                fc_data[0] += *fc2[0]->data(b, 0, 0, 0);
                fc_data[1] += *fc2[0]->data(b, 0, 0, 1);

            }
            fc_data[0] /= 1.0 * softmax[0]-> batch_size();
            fc_data[1] /= 1.0 * softmax[0]-> batch_size();
            test_accuracy /= softmax[0]-> batch_size();
            test_loss /= softmax[0]-> batch_size();
            cout<<k * (test_iters+test_iters) + i<<" avg accuracy: "<<test_accuracy<<" avg loss: "<<test_loss<<" fc2: "<<fc_data[0]<<" "<<fc_data[1]<<endl;
            test_accuracy = 0;
            test_loss = 0;

            fc_data[0] = fc_data[0] = 0;

//            cout<<k * (train_iters+test_iters) + i<<" sm: "<<*softmax[0]->data(0, 0, 0, 0)<<" l: "<<((*image_data0[1]->data(0, 0, 0, 0) == 0) ? 1 : 0)<<" loss: "<<*loss[0]->data(0, 0, 0, 0)<<"\tfc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;

//            cout<<k * (train_iters+test_iters) + train_iters + i<<" loss: "<<*loss[0]->data(0, 0, 0, 0)<<endl;
//            cout<<"fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;
////            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;

//            cout<<"sm: "<<*softmax[0]->data(0, 0, 0, 0)<<" l: "<<l<<endl;


            for(int b = 0; b < loss[0]->batch_size(); b++) {
                if(((*image_data0[1]->data(b, 0, 0, 0) == 0 )? 1 : 0) == *softmax[0]->data(b, 0, 0, 0)) {
                    accuracy++;
                }
                loss_ += *loss[0]->data(b, 0, 0, 0);
            }
        }
        cout<<"accuracy: "<< (accuracy * 1.0) / (test_iters * 1.0 * loss[0]->batch_size()) <<endl;
        loss_ /= test_iters * 1.0 * loss[0]->batch_size();
        cout<<"test loss: "<<loss_<<endl;
        net.WeightsToHDF5("weights/net" + to_string(k) + ".hdf5");

//        if(k * (train_iters+test_iters)) {
//            lr_rate *= lr_rate * gamma;
//        }
    }



    //            cout<<image_data0[0]->name()<<endl;
    //            for(int b = 0; b < image_data0[0]->batch_size(); b++) {
    //                data0 = image_data0[0]->Data(b);
    //                for(int c = 0; c < data0->depth(); c++) {
    //                    for(int y = 0; y < data0->height(); y++) {
    //                        for(int x = 0; x < data0->width(); x++) {
    //                            cout<<*data0->data(x, y, c)<<" ";
    //                        }
    //                        cout<<endl;
    //                    }
    //                    cout<<endl<<endl;
    //                }
    //            }

    //            cout<<conv0[0]->name()<<endl;
    //            for(int b = 0; b < conv0[0]->batch_size(); b++) {
    //                data0 = conv0[0]->Data(b);
    //                for(int c = 0; c < data0->depth(); c++) {
    //                    for(int y = 0; y < data0->height(); y++) {
    //                        for(int x = 0; x < data0->width(); x++) {
    //                            cout<<*data0->data(x, y, c)<<" ";
    //                        }
    //                        cout<<endl;
    //                    }
    //                    cout<<endl<<endl;
    //                }
    //            }

    //            cout<<conv1[0]->name()<<endl;
    //            for(int b = 0; b < conv1[0]->batch_size(); b++) {
    //                data0 = conv1[0]->Data(b);
    //                for(int c = 0; c < data0->depth(); c++) {
    //                    for(int y = 0; y < data0->height(); y++) {
    //                        for(int x = 0; x < data0->width(); x++) {
    //                            cout<<*data0->data(x, y, c)<<" ";
    //                        }
    //                        cout<<endl;
    //                    }
    //                    cout<<endl<<endl;
    //                }
    //            }

    //            cout<<fc2[0]->name()<<endl;
    //            for(int b = 0; b < fc2[0]->batch_size(); b++) {
    //                data0 = fc2[0]->Data(b);
    //                for(int c = 0; c < data0->depth(); c++) {
    //                    for(int y = 0; y < data0->height(); y++) {
    //                        for(int x = 0; x < data0->width(); x++) {
    //                            cout<<*data0->data(x, y, c)<<" ";
    //                        }
    //                        cout<<endl;
    //                    }
    //                    cout<<endl<<endl;
    //                }
    //            }

	return 0;
}


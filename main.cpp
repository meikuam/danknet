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
                            fc2,
                            loss;
    double lr_rate = 0.0001;
    double weight_decay = 0.00005;

    if(argc < 3)
        return 0;

    cout<<"Add image_data layer"<<endl;
    string train_path(argv[1]);
    string test_path(argv[2]);
    cout<<"train path: "<<train_path<<endl<<"test path: "<<test_path<<endl;

    net.AddLayer(new ImageDataLayer<double>(23, 23, 3, 1, 2,
                                           train_path,
                                           test_path,
                                           "input", image_data0));
    net.AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 5, 3, 3, 0, 0, "conv0", image_data0, conv0));
    net.AddLayer(new ConvolutionalLayer<double>(3, 3, 5, 25, 2, 2, 0, 0, "conv1", conv0, conv1));
    net.AddLayer(new FullyConnectedLayer<double>(2, "fc2", conv1, fc2));
    fc2.push_back(image_data0[1]);
    net.AddLayer(new LossLayer<double>("loss", fc2, loss));

    cout<<"Forward"<<endl;
    int test_iters = 15;
    int train_iters = 200;
    int iters = 2;
    Data3d<double>* data0;

    net.WeightsToHDF5("net.hdf5");
    for(int k = 0; k < iters; k++) {
        net.lr_rate(lr_rate);
        net.weight_decay(weight_decay);
        net.phase(TRAIN);
        cout<<"phase_: TRAIN"<<endl;
        for(int i = 0; i < train_iters; i++) {
            net.Forward();

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


            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;
            cout<<"fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;
            cout<<"loss: "<<*loss[0]->data(0, 0, 0, 0)<<endl;
            net.Backward();
        }

        net.phase(TEST);
        cout<<"phase_: TEST"<<endl;
        double loss_ = 0;
        for(int i = 0; i < test_iters; i++) {
            net.Forward();
            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;
            cout<<"fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;
            cout<<"loss: "<<*loss[0]->data(0, 0, 0, 0)<<endl;
            loss_ += *loss[0]->data(0, 0, 0, 0);
        }
        loss_ /= test_iters;
        cout<<"test loss: "<<loss_<<endl;
        net.WeightsToHDF5("net" + to_string(iters) + ".hdf5");
    }





	return 0;
}

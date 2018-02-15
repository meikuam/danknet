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
    Net<float> net;


    vector<Blob<float>*>    image_data0,
                            conv0,
                            conv1,
                            ip2,
                            loss;
    float lr_rate = 0.04;
    float weight_decay = 0.01;

    if(argc < 3)
        return 0;

    cout<<"Add image_data layer"<<endl;
    string train_path(argv[1]);
    string test_path(argv[2]);
    cout<<"train path: "<<train_path<<endl<<"test path: "<<test_path<<endl;

    net.AddLayer(new ImageDataLayer<float>(23, 23, 3, 1, 2,
                                           train_path,
                                           test_path,
                                           "input", image_data0));
    net.AddLayer(new ConvolutionalLayer<float>(5, 5, 3, 32, 3, 3, 0, 0, "conv0", image_data0, conv0));
    net.AddLayer(new ConvolutionalLayer<float>(3, 3, 32, 256, 2, 2, 0, 0, "conv1", conv0, conv1));
    net.AddLayer(new FullyConnectedLayer<float>(2, "ip2", conv1, ip2));
    ip2.push_back(image_data0[1]);
    net.AddLayer(new LossLayer<float>("loss", ip2, loss));

    cout<<"Forward"<<endl;
    int test_iters = 1;
    int train_iters = 1;
    int iters = 1;
    Data3d<float>* data0;

    net.WeightsToHDF5("net.hdf5");
    for(int k = 0; k < iters; k++) {
        net.lr_rate(lr_rate);
        net.phase(TRAIN);
        for(int i = 0; i < train_iters; i++) {
            net.Forward();

            cout<<image_data0[0]->name()<<endl;
            for(int b = 0; b < image_data0[0]->batch_size(); b++) {
                data0 = image_data0[0]->Data(b);
                for(int c = 0; c < data0->depth(); c++) {
                    for(int y = 0; y < data0->height(); y++) {
                        for(int x = 0; x < data0->width(); x++) {
                            cout<<*data0->data(x, y, c)<<" ";
                        }
                        cout<<endl;
                    }
                    cout<<endl<<endl;
                }
            }

            cout<<conv0[0]->name()<<endl;
            for(int b = 0; b < conv0[0]->batch_size(); b++) {
                data0 = conv0[0]->Data(b);
                for(int c = 0; c < data0->depth(); c++) {
                    for(int y = 0; y < data0->height(); y++) {
                        for(int x = 0; x < data0->width(); x++) {
                            cout<<*data0->data(x, y, c)<<" ";
                        }
                        cout<<endl;
                    }
                    cout<<endl<<endl;
                }
            }

            cout<<conv1[0]->name()<<endl;
            for(int b = 0; b < conv1[0]->batch_size(); b++) {
                data0 = conv1[0]->Data(b);
                for(int c = 0; c < data0->depth(); c++) {
                    for(int y = 0; y < data0->height(); y++) {
                        for(int x = 0; x < data0->width(); x++) {
                            cout<<*data0->data(x, y, c)<<" ";
                        }
                        cout<<endl;
                    }
                    cout<<endl<<endl;
                }
            }

            cout<<ip2[0]->name()<<endl;
            for(int b = 0; b < ip2[0]->batch_size(); b++) {
                data0 = ip2[0]->Data(b);
                for(int c = 0; c < data0->depth(); c++) {
                    for(int y = 0; y < data0->height(); y++) {
                        for(int x = 0; x < data0->width(); x++) {
                            cout<<*data0->data(x, y, c)<<" ";
                        }
                        cout<<endl;
                    }
                    cout<<endl<<endl;
                }
            }


            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;
            cout<<"ip2: "<<*ip2[0]->data(0, 0, 0, 0)<<" "<<*ip2[0]->data(0, 0, 0, 1)<<endl;
            cout<<"loss: "<<*loss[0]->data(0, 0, 0, 0)<<" "<<*loss[0]->data(0, 0, 0, 1)<<endl;
            net.Backward();
        }

        net.phase(TEST);
        for(int i = 0; i < test_iters; i++) {
            net.Forward();
        }
        net.WeightsToHDF5("net" + to_string(iters) + ".hdf5");
    }





	return 0;
}

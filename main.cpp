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

    Net<double> LeNet;


    Blob<double> data("data", Shape(1, 1, 2));
    Blob<double> label("label", Shape(1, 1, 2));

    vector<Blob<double>*>    ip0_bottom,
                            ip0_top,
                            ip1_top,
                            ip2_top,
                            softmax_top;


    ip0_bottom.push_back(&data);
    cout<<"AddLayer"<<endl;

    double lr_rate = 0.01;

//    LeNet.AddLayer(new ConvolutionalLayer<double>(7, 7, 3, 3, 1, 1, 0, 0, lr_rate, "conv0", conv0_bottom, conv0_top));
//    LeNet.AddLayer(new PoolingLayer<double>(2, 2, 2, 2, 0, 0, "pool1", conv0_top, pool1_top));
//    LeNet.AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 2, 1, 1, 0, 0, lr_rate, "conv2", pool1_top, conv2_top));
//    LeNet.AddLayer(new PoolingLayer<double>(2, 2, 2, 2, 1, 1, "pool3", conv2_top, pool3_top));
//    LeNet.AddLayer(new ConvolutionalLayer<double>(5, 5, 2, 10, 1, 1, 0, 0, lr_rate, "conv4", pool3_top, conv4_top));
//    LeNet.AddLayer(new ConvolutionalLayer<double>(1, 1, 10, 3, 1, 1, 0, 0, lr_rate, "ip5", conv4_top, ip5_top));
//    ip5_top.push_back(&label);
//    LeNet.AddLayer(new LossLayer<double>(lr_rate, "loss", ip5_top, softmax_top));
    LeNet.AddLayer(new ConvolutionalLayer<double>(1, 1, 2, 10, 1, 1, 0, 0, lr_rate, "ip0", ip0_bottom, ip0_top));
    LeNet.AddLayer(new ConvolutionalLayer<double>(1, 1, 10, 2, 1, 1, 0, 0, lr_rate, "ip1", ip0_top, ip1_top));
    ip1_top.push_back(&label);
    LeNet.AddLayer(new LossLayer<double>(lr_rate, "loss", ip1_top, softmax_top));

//    cout<<"WeightsFromHDF5"<<endl;
//    LeNet.WeightsFromHDF5("lenet1900.hdf5");
    cout<<"---------------------Forward----------------------"<<endl;
    for(int i = 0, k = 0; i < 5000; i++, k++) {
        if(i%2 == 0 ){
            *label.Data(0)->data(0, 0, 0) = 0;
            *label.Data(0)->data(0, 0, 1) = 1;
            if(k%2 == 0) {
                *data.Data(0)->data(0, 0, 0) = 0;
                *data.Data(0)->data(0, 0, 1) = 1;
            } else {
                *data.Data(0)->data(0, 0, 0) = 1;
                *data.Data(0)->data(0, 0, 1) = 0;
            }

        } else {
            *label.Data(0)->data(0, 0, 0) = 1;
            *label.Data(0)->data(0, 0, 1) = 0;

            if(k%2 == 0) {
                *data.Data(0)->data(0, 0, 0) = 0;
                *data.Data(0)->data(0, 0, 1) = 0;
            } else {
                *data.Data(0)->data(0, 0, 0) = 1;
                *data.Data(0)->data(0, 0, 1) = 1;
            }
        }


    LeNet.Forward();
//    for(int i = 0; i < 10; i++) {
//        cout<<*ip5_top[0]->data(0,0,0,i)<< " ";
//    }
    cout<<i<<" "<<softmax_top[0]->name()<<" "<<*softmax_top[0]->data(0, 0, 0, 0)<<endl;
    LeNet.Backward();

    if(i%1000 == 0) {
        cout<<"WeightsToHDF5"<<endl;
        LeNet.WeightsToHDF5("lenet" + to_string(i) + ".hdf5");
    }
}
    cout<<"WeightsToHDF5"<<endl;
    LeNet.WeightsToHDF5("lenet2.hdf5");


    //-------
//    int             c;
//    bool            im_load        = false,
//            w_load         = false;
//    string          w_filename;
//    while ((c = getopt(argc, argv, "i:w:h:?")) != -1) {
//        switch(c)
//        {
//        case 'i':
//            cout<<"i: "<< optarg <<endl;
//            if (optind > argc) {
//                cout<<"Expected argument after options\n"<<endl;
//                exit(EXIT_FAILURE);
//            } else {
//                img = imread(optarg, CV_LOAD_IMAGE_COLOR);
//                if(img.data) {
//                    cout<<"Image: "<<optarg<<" opened"<<endl;
//                    im_load = true;
//                } else {
//                    cout<<"Could not open or find the image: "<<optarg<<endl;
//                }
//            }
//            break;
//        case 'w':
//            cout<<"w: "<<optarg<<endl;
//            if (optind > argc) {
//                cout<<"Expected argument after options\n"<<endl;
//                exit(EXIT_FAILURE);
//            } else {
//                if(fexists(optarg)) {
//                    if(H5File::isHdf5(optarg)) {
//                        cout<<"Weights file: "<<optarg<<" is exists"<<endl;
//                        w_filename = optarg;
//                        w_load = true;
//                    }
//                } else {
//                    cout<<"Could not find weights file: "<<optarg<<endl;
//                }
//            }
//            break;
//        case 'd':
//            //data.txt
//            cout<<"d: "<<optind<<" "<<optarg<<endl;
//            if (optind >= argc) {
//                cout<<"Expected argument after options\n"<<endl;
//                exit(EXIT_FAILURE);
//            } else {
//            }
//            break;
//        case '?':
//        case 'h':
//            cout<<"Usage:"<<endl;
//            cout<<"    By defauilt the image \"0.png\" and"<<endl;
//            cout<<"    \"cnn_weights.hdf5\" are loaded."<<endl<<endl;
//            cout<<"    example:"<<endl;
//            cout<<"    US_testnet -i image.png -w weights.hdf5"<<endl;
//            exit(EXIT_SUCCESS);
//        }
//    }
//    if(!im_load) {
//        cout<<"Trying to load \"0.png\""<<endl;
//        img = imread("0.png", CV_LOAD_IMAGE_COLOR);
//        if(img.data) {
//            cout<<"Image: \"0.png\" opened"<<endl;
//            im_load = true;
//        } else {
//            cout<<"Could not open or find the image: \"0.png\""<<endl;
//        }
//    }
//    if(!w_load) {
//        cout<<"Trying to find \"cnn_weights.hdf5\""<<endl;
//        if(fexists("cnn_weights.hdf5"))  {
//            if(H5File::isHdf5("cnn_weights.hdf5")) {
//                cout<<"Weights file: \"cnn_weights.hdf5\" is exists"<<endl;
//                w_filename = "cnn_weights.hdf5";
//                w_load = true;
//            }
//        } else {
//            cout<<"Could not find weights file:\
//                  \"cnn_weights.hdf5\""<<endl;
//            exit(EXIT_FAILURE);
//        }
//    }
       //---------------------------------------------------------------


	return 0;
}

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

    Net<float> edge_detector;


    Blob<float> data("data", Shape(3, 1, 1));
    Blob<float> label("label", Shape(1));

    Data3d<float>* data0 = data.data(0);
    for(int x = 0; x < data0->width(); x++) {
        for(int y = 0; y < data0->height(); y++) {
            for(int c = 0; c < data0->depth(); c++) {
                *data0->data(x, y, c) = 10.f;
            }
        }
    }
    for(int x = 0; x < data0->width(); x++) {
        for(int y = 0; y < data0->height(); y++) {
            for(int c = 0; c < data0->depth(); c++) {
                cout<<*data0->data(x, y, c)<<" ";
            }
        }
    }
    vector<Blob<float>*>    conv0_bottom,
                            conv0_top,
                            conv1_top,
                            conv2_top,
                            conv3_top,
                            conv4_top;

    conv0_bottom.push_back(&data);

    cout<<"AddLayer"<<endl;

    edge_detector.AddLayer(new Layer<float>("conv0", conv0_bottom, conv0_top));
    edge_detector.AddLayer(new Layer<float>("conv1", conv0_top, conv1_top));
    edge_detector.AddLayer(new Layer<float>("conv2", conv1_top, conv2_top));
    edge_detector.AddLayer(new Layer<float>("conv3", conv2_top, conv3_top));
    edge_detector.AddLayer(new Layer<float>("conv4", conv3_top, conv4_top));

    cout<<"Forward"<<endl;
    edge_detector.Forward();

    cout<<"cout"<<endl;
    edge_detector.Forward();
    cout<<conv0_bottom[0]->name()<<" "<<*conv0_bottom[0]->data(0)->data(0,0,0)<<endl;
    cout<<conv0_top[0]->name()<<" "<<*conv0_top[0]->data(0)->data(0,0,0)<<endl;
    cout<<conv1_top[0]->name()<<" "<<*conv1_top[0]->data(0)->data(0,0,0)<<endl;
    cout<<conv2_top[0]->name()<<" "<<*conv2_top[0]->data(0)->data(0,0,0)<<endl;
    cout<<conv3_top[0]->name()<<" "<<*conv3_top[0]->data(0)->data(0,0,0)<<endl;
    cout<<conv4_top[0]->name()<<" "<<*conv4_top[0]->data(0)->data(0,0,0)<<endl;

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

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
    Net<float> my_net();

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

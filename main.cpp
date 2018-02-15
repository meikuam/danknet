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
                            ip3,
                            ip4,
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
    net.Forward();


	return 0;
}

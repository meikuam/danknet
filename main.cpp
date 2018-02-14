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

    Net<double> net;


    vector<Blob<float>*>    image_data0,
                            conv0,
                            conv1,
                            ip2,
                            ip3,
                            ip4,
                            softmax_top;
    cout<<"Add image_data layer"<<endl;
    float lr_rate = 0.04;
    float weight_decay = 0.01;

    net.AddLayer(new ImageDataLayer(23, 23, 3, 1, 2,
                                    "/media/hp/647ad5df-5eef-4739-b3f3-f267d3fdcaf8/datasets/edge_dataset/sub_train.txt",
                                    "/media/hp/647ad5df-5eef-4739-b3f3-f267d3fdcaf8/datasets/edge_dataset/sub_test.txt",
                                    "input", image_data0));
    net.AddLayer(new ConvolutionalLayer(5, 5, 3, 32, 3, 3, 0, 0, "conv0", image_data0, conv0));
    net.AddLayer(new ConvolutionalLayer(3, 3, 32, 256, 2, 2, 0, 0, "conv1", conv0, conv1));
    net.AddLayer(new FullyConnectedLayer(2, "ip2", conv1, ip2));


	return 0;
}

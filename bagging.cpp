#include <iostream>
#include <unistd.h>
#include <fstream>

#include <QImage>
#include <QString>
#include <omp.h>

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
    if(argc < 2)
        return 0;

    int insts = 8; //net instances

    QString input_image_path(argv[1]);

    cout<<"input image path: "<<input_image_path.toStdString()<<endl;

    //create vector of nets
    vector<Net<double>> net(insts);
    //create vector of data
    vector<vector<Blob<double>*>> image_data0(insts), conv1(insts),
                                  pool2(insts), conv3(insts),
                                  pool4(insts), fc5(insts),
                                  fc6(insts);


    // open image
    Blob<double>* image_blob = new Blob<double>("image", Shape(64, 64, 3, 1));
    QImage img(input_image_path);
    *(image_blob->Data(0)) = img.convertToFormat(QImage::Format_RGB888).scaled(64, 64);

    Shape sh = image_blob->shape();
    int num = sh.width() * sh.height() * sh.depth();
    double* dat = image_blob->data(0);
    for(int i = 0; i < num; i++) {
        dat[i] = (dat[i] / 255.0);
    }

    //create nets
    for(int i = 0; i < insts; i++) {
        image_data0[i].push_back(image_blob);
        net[i].AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 32, 1, 1, 0, 0, leakyReLU, "conv1", image_data0[i], conv1[i]));
        net[i].AddLayer(new PoolingLayer<double>(5, 5, 5, 5, 0, 0, "pool2", conv1[i], pool2[i]));
        net[i].AddLayer(new ConvolutionalLayer<double>(5, 5, 32, 64, 1, 1, 0, 0, leakyReLU, "conv3", pool2[i], conv3[i]));
        net[i].AddLayer(new PoolingLayer<double>(3, 3, 3, 3, 0, 0, "pool4", conv3[i], pool4[i]));
        net[i].AddLayer(new FullyConnectedLayer<double>(800, leakyReLU, "fc5", pool4[i], fc5[i]));
        net[i].AddLayer(new FullyConnectedLayer<double>(2, leakyReLU, "fc6", fc5[i], fc6[i]));
        net[i].WeightsFromHDF5("weights/net" + to_string(i) + ".hdf5");
        net[i].phase(TEST);
    }

    cout<<"Forward"<<endl;

    //test iterations
    //parallel Forward for each instance
#pragma omp parallel for num_threads(insts)
    for(int inst = 0; inst < insts; inst++) {
        int tid = omp_get_thread_num();                     // thread id
        net[tid].Forward();
    }
    double fc_data[2];
    //get top data from each instance
    for(int inst = 0; inst < insts; inst++) {
        fc_data[0] += *fc6[inst][0]->data(0, 0, 0, 0); //*softmax[inst][0]->data(b, 0, 0, 0);
        fc_data[1] += *fc6[inst][0]->data(0, 0, 0, 1); //*softmax[inst][0]->data(b, 0, 0, 1);
    }
    //get average top data
    fc_data[0] /= 1.0 * insts;
    fc_data[1] /= 1.0 * insts;

    if(fc_data[1] > fc_data[0]) {
        // label = 1
        cout<<"cat"<<endl;
    } else {
        // label = 0
        cout<<"dog"<<endl;
    }

    return 0;
}

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
                            softmax,
                            loss;

    Data3d<double>* data0;
    if(argc < 4)
        return 0;

    QString input_image_path(argv[1]);
    QString output_image_path(argv[2]);
    string weights_path(argv[3]);

    cout<<"input_image_path: "<<input_image_path.toStdString()<<endl<<"output_image_path: "<<output_image_path.toStdString()<<endl;
    cout<<"weights_path: "<<weights_path<<endl;


    QImage input(input_image_path);
    input = input.convertToFormat(QImage::Format_RGB888);
    QImage output(input.width(), input.height(), QImage::Format_RGB888);

    Blob<double> data("data", Shape(5, 5, 3));
    Data3d<double>* img_data = data.Data(0);
    image_data0.push_back(&data);

//    net.AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 32, 3, 3, 0, 0, "conv0", image_data0, conv0));



    net.AddLayer(new ConvolutionalLayer<double>(3, 3, 3, 16, 1, 1, 0, 0, "fc1", image_data0, conv0));
    net.AddLayer(new FullyConnectedLayer<double>(300, "fc2", conv0, conv1));
    net.AddLayer(new FullyConnectedLayer<double>(2, "fc3", conv1, fc2));
    net.AddLayer(new SoftmaxLayer<double>("softmax",fc2, softmax));

    net.phase(TEST);
    net.WeightsFromHDF5(weights_path);
    for(int x = 0; x < input.width(); x++) {
        for( int y = 0; y < input.height(); y++) {
            output.setPixel(x, y, qRgb(0, 0, 0));
        }
    }
    for(int x = 0; x < input.width() - 5; x = x+1) {
        if(x%10 == 0) {
            cout<<"process: "<<(x * 1.0)/ (1.0 * input.width())<<endl;
        }
        for( int y = 0; y < input.height() - 5; y = y+1) {
            *img_data = input.copy(x, y, 5, 5);
            double* dat = img_data->data();
            Shape sh = img_data->shape();
            int num = sh.width() * sh.height() * sh.depth();
            for(int i = 0; i < num; i++) {
                dat[i] = (double) (dat[i] / 255.0);
            }
            net.Forward();
//            cout<<"x: "<<x<<" y: "<<y<<" fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<fc2[0]->data(0, 0, 0, 0)[1]<< " edge: "<<*softmax[0]->data(0, 0, 0, 0)<<endl;
            if(*softmax[0]->data(0, 0, 0, 0) == 1) {
                output.setPixel(x + 3, y + 3, qRgb(50, 250, 50));
            } else {
                output.setPixel(x + 3, y + 3, qRgb(0, 0, 0));
//                output.setPixel(x + 3, y + 3, input.pixel(x + 3, y + 3));
            }
        }
    }
    cout<<"save output image"<<endl;
    output.save(output_image_path);

//    for(int k = 0; k < iters; k++) {
//        lr_rate *= 0.9995;
//        cout<<"lr_rate: "<<lr_rate<<endl;
//        net.lr_rate(lr_rate);
//        net.weight_decay(weight_decay);
//        net.phase(TRAIN);
//        cout<<"phase_: TRAIN"<<endl;
//        for(int i = 0; i < train_iters; i++) {
//            net.Forward();


//            cout<<k * (train_iters+test_iters) + i<<" loss: "<<*loss[0]->data(0, 0, 0, 0)<<endl;
//            cout<<"fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;
////            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;
//            cout<<"sm: "<<<<" l: "<<((*image_data0[1]->data(0, 0, 0, 0) == 0) ? 1 : 0)<<endl;
//            net.Backward();
//        }

//        net.phase(TEST);
//        cout<<"phase_: TEST"<<endl;
//        double loss_ = 0;
//        int accuracy = 0;
//        for(int i = 0; i < test_iters; i++) {
//            net.Forward();

//            cout<<k * (train_iters+test_iters) + train_iters + i<<" loss: "<<*loss[0]->data(0, 0, 0, 0)<<endl;
//            cout<<"fc2: "<<*fc2[0]->data(0, 0, 0, 0)<<" "<<*fc2[0]->data(0, 0, 0, 1)<<endl;
////            cout<<"label: "<<*image_data0[1]->data(0, 0, 0, 0)<<" "<<*image_data0[1]->data(0, 0, 0, 1)<<endl;
//            int l = ((*image_data0[1]->data(0, 0, 0, 0) == 0 )? 1 : 0);
//            cout<<"sm: "<<*softmax[0]->data(0, 0, 0, 0)<<" l: "<<l<<endl;

//            if(l == *softmax[0]->data(0, 0, 0, 0)) {
//                accuracy++;
//            }
//            loss_ += *loss[0]->data(0, 0, 0, 0);
//        }
//        cout<<"accuracy: "<< (accuracy * 1.0) / (test_iters * 1.0) <<endl;
//        loss_ /= test_iters;
//        cout<<"test loss: "<<loss_<<endl;
//        net.WeightsToHDF5("net" + to_string(k) + ".hdf5");
//    }



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


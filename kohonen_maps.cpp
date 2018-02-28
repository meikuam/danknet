#include <iostream>
#include <unistd.h>
#include <fstream>

#include "danknet.h"


using namespace std;
using namespace danknet;

struct comma_separator : std::numpunct<char> {
    virtual char do_decimal_point() const override { return ','; }
};

int main(int argc, char *argv[])
{
    std::cout.imbue(std::locale(std::cout.getloc(), new comma_separator));
    cout<<"start"<<endl;

    if(argc < 3) {
        cout<<""<<endl;
        return 0;
    }

    QString input_train_path(argv[1]);
    QString input_test_path(argv[2]);

    int batch_size = 5;
    int test_iters = 10;
    int train_iters = 10;
    int iters = 10;


    int data_depth = 8;
    int data_labels = 2;
    int width = 4;
    int height = 4;

    //data Blobs
    vector<Blob<double>*>       input_data0,
                                kohonen_top;

    //create Net
    cout<<"create Net"<<endl;
    Net<double> net;
    net.AddLayer(new TextDataLayer<double>(data_depth, data_labels, batch_size, input_train_path.toStdString(), input_test_path.toStdString(), "text input", input_data0));
    net.AddLayer(new KohonenLayer<double>(width, height, "kohonen", input_data0, kohonen_top));
    Data3d<double>* data;

    cout<<"---------------------Forward----------------------"<<endl;

    for(int k = 0; k < iters; k++) {
        cout<<"phase_: TRAIN"<<endl;
        net.phase(TRAIN);
        for(int i = 0; i < train_iters; i++) {
            net.Forward();
            for(int b = 0; b < batch_size; b++) {
            }
//            net.Backward();
        }
        net.phase(TEST);
        cout<<"phase_: TEST"<<endl;
        for(int i = 0; i < test_iters; i++) {
            net.Forward();
        }
    }
    net.WeightsToHDF5("net.hdf5");

    return 0;
}

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
                            ip0,
                            ip1,
                            ip2,
                            ip3,
                            ip4,
                            softmax_top;
    cout<<"Add image_data layer"<<endl;
    double lr_rate = 0.04;


	return 0;
}

#include <iostream>
#include <unistd.h>
#include <fstream>

#include <QImage>
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
//    std::locale cp1251_locale("ru_RU.CP1251");
//    std::locale::global(cp1251_locale);
    cout<<"start"<<endl;
    //--------------------------------------------
    if(argc < 3)
        return 0;

    int insts = 4; //net instances - number of cores
    double train_data_percent = 0.6;

    //learning params
    int test_iters = 10;
    int train_iters = 20;
    int iters = 1000000;
    int step_size = 1000;

    int batch_size = 50;

    double lr_rate = 0.001;
    double weight_decay = 0.00005;
    double momentum = 0.5;
    double gamma = 0.2;



    string train_path(argv[1]);
    string test_path(argv[2]);
    cout<<"train path: "<<train_path<<endl<<"test path: "<<test_path<<endl;

    vector<QString> train_data_;
    vector<QString> test_data_;
    vector<int>     train_labels_;
    vector<int>     test_labels_;

    //read train images and labels
    ifstream train_file(train_path);
    string line;
    QString qline;
    while(getline(train_file, line)) {
        qline = QString::fromStdString(line);
        auto parts = qline.split(' ');
        train_data_.push_back(parts[0]);
        train_labels_.push_back(parts[1].toInt());
    }
    train_file.close();

    //read test images and labels
    ifstream test_file(test_path);
    while(getline(test_file, line)) {
        qline = QString::fromStdString(line);
        auto parts = qline.split(' ');
        test_data_.push_back(parts[0]);
        test_labels_.push_back(parts[1].toInt());
    }
    test_file.close();
    //create bags of training data
    vector<vector<QString>> train_data(insts);
    vector<vector<int>>     train_labels(insts);

//    default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
//    uniform_int_distribution<int> distribution(0, train_data_.size() - 1);
//    int train_data_size = train_data_.size() * train_data_percent;

//    for(int inst = 0; inst < insts; inst++) {
////        cout<<"inst: "<<inst<<endl;
//        for(int i = 0; i < train_data_size; i++) {
//            int item = distribution(generator);
////            cout<<item<<" ";
//            train_data[inst].push_back(train_data_[item]);
//            train_labels[inst].push_back(train_labels_[item]);
//        }
//        cout<<"items in "<<inst<<" "<<train_data[inst].size()<<endl;
//    }
    int train_data_len = train_data_.size() / insts;
    for(int inst = 0; inst < insts; inst++) {
        for(int i = inst * train_data_len; i < (inst + 1) * train_data_len; i++) {
             train_data[inst].push_back(train_data_[i]);
             train_labels[inst].push_back(train_labels_[i]);
        }
    }
    //create vector of nets
    vector<Net<double>> net(insts);
    //create vector of data
    vector<vector<Blob<double>*>> image_data0(insts), conv1(insts),
                                  pool2(insts), conv3(insts),
                                  pool4(insts), fc5(insts),
                                  fc6(insts), softmax(insts),
                                  loss(insts);

//    int num = 30;
    //create nets
    for(int i = 0; i < insts; i++) {
        net[i].AddLayer(new ImageDataLayer<double>(64, 64, 3, batch_size, 2,
                                                   train_data[i], train_labels[i],
                                                   test_data_, test_labels_,
                                                   "input0", image_data0[i]));

        net[i].AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 32, 1, 1, 0, 0, leakyReLU, "conv1", image_data0[i], conv1[i]));
        net[i].AddLayer(new PoolingLayer<double>(5, 5, 5, 5, 0, 0, "pool2", conv1[i], pool2[i]));
        net[i].AddLayer(new ConvolutionalLayer<double>(5, 5, 32, 64, 1, 1, 0, 0, leakyReLU, "conv3", pool2[i], conv3[i]));
        net[i].AddLayer(new PoolingLayer<double>(3, 3, 3, 3, 0, 0, "pool4", conv3[i], pool4[i]));
        net[i].AddLayer(new FullyConnectedLayer<double>(300, leakyReLU, "fc5", pool4[i], fc5[i]));
        net[i].AddLayer(new FullyConnectedLayer<double>(2, leakyReLU, "fc6", fc5[i], fc6[i]));

        vector<Blob<double>*> &net_top = fc6[i];
        net_top.push_back(image_data0[i][1]);
//        net[i].AddLayer(new SoftmaxLayer<double>("softmax", net_top, softmax[i]));
        net[i].AddLayer(new SoftmaxLossLayer<double>(leakyReLU, "loss", net_top, loss[i]));

        net[i].lr_rate(lr_rate);
        net[i].weight_decay(weight_decay);
        net[i].momentum(momentum);
        net[i].gamma(gamma);
        net[i].step_size(step_size);
//        net[i].WeightsFromHDF5("weights/net" + to_string(i) + " " + to_string(num) + ".hdf5");
    }

    cout<<"Forward"<<endl;
//    net.WeightsFromHDF5("net110.hdf5");
//    net.WeightsToHDF5("weights/net.hdf5");

    //train iterations
    for(int k = 0; k < iters; k++) {
        // phase TRAIN
        for(int inst = 0; inst < insts; inst++) {
            cout<<inst<<" lr_rate: "<<net[inst].lr_rate()<<endl;
            net[inst].phase(TRAIN);
            cout<<inst<<" phase_: TRAIN"<<endl;
        }
        for(int i = 0; i < train_iters; i++) {
            double train_accuracy_ = 0;
            double train_loss_ = 0;
#pragma omp parallel for num_threads(insts)
                for(int inst = 0; inst < insts; inst++) {
                    int tid = omp_get_thread_num();                     // thread id
                    double train_accuracy = 0;
                    double train_loss = 0;
                    double fc_data[2];

                    net[tid].Forward();

                    for(int b = 0; b < batch_size; b++) {
//                        if(((*image_data0[tid][1]->data(b, 0, 0, 0) == 0) ? 0 : 1) == *softmax[tid][0]->data(b, 0, 0, 0)) {
                        if(*fc6[tid][0]->data(b, 0, 0, 1) > *fc6[tid][0]->data(b, 0, 0, 0)) {
                            if(*image_data0[0][1]->data(b, 0, 0, 0) == 1) {
                                //true positive
                                train_accuracy++;
                            } else {
                                //false positive
                            }
                        } else {
                            if(*image_data0[0][1]->data(b, 0, 0, 0) == 0) {
                                //true negative
                                train_accuracy++;
                            } else {
                                //false negative
                            }
                        }
                        train_loss += *loss[tid][0]->data(b, 0, 0, 0);
                        fc_data[0] += *fc6[tid][0]->data(b, 0, 0, 0);
                        fc_data[1] += *fc6[tid][0]->data(b, 0, 0, 1);

                    }
                    fc_data[0] /= 1.0 * batch_size;
                    fc_data[1] /= 1.0 * batch_size;
                    train_accuracy /= batch_size;
                    train_loss /= batch_size;
                    cout<<k * (train_iters+test_iters) + i<<" tid: "<<tid<<" avg accuracy: "<<train_accuracy<<" avg loss: "<<train_loss<<" fc6: "<<fc_data[0]<<" "<<fc_data[1]<<endl;

                    #pragma omp critical
                    {
                        train_accuracy_ += train_accuracy;
                        train_loss_ += train_loss;
                    }
                    fc_data[0] = fc_data[0] = 0;

                    net[tid].Backward();
                }
            train_accuracy_ /= 1.0 * insts;
            train_loss_ /= 1.0 * insts;

            cout<<k * (train_iters+test_iters) + i<<" Avg accuracy: "<<train_accuracy_<<" Avg loss: "<<train_loss_<<endl;
        }


        //phase TEST
        for(int inst = 0; inst < insts; inst++) {
            net[inst].phase(TEST);
            cout<<inst<<" phase_: TEST"<<endl;
        }
        double test_accuracy = 0;
        double test_loss = 0;

        //test iterations
        for(int i = 0; i < test_iters; i++) {
            cout<<k * (test_iters + train_iters) + train_iters + i<<" ";
            //parallel Forward for each instance
#pragma omp parallel for num_threads(insts)
            for(int inst = 0; inst < insts; inst++) {
                int tid = omp_get_thread_num();                     // thread id
                net[tid].Forward();
            }

            double accuracy = 0;
            double loss_ = 0;
            double fc_data[batch_size][2];
            //get top data from each instance
            for(int inst = 0; inst < insts; inst++) {
                //batch iterations
                for(int b = 0; b < batch_size; b++) {
                    fc_data[b][0] += *fc6[inst][0]->data(b, 0, 0, 0); //*softmax[inst][0]->data(b, 0, 0, 0);
                    fc_data[b][1] += *fc6[inst][0]->data(b, 0, 0, 1); //*softmax[inst][0]->data(b, 0, 0, 1);
                }
            }
            for(int b = 0; b < batch_size; b++) {
                //get average top data
                fc_data[b][0] /= 1.0 * insts;
                fc_data[b][1] /= 1.0 * insts;
                //calc accuracy in batch
//                cout<<fc_data[b][0]<<" "<<fc_data[b][1];
                if(fc_data[b][1] > fc_data[b][0]) {
                    if(*image_data0[0][1]->data(b, 0, 0, 0) == 1) {
                        cout<<" + ";
                        //true positive
                        accuracy++;
                    } else {
                        cout<<" - ";
                        //false positive
                    }
                } else {
                    if(*image_data0[0][1]->data(b, 0, 0, 0) == 0) {
                        //true negative
                        cout<<" + ";
                        accuracy++;
                    } else {
                        cout<<" - ";
                        //false negative
                    }
                }
                loss_ += (fc_data[b][0] - *image_data0[0][1]->data(b, 0, 0, 0)) * (fc_data[b][0] - *image_data0[0][1]->data(b, 0, 0, 0));

            }
            accuracy /= batch_size;
            loss_ /= batch_size * 2.0;

            cout<<" accuracy: "<<accuracy<<" loss: "<<loss_<<endl;

            for(int b = 0; b < batch_size; b++) {
            }
            test_accuracy += accuracy;
            test_loss += loss_;
        }
        cout<<"Avg accuracy: "<< (test_accuracy * 1.0) / (test_iters * 1.0)<<" Avg loss: "<<(test_loss) / (test_iters * 1.0)<<endl;
//        loss_ /= test_iters * 1.0 * loss[0]->batch_size();
//        cout<<"test loss: "<<loss_<<endl;


        for(int inst = 0; inst < insts; inst++) {
            net[inst].WeightsToHDF5("weights/net" + to_string(inst) + " " + to_string(k) + ".hdf5");
        }
    }

	return 0;
}



//#include <iostream>
//#include <unistd.h>
//#include <fstream>

//#include <QImage>

//#include "danknet.h"

//using namespace std;
//using namespace danknet;

//bool fexists(const char *filename) {
//  std::ifstream ifile(filename);
//  return (bool)ifile;
//}


//int main(int argc, char *argv[])
//{
//    cout<<"start"<<endl;
//    //--------------------------------------------
//    Net<double> net;


//    vector<Blob<double>*>       image_data0,
//                                conv1,
//                                pool2,
//                                conv3,
//                                pool4,
//                                fc5,
//                                fc6,
//                                softmax,
//                                loss;

//    if(argc < 3)
//        return 0;

//    cout<<"Add image_data layer"<<endl;
//    string train_path(argv[1]);
//    string test_path(argv[2]);
//    cout<<"train path: "<<train_path<<endl<<"test path: "<<test_path<<endl;


//    cout<<"Add ImageDataLayer"<<endl;

//    net.AddLayer(new ImageDataLayer<double>(64, 64, 3, 4, 2,
//                                           train_path,
//                                           test_path,
//                                           "input0", image_data0));

//    net.AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 32, 1, 1, 0, 0, ReLU, "conv1", image_data0, conv1));
//    net.AddLayer(new PoolingLayer<double>(5, 5, 5, 5, 0, 0, "pool2", conv1, pool2));
//    net.AddLayer(new ConvolutionalLayer<double>(5, 5, 32, 64, 1, 1, 0, 0, ReLU, "conv3", pool2, conv3));
//    net.AddLayer(new PoolingLayer<double>(3, 3, 3, 3, 0, 0, "pool4", conv3, pool4));
//    net.AddLayer(new FullyConnectedLayer<double>(300, ReLU, "fc5", pool4, fc5));
//    net.AddLayer(new FullyConnectedLayer<double>(2, ReLU, "fc6", fc5, fc6));

//    vector<Blob<double>*> &net_top = fc6;
//    net_top.push_back(image_data0[1]);
//    net.AddLayer(new SoftmaxLayer<double>("softmax", net_top, softmax));
//    net.AddLayer(new LossLayer<double>("loss", net_top, loss));


//    cout<<"Forward"<<endl;
//    int test_iters = 100;
//    int train_iters = 500;
//    int iters = 1000000;
//    int step_size = 10000;

//    Data3d<double>* data0;

//    double lr_rate = 0.0001;
//    double weight_decay = 0.000005;
//    double momentum = 0.6;
//    double gamma = 0.1;

//    net.lr_rate(lr_rate);
//    net.weight_decay(weight_decay);
//    net.momentum(momentum);
//    net.gamma(gamma);
//    net.step_size(step_size);

////    net.WeightsFromHDF5("net110.hdf5");
//    net.WeightsToHDF5("weights/net.hdf5");

//    double fc_data[2];
//    for(int k = 0; k < iters; k++) {
//        cout<<"lr_rate: "<<net.lr_rate()<<endl;
//        net.phase(TRAIN);
//        cout<<"phase_: TRAIN"<<endl;
//        double train_accuracy = 0;
//        double train_loss = 0;
//        for(int i = 0; i < train_iters; i++) {
//            net.Forward();

//            for(int b = 0; b < softmax[0]-> batch_size(); b++) {
//                if(((*image_data0[1]->data(b, 0, 0, 0) == 0) ? 1 : 0) == *softmax[0]->data(b, 0, 0, 0)) {
//                    train_accuracy++;
//                }
//                train_loss += *loss[0]->data(b, 0, 0, 0);
//                fc_data[0] += *net_top[0]->data(b, 0, 0, 0);
//                fc_data[1] += *net_top[0]->data(b, 0, 0, 1);

//            }
//            fc_data[0] /= 1.0 * softmax[0]-> batch_size();
//            fc_data[1] /= 1.0 * softmax[0]-> batch_size();
//            train_accuracy /= softmax[0]-> batch_size();
//            train_loss /= softmax[0]-> batch_size();
//            cout<<k * (train_iters+test_iters) + i<<" avg accuracy: "<<train_accuracy<<" avg loss: "<<train_loss<<" fc2: "<<fc_data[0]<<" "<<fc_data[1]<<endl;
//            train_accuracy = 0;
//            train_loss = 0;

//            fc_data[0] = fc_data[0] = 0;

//            net.Backward();
//        }

//        net.phase(TEST);
//        cout<<"phase_: TEST"<<endl;
//        double test_accuracy = 0;
//        double test_loss = 0;

//        double loss_ = 0;
//        int accuracy = 0;
//        for(int i = 0; i < test_iters; i++) {
//            net.Forward();

//            for(int b = 0; b < softmax[0]-> batch_size(); b++) {
//                if(((*image_data0[1]->data(b, 0, 0, 0) == 0) ? 1 : 0) == *softmax[0]->data(b, 0, 0, 0)) {
//                    test_accuracy++;
//                }
//                test_loss += *loss[0]->data(b, 0, 0, 0);
//                fc_data[0] += *net_top[0]->data(b, 0, 0, 0);
//                fc_data[1] += *net_top[0]->data(b, 0, 0, 1);

//            }
//            fc_data[0] /= 1.0 * softmax[0]-> batch_size();
//            fc_data[1] /= 1.0 * softmax[0]-> batch_size();
//            test_accuracy /= softmax[0]-> batch_size();
//            test_loss /= softmax[0]-> batch_size();
//            cout<<k * (test_iters+test_iters) + i<<" avg accuracy: "<<test_accuracy<<" avg loss: "<<test_loss<<" fc2: "<<fc_data[0]<<" "<<fc_data[1]<<endl;
//            test_accuracy = 0;
//            test_loss = 0;

//            fc_data[0] = fc_data[0] = 0;


//            for(int b = 0; b < loss[0]->batch_size(); b++) {
//                if(((*image_data0[1]->data(b, 0, 0, 0) == 0 )? 1 : 0) == *softmax[0]->data(b, 0, 0, 0)) {
//                    accuracy++;
//                }
//                loss_ += *loss[0]->data(b, 0, 0, 0);
//            }
//        }
//        cout<<"accuracy: "<< (accuracy * 1.0) / (test_iters * 1.0 * loss[0]->batch_size()) <<endl;
//        loss_ /= test_iters * 1.0 * loss[0]->batch_size();
//        cout<<"test loss: "<<loss_<<endl;
//        net.WeightsToHDF5("weights/net" + to_string(k) + ".hdf5");
//    }

//	return 0;
//}


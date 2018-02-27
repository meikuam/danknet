#include <iostream>
#include <unistd.h>
#include <fstream>

#include <QImage>
#include <omp.h>

#include "danknet.h"

using namespace std;
using namespace danknet;


int main(int argc, char *argv[])
{
    cout<<"start"<<endl;
    //--------------------------------------------
    if(argc < 3)
        return 0;

    int insts = 8; //net instances - number of cores

    //learning params
    int test_iters = 20;
    int train_iters = 50;
    int iters = 1000000;
    int step_size = 300;

    int batch_size = 2;

    double lr_rate = 0.0001;
    double weight_decay = 0.0000005;
    double momentum = 0.3;
    double gamma = 0.5;



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
    vector<vector<QString>> train_data(insts), test_data(1);
    vector<vector<int>>     train_labels(insts), test_labels(1);

    int train_data_len = train_data_.size() / insts;
    for(int inst = 0; inst < insts; inst++) {
        for(int i = inst * train_data_len; i < (inst + 1) * train_data_len; i++) {
             train_data[inst].push_back(train_data_[i]);
             train_labels[inst].push_back(train_labels_[i]);
        }
    }

    int test_data_len = 40;
    for(int i = 0; i < test_data_len; i++) {
        test_data[0].push_back(test_data_[i]);
        test_labels[0].push_back(test_labels_[i]);
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
                                                   test_data[0], test_labels[0],
                                                   "input0", image_data0[i]));

        net[i].AddLayer(new ConvolutionalLayer<double>(5, 5, 3, 32, 1, 1, 0, 0, leakyReLU, "conv1", image_data0[i], conv1[i]));
        net[i].AddLayer(new PoolingLayer<double>(5, 5, 5, 5, 0, 0, "pool2", conv1[i], pool2[i]));
        net[i].AddLayer(new ConvolutionalLayer<double>(5, 5, 32, 64, 1, 1, 0, 0, leakyReLU, "conv3", pool2[i], conv3[i]));
        net[i].AddLayer(new PoolingLayer<double>(3, 3, 3, 3, 0, 0, "pool4", conv3[i], pool4[i]));
        net[i].AddLayer(new FullyConnectedLayer<double>(800, leakyReLU, "fc5", pool4[i], fc5[i]));
        net[i].AddLayer(new FullyConnectedLayer<double>(2, leakyReLU, "fc6", fc5[i], fc6[i]));

        vector<Blob<double>*> &net_top = fc6[i];
        net_top.push_back(image_data0[i][1]);
        net[i].AddLayer(new SoftmaxLossLayer<double>(leakyReLU, "loss", net_top, loss[i]));

        net[i].lr_rate(lr_rate);
        net[i].weight_decay(weight_decay);
        net[i].momentum(momentum);
        net[i].gamma(gamma);
        net[i].step_size(step_size);
    }

    cout<<"Forward"<<endl;

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
                    fc_data[b][0] += *fc6[inst][0]->data(b, 0, 0, 0);
                    fc_data[b][1] += *fc6[inst][0]->data(b, 0, 0, 1);
                }
            }
            for(int b = 0; b < batch_size; b++) {
                //get average top data
                fc_data[b][0] /= 1.0 * insts;
                fc_data[b][1] /= 1.0 * insts;
                //calc accuracy in batch
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

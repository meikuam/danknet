#include <iostream>
#include <unistd.h>
#include <fstream>

#include "danknet.h"


#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

using namespace std;
using namespace danknet;
using namespace QtCharts;

struct comma_separator : std::numpunct<char> {
    virtual char do_decimal_point() const override { return ','; }
};

int main(int argc, char *argv[])
{
    QLineSeries *series = new QLineSeries();
    series->setPointLabelsVisible(true);    // is false by default
    series->setPointLabelsColor(Qt::blue);
    series->setPointLabelsFormat("@yPoint");
    series->setColor(Qt::blue);

    QLineSeries *series2 = new QLineSeries();
    series2->setPointLabelsVisible(true);    // is false by default
    series2->setPointLabelsColor(Qt::red);
    series2->setPointLabelsFormat("@yPoint");
    series2->setColor(Qt::red);

    QLineSeries *loss_series = new QLineSeries();
    loss_series->setColor(Qt::black);





    std::cout.imbue(std::locale(std::cout.getloc(), new comma_separator));
    cout<<"start"<<endl;

    if(argc < 3) {
        cout<<"pred_data_train train.txt test.txt input_depth batch_size show_grapth lr_rate momentum weight_decay"<<endl;
        return 0;
    }

    QString input_train_path(argv[1]); //  /home/hotoru/datasets/dollar/dollar_abs_train.txt
    QString input_test_path(argv[2]);  //  /home/hotoru/datasets/dollar/dollar_abs_test.txt
//    QString input_train_path = "/home/hotoru/datasets/dollar/dollar_diff_train.txt";
//    QString input_test_path = "/home/hotoru/datasets/dollar/dollar_diff_test.txt";

    int input_depth = 5;
    int output_depth = 1;
    int batch_size = 5;

    double lr_rate = 0.01;
    double weight_decay = 0.000000;
    double momentum = 0.1;
    double gamma = 0.6;


    bool show_graph = true;
    if(argc >= 4) {
        input_depth = atoi(argv[3]);
    }
    if(argc >= 5) {
        batch_size = atoi(argv[4]);
    }
    if(argc >= 6) {
        show_graph = atoi(argv[5]) == 1 ? true : false;
    }
    if(argc >= 7) {
        lr_rate = atof(argv[6]);
    }
    if(argc >= 8) {
        momentum = atof(argv[7]);
    }
    if(argc >= 9) {
        weight_decay = atof(argv[8]);
    }

    int test_iters = (104 - input_depth) / batch_size;
    int train_iters = 5000;
    int iters = 800;
    int step_size = 50000;
    //data Blobs
    vector<Blob<double>*>       input_data0,
                                fc0_top,
                                fc1_top,
                                loss_top;
    //create Net
    cout<<"create Net"<<endl;
    Net<double> pred_net;
    pred_net.AddLayer(new DataLayer<double>(input_depth, output_depth, batch_size, input_train_path.toStdString(), input_test_path.toStdString(), "input", input_data0));
    pred_net.AddLayer(new FullyConnectedLayer<double>(10, bipolSigmoid, "fc0", input_data0, fc0_top));
    pred_net.AddLayer(new FullyConnectedLayer<double>(1, bipolSigmoid, "fc2", fc0_top, fc1_top));

    vector<Blob<double>*> &net_top = fc1_top;
    net_top.push_back(input_data0[1]);
    pred_net.AddLayer(new LossLayer<double>(bipolSigmoid, "loss", net_top, loss_top));




    Data3d<double>* data;



    pred_net.lr_rate(lr_rate);
    pred_net.weight_decay(weight_decay);
    pred_net.momentum(momentum);
    pred_net.gamma(gamma);
    pred_net.step_size(step_size);

    cout<<"lr_rate "<<lr_rate<<";"<<endl;
    cout<<"weight_decay "<<weight_decay<<";"<<endl;
    cout<<"momentum "<<momentum<<";"<<endl;
    cout<<"gamma "<<gamma<<";"<<endl;
    cout<<"test_iters "<<test_iters<<";"<<endl;
    cout<<"train_iters "<<train_iters<<";"<<endl;
    cout<<"iters "<<iters<<";"<<endl;
    cout<<"step_size "<<step_size<<";"<<endl;
    cout<<"---------------------Forward----------------------"<<endl;

    double loss_ = 0;

    for(int k = 0; k < iters; k++) {
//        cout<<"lr_rate: "<<pred_net.lr_rate()<<endl;
        pred_net.phase(TRAIN);
//        cout<<"phase_: TRAIN"<<endl;
        double iter_loss = 0;
        for(int i = 0; i < train_iters; i++) {
            pred_net.Forward();
            double batch_loss = 0;
            for(int b = 0; b < batch_size; b++) {
                batch_loss += *loss_top[0]->data(b, 0, 0, 0);
//                cout<<"net top: "<<*net_top[0]->data(b, 0, 0, 0)<<" label: "<<*input_data0[1]->data(b, 0, 0, 0)<<endl;
            }
            batch_loss /= (double)batch_size;
            iter_loss += batch_loss;
            pred_net.Backward();
        }
        iter_loss /= (double)train_iters;
        loss_ += iter_loss;

//        *series << QPointF(k * train_iters, iter_loss);
//        cout<<k * train_iters<<" avg loss: "<<iter_loss<<" net top: "<<*net_top[0]->data(0, 0, 0, 0)<<" label: "<<*input_data0[1]->data(0, 0, 0, 0)<<endl;
        *loss_series<< QPointF(k*train_iters, iter_loss);


//        pred_net.phase(TEST);
////        cout<<"phase_: TEST"<<endl;
//        double test_loss = 0;
//        double loss_ = 0;
//        for(int i = 0; i < test_iters; i++) {
//            pred_net.Forward();
//            test_loss = 0;
//            for(int b = 0; b < batch_size; b++) {
//                test_loss += *loss_top[0]->data(b, 0, 0, 0);
//            }
//            test_loss /= batch_size;
////            cout<<k * (test_iters + train_iters) + train_iters + i<<" avg loss: "<<test_loss<<" net top: "<<*net_top[0]->data(0, 0, 0, 0)<<" label: "<<*input_data0[1]->data(0, 0, 0, 0)<<endl;
////            cout<<test_loss<<" "<<*net_top[0]->data(0, 0, 0, 0)<<" "<<*input_data0[1]->data(0, 0, 0, 0)<<endl;
//            loss_ += test_loss;
//        }
//        loss_ /= test_iters;
////        cout<<k * (test_iters + train_iters) + train_iters + test_iters<<" avg loss: "<<loss_<<endl;
////        cout<<loss_<<";"<<endl;
////        pred_net.WeightsToHDF5("weights/pred_net" + to_string(k) + ".hdf5");
    }
    loss_ /= (double)iters;
    cout<<train_iters * iters<<" avg loss: "<<loss_<<endl;



    pred_net.phase(TEST);
    for(int i = 0; i < test_iters; i++) {
        pred_net.Forward();
        for(int b = 0; b < batch_size; b++) {

            *series << QPointF(i*batch_size + b, *net_top[0]->data(b, 0, 0, 0));
            *series2 << QPointF(i*batch_size + b, *input_data0[1]->data(b, 0, 0, 0));
            cout<<*input_data0[1]->data(b, 0, 0, 0)<<";"<<*net_top[0]->data(b, 0, 0, 0)<<";"<<endl;
        }
    }


    if(show_graph) {

        QApplication a(argc, argv);

        //data predicted
        QChart *chart = new QChart();
        chart->legend()->hide();
        chart->addSeries(series);
        chart->addSeries(series2);
        chart->createDefaultAxes();
        chart->setTitle("Predicted data - blue, true data - red. Input depth = " + QString::number(input_depth));

        QChartView *chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        QMainWindow window;
        window.setCentralWidget(chartView);
        window.resize(1000, 500);
        window.show();

        //loss
        QChart *chart2 = new QChart();
        chart2->legend()->hide();
        chart2->addSeries(loss_series);
        chart2->createDefaultAxes();
        chart2->setTitle("Loss");

        QChartView *chartView2 = new QChartView(chart2);
        chartView2->setRenderHint(QPainter::Antialiasing);

        QMainWindow window2;
        window2.setCentralWidget(chartView2);
        window2.resize(1000, 500);
        window2.show();
        return a.exec();
    }
    return 0;
}

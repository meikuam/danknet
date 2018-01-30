#ifndef NET_H
#define NET_H

#include <vector>

using namespace std;

class Net {
private:
    vector<Layer*>    layers_;
public:
    Net();
    ~Net();
};
#endif // NET_H

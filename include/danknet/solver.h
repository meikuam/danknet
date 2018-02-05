#ifndef SOLVER_H
#define SOLVER_H

namespace danknet {


template <typename Dtype>
class Solver {
 public:
    explicit Solver();
    virtual void Fit();

};
} // namespace danknet

#endif // SOLVER_H

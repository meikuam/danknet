#ifndef COMMON_H
#define COMMON_H

namespace danknet {

#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

} // namespace danknet
#endif // COMMON_H

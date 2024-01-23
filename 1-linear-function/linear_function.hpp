#include "numbers.hpp"

template <math::Ring T>
class LinearFunction
{
public:
  LinearFunction(T m, T n)
  {
    m_ = m;
    n_ = n;
  }
  T operator()(T x) const
  {
    return m_ * x + n_;
  }

private:
  T m_;
  T n_;
};

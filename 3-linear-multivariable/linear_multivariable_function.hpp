#include <numbers.hpp>
#include <matrix.hpp>

template <math::Ring T>
class LinearMultivariableFunction
{
  using Matrix = math::Matrix<T>;

public:
  LinearMultivariableFunction(Matrix m, Matrix n) : m_(m), n_(n)
  {
  }
  Matrix operator()(Matrix x) const
  {
    return x * m_ + n_;
  }

private:
  Matrix m_;
  Matrix n_;
};

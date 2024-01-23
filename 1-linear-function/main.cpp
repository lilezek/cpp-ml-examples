#include "linear_function.hpp"
#include <iostream>
#include <stdlib.h>

#include <numbers.hpp>
#include <matrix.hpp>

// Parameters of the neural network and the learning process
auto w = math::Matrix{{1.l}};
auto b = math::Matrix{{0.l}};
auto learningRate = 0.1f;
auto learningRateDecay = 0.99f;
auto seed = 0;

math::Matrix<long double> NeuralNetwork(const math::Matrix<long double> &x)
{
  return w * x + b;
}

void UpdateWeightAndBias(
    const math::Matrix<long double> &x,
    const math::Matrix<long double> &y,
    const math::Matrix<long double> &yHat)
{

  auto dLdyHat = (y - yHat) * 2.;
  auto dyHatdw = x;

  auto dLdw = dLdyHat * dyHatdw;
  auto dLdb = dLdyHat;

  w = w + dLdw * learningRate;
  b = b + dLdb * learningRate;

  learningRate *= learningRateDecay;
}

int main()
{
  srand(seed);
  auto m = math::RandomRange(-100.l, 100.l);
  auto n = math::RandomRange(-100.l, 100.l);

  LinearFunction lf(m, n);

  for (int i = 0; i < 50; i++)
  {
    auto x = math::RandomRange(-5.l, 5.l);
    auto y = math::Matrix{{lf(x)}};
    auto yHat = NeuralNetwork(math::Matrix{{x}});

    UpdateWeightAndBias(math::Matrix{{x}}, y, yHat);
  }

  std::cout << "m: " << m << " - w: " << w;
  std::cout << "n: " << n << " - b: " << b;

  return 0;
}
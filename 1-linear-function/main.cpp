#include "linear_function.hpp"
#include <iostream>
#include <stdlib.h>

#include <numbers.hpp>
#include <matrix.hpp>

// Parameters of the neural network and the learning process
auto w = math::Matrix{{1.f}};
auto b = math::Matrix{{0.f}};
auto learningRate = 0.1f;
auto learningRateDecay = 0.99f;
auto seed = 0;

math::Matrix<float> NeuralNetwork(const math::Matrix<float> &x)
{
  return w * x + b;
}

void UpdateWeightAndBias(
    const math::Matrix<float> &x,
    const math::Matrix<float> &y,
    const math::Matrix<float> &yHat)
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
  auto m = math::RandomRange(-100.f, 100.f);
  auto n = math::RandomRange(-100.f, 100.f);

  LinearFunction lf(m, n);

  for (int i = 0; i < 50; i++)
  {
    auto x = math::RandomRange(-5.f, 5.f);
    auto y = math::Matrix{{lf(x)}};
    auto yHat = NeuralNetwork(math::Matrix{{x}});

    UpdateWeightAndBias(math::Matrix{{x}}, y, yHat);
  }

  std::cout << "m: " << m << " - w: " << w;
  std::cout << "n: " << n << " - b: " << b;

  return 0;
}
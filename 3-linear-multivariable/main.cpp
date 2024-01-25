#include <iostream>
#include <matrix.hpp>
#include "linear_multivariable_function.hpp"
#include "../2-computing-grads/grads.hpp"

using Graph = OperationGraph<float>;
using Node = OperationNode<float>;

const auto identity = math::Matrix<float>::identity(2);
const auto learning_rate_decay = 0.9999f;
auto learning_rate = 0.01f;

// The operation:
// L = (y - yHat)^2
// yHat = w * x + b
Graph graph(8);

Node &x = graph.CreateLeaf();
Node &w = graph.CreateLeaf();
Node &b = graph.CreateLeaf();
Node &y = graph.CreateLeaf();

// (w * x)
Node &w_by_x = graph.CreateNode(
    w, x,
    [](auto w, auto x)
    { return x * w; },
    [](auto w, auto x)
    { return x.transpose(); },
    [](auto w, auto x)
    { return w.transpose(); });

// (w * x) + b
Node &yHat = graph.CreateNode(
    w_by_x, b,
    [](auto w_by_x, auto b)
    { return w_by_x + b; },
    [](auto w_by_x, auto b)
    { return identity; },
    [](auto w_by_x, auto b)
    { return identity; });

// (y - yHat)^2
Node &L = graph.CreateNode(
    y, yHat,
    [](auto y, auto yHat)
    { return (y - yHat) * (y - yHat); },
    [](auto y, auto yHat)
    { return (yHat - y) * 2; },
    [](auto y, auto yHat)
    { return (y - yHat) * 2; });

int main()
{
  w = math::Matrix<float>::identity(2);
  b = math::Matrix<float>::zero(2, 2);

  auto m = math::RandomMatrix(-5.f, 5.f, 2, 2);
  auto n = math::RandomMatrix(-5.f, 5.f, 2, 2);

  // auto m = math::Matrix<float>::identity(2);
  // auto n = math::Matrix<float>::identity(2);

  LinearMultivariableFunction<float> lf(m, n);

  for (int i = 0; i < 1000; i++)
  {
    x = math::RandomMatrix(-5.f, 5.f, 2, 2);
    y = lf(x.value());

    auto value = L.forward();
    // std::cout << "L: " << value;

    L.backward();
    // std::cout << "w gradient: " << w.gradient();
    // std::cout << "b gradient: " << b.gradient();

    w = w.value() + w.gradient() * learning_rate;
    b = b.value() + b.gradient() * learning_rate;
    learning_rate = learning_rate * learning_rate_decay;

    // std::cout << "w: " << w.value();
    // std::cout << "b: " << b.value() << std::endl;
  }

  std::cout << "m: " << m;
  std::cout << "n: " << n;
  std::cout << "w: " << w.value();
  std::cout << "b: " << b.value() << std::endl;

  return 0;
}
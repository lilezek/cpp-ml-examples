#include <iostream>
#include <matrix.hpp>
#include "../1-linear-function/linear_function.hpp"
#include "grads.hpp"

using Graph = OperationGraph<float>;
using Node = OperationNode<float>;

const auto identity = math::Matrix<float>::identity(1);
const auto learning_rate_decay = 0.99f;
auto learning_rate = 0.1f;

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
    { return w * x; },
    [](auto w, auto x)
    { return x; },
    [](auto w, auto x)
    { return w; });

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
  w = math::Matrix{{1.f}};
  b = math::Matrix{{0.f}};

  auto m = math::RandomRange(-100.f, 100.f);
  auto n = math::RandomRange(-100.f, 100.f);

  LinearFunction lf(m, n);

  for (int i = 0; i < 100; i++)
  {
    auto xRand = math::RandomRange(-5.f, 5.f);
    x = math::Matrix{{xRand}};
    y = math::Matrix{{lf(xRand)}};

    auto value = L.forward();
    std::cout << "L: " << value;

    L.backward();
    // std::cout << "w gradient: " << w.gradient();
    // std::cout << "b gradient: " << b.gradient();

    w = w.value() + w.gradient() * learning_rate;
    b = b.value() + b.gradient() * learning_rate;
    learning_rate = learning_rate * learning_rate_decay;

    // std::cout << "w: " << w.value();
    // std::cout << "b: " << b.value() << std::endl;
  }

  std::cout << "w: " << w.value();
  std::cout << "b: " << b.value() << std::endl;

  return 0;
}
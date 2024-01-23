#pragma once
#include <concepts>
#include <stdlib.h>

namespace math
{
  // The minimum requirements for a type to be used in matrices and such.
  template <typename T>
  concept Ring = requires(T a, T b) {
    {
      a + b
    } -> std::convertible_to<T>;
    {
      a *b
    } -> std::convertible_to<T>;
    {
      a - b
    } -> std::convertible_to<T>;
    {
      a / b
    } -> std::convertible_to<T>;
  };

  template <Ring T>
  T RandomRange(T min, T max)
  {
    return (T)rand() / RAND_MAX * (max - min) + min;
  }
}
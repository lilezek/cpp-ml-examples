#pragma once
#include "numbers.hpp"

#include <vector>
#include <memory>

namespace math
{
  /**
   * @brief Very basic implementation of a matrix.
   */
  template <Ring T>
  class Matrix
  {
  private:
    std::vector<T> data_;
    int rows_;
    int columns_;

  public:
    [[nodiscard]] static Matrix<T> diagonal(const T &value, int size)
    {
      Matrix<T> result(size, size);
      for (int i = 0; i < size; i++)
      {
        result(i, i) = value;
      }
      return result;
    }

    [[nodiscard]] static Matrix<T> identity(int size)
    {
      return diagonal((T)1, size);
    }

    [[nodiscard]] static Matrix<T> zero(int rows, int columns)
    {
      return Matrix<T>(rows, columns);
    }

    Matrix(int rows,
           int columns) : data_(rows * columns),
                          rows_(rows),
                          columns_(columns)
    {
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list) : Matrix(list.size(), list.begin()->size())
    {
      // Copy the data from the initializer list
      int i = 0;
      for (auto row : list)
      {
        for (auto element : row)
        {
          data_[i] = element;
          i++;
        }
      }
    }

    [[nodiscard]] int rows() const
    {
      return rows_;
    }

    [[nodiscard]] int columns() const
    {
      return columns_;
    }

    T &operator()(int row, int column)
    {
      return data_[row * columns_ + column];
    }

    [[nodiscard]] T operator()(int row, int column) const
    {
      return data_[row * columns_ + column];
    }

    Matrix operator+(const Matrix &other) const
    {
      Matrix result(rows_, columns_);
      for (int i = 0; i < rows_; i++)
      {
        for (int j = 0; j < columns_; j++)
        {
          result(i, j) = (*this)(i, j) + other(i, j);
        }
      }

      return result;
    }

    Matrix operator-(const Matrix &other) const
    {
      Matrix result(rows_, columns_);
      for (int i = 0; i < rows_; i++)
      {
        for (int j = 0; j < columns_; j++)
        {
          result(i, j) = (*this)(i, j) - other(i, j);
        }
      }

      return result;
    }

    Matrix operator*(const Matrix &other) const
    {
      Matrix result(rows_, other.columns_);
      for (int i = 0; i < rows_; i++)
      {
        for (int j = 0; j < other.columns_; j++)
        {
          T sum = 0;
          for (int k = 0; k < columns_; k++)
          {
            sum += (*this)(i, k) * other(k, j);
          }
          result(i, j) = sum;
        }
      }

      return result;
    }

    Matrix operator*(const T &scalar) const
    {
      Matrix result(rows_, columns_);
      for (int i = 0; i < rows_; i++)
      {
        for (int j = 0; j < columns_; j++)
        {
          result(i, j) = (*this)(i, j) * scalar;
        }
      }

      return result;
    }

    Matrix transpose() const
    {
      Matrix result(columns_, rows_);
      for (int i = 0; i < rows_; i++)
      {
        for (int j = 0; j < columns_; j++)
        {
          result(j, i) = (*this)(i, j);
        }
      }
      return result;
    }
  };

  template <Ring T>
  std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix)
  {
    for (int i = 0; i < matrix.rows(); i++)
    {
      for (int j = 0; j < matrix.columns(); j++)
      {
        os << matrix(i, j) << " ";
      }
      os << std::endl;
    }
    return os;
  }
}
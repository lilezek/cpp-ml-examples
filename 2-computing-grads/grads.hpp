#include <functional>
#include <vector>

#include <matrix.hpp>
#include <numbers.hpp>

template <math::Ring T>
class OperationNode;

template <math::Ring T>
class OperationGraph
{
  friend class OperationNode<T>;
  using Matrix = math::Matrix<T>;
  using Operation = std::function<Matrix(Matrix, Matrix)>;

private:
  std::vector<OperationNode<T>> nodes_;

public:
  OperationGraph(size_t capacity)
  {
    nodes_.reserve(capacity);
  }

  [[nodiscard]] OperationNode<T> &CreateLeaf(size_t rows = 1, size_t columns = 1)
  {
    // Check if vector has enough space
    if (nodes_.size() >= nodes_.capacity()) [[unlikely]]
    {
      throw std::runtime_error("Allocate enough space for the nodes.");
    }

    nodes_.push_back(OperationNode<T>(math::Matrix<T>(rows, columns)));
    return nodes_.back();
  }

  [[nodiscard]] OperationNode<T> &CreateNode(
      OperationNode<T> &left,
      OperationNode<T> &right,
      Operation operation,
      Operation leftGradient,
      Operation rightGradient)
  {
    if (nodes_.size() >= nodes_.capacity()) [[unlikely]]
    {
      throw std::runtime_error("Allocate enough space for the nodes.");
    }

    nodes_.push_back(OperationNode<T>(
        left, right,
        operation,
        leftGradient,
        rightGradient));
    return nodes_.back();
  }
};

template <math::Ring T>
class OperationNode
{
  friend class OperationGraph<T>;
  using Matrix = math::Matrix<T>;
  using Operation = std::function<Matrix(Matrix, Matrix)>;
  using Child = OperationNode<T> *const;

private:
  Child left_;
  Child right_;
  Operation operation_;
  Operation leftGradient_;
  Operation rightGradient_;

  Matrix value_;
  Matrix gradient_;

  /**
   * @brief Constructs a new leaf object. This is private because
   *        nodes can only be constructed by the OperationGraph class.
   */
  OperationNode(const Matrix &m) : value_(m),
                                   gradient_(m),
                                   left_(nullptr),
                                   right_(nullptr)
  {
  }

  /**
   * @brief Constructs a new node object. This is private because
   *        nodes can only be constructed by the OperationGraph class.
   */
  OperationNode(OperationNode<T> &left,
                OperationNode<T> &right,
                Operation operation,
                Operation leftGradient,
                Operation rightGradient) : left_(&left),
                                           right_(&right),
                                           operation_(operation),
                                           leftGradient_(leftGradient),
                                           rightGradient_(rightGradient),
                                           value_(Matrix::zero(1, 1)),
                                           gradient_(Matrix::zero(1, 1))
  {
  }

public:
  Matrix forward()
  {
    if (left_ == nullptr && right_ == nullptr)
    {
      return value_;
    }

    auto leftValue = left_->forward();
    auto rightValue = right_->forward();

    value_ = operation_(leftValue, rightValue);

    return value_;
  }

  void backward(const Matrix &gradient = Matrix::identity(1))
  {
    if (left_ == nullptr && right_ == nullptr)
    {
      gradient_ = gradient;
      return;
    }

    auto leftValue = left_->value_;
    auto rightValue = right_->value_;

    auto leftGradient = leftGradient_(leftValue, rightValue);
    auto rightGradient = rightGradient_(leftValue, rightValue);

    left_->backward(leftGradient * gradient);
    right_->backward(rightGradient * gradient);
  }

  [[nodiscard]] const Matrix &gradient() const
  {
    return gradient_;
  }

  [[nodiscard]] const Matrix &value() const
  {
    return value_;
  }

  OperationNode &operator=(const Matrix &m)
  {
    value_ = m;
    return *this;
  }
};
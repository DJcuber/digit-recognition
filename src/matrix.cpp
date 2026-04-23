#include <neural_network/matrix.h>

#include <stdexcept>
#include <vector>

// WARN:
#include <iostream>

neural_network::Matrix::Matrix() {}

neural_network::Matrix::Matrix(std::vector<std::vector<double>>& _mat)
    : mat(_mat) {
  y = mat.size();
  if (!mat.empty())
    x = mat[0].size();
  else
    x = 0;
}

neural_network::Matrix::Matrix(unsigned _y, unsigned _x)
    : x(_x),
      y(_y),
      mat(std::vector<std::vector<double>>(_y, std::vector<double>(_x))) {}

std::vector<double> neural_network::Matrix::multiply(std::vector<double>& v) {
  if ((size_t)this->x != v.size()) {
    throw std::runtime_error("Undefined matrix multiplication");
  }

  std::vector<double> new_vector(this->y);

  for (int i = 0; i < this->y; ++i) {
    double sum{0};
    for (int j = 0; j < this->x; ++j) {
      sum += v[j] * this->mat[i][j];
    }
    new_vector[i] = sum;
  }

  return new_vector;
}

void neural_network::Matrix::add(neural_network::Matrix m) {
  if (this->x != m.x || this->y != m.y) {
    throw std::runtime_error("Undefined matrix addition");
  }

  for (int i = 0; i < this->y; ++i) {
    for (int j = 0; j < this->x; ++j) {
      this->mat[i][j] += m.mat[i][j];
    }
  }
}

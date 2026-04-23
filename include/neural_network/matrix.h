#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>

namespace neural_network {
class Matrix {
 public:
  int x{}, y{};
  std::vector<std::vector<double>> mat;

  Matrix();
  Matrix(std::vector<std::vector<double>>& _mat);
  Matrix(unsigned _y, unsigned _x);

  std::vector<double> multiply(std::vector<double>& v);
  void add(Matrix m);
};
};  // namespace neural_network

#endif

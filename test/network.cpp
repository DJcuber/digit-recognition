#include <neural_network/network.h>
#include <test/network.h>

#include <cassert>
#include <cstddef>
#include <vector>

// WARN:
#include <iostream>

bool test_next_line() {
  neural_network::init_stream("mnist/mnist_train.csv");
  neural_network::Data expected = {
      5, {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   18,
          18,  18,  126, 136, 175, 26,  166, 255, 247, 127, 0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   30,  36,  94,  154, 170, 253,
          253, 253, 253, 253, 225, 172, 253, 242, 195, 64,  0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   49,  238, 253, 253, 253, 253, 253,
          253, 253, 253, 251, 93,  82,  82,  56,  39,  0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   18,  219, 253, 253, 253, 253, 253,
          198, 182, 247, 241, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   80,  156, 107, 253, 253, 205,
          11,  0,   43,  154, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   14,  1,   154, 253, 90,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   139, 253, 190,
          2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   11,  190, 253,
          70,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   35,  241,
          225, 160, 108, 1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   81,
          240, 253, 253, 119, 25,  0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          45,  186, 253, 253, 150, 27,  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   16,  93,  252, 253, 187, 0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   249, 253, 249, 64,  0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          46,  130, 183, 253, 253, 207, 2,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   39,  148,
          229, 253, 253, 253, 250, 182, 0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   24,  114, 221, 253,
          253, 253, 253, 201, 78,  0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   23,  66,  213, 253, 253, 253,
          253, 198, 81,  2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   18,  171, 219, 253, 253, 253, 253, 195,
          80,  9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   55,  172, 226, 253, 253, 253, 253, 244, 133, 11,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   136, 253, 253, 253, 212, 135, 132, 16,  0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}};

  neural_network::Data result = neural_network::next_line();
  assert(expected.value == result.value);
  for (std::size_t i = 0; i < expected.image.size(); ++i) {
    if (expected.image[i] / 255.0 != result.image[i]) {
      std::cout << i << ' ' << expected.image[i] << ' ' << result.image[i]
                << '\n';
    }
    assert((expected.image[i] / 255.0) == result.image[i]);
  }

  return true;
}

bool test_network_constructor() {
  neural_network::Matrix expected(3, 4);

  expected.mat = {{-0.31, 0.48, -0.49, 0.77},
                  {0.47, -0.27, 0.31, 0.51},
                  {-0.41, -0.05, -0.1, 0.96}};
  // {-0.11, 0.2, 0.5, 0.47}};

  neural_network::Network result("test_network.csv", {3, 3, 1});

  for (std::size_t i = 0; i < expected.mat.size(); ++i) {
    for (std::size_t j = 0; j < expected.mat[i].size(); ++j) {
      assert(std::abs(expected.mat[i][j] - result.weights[0].mat[i][j]) <=
             0.01);
    }
  }

  result.write_network("test_network.csv");

  return true;
}

bool test_activations() {
  std::vector<std::vector<double>> expected = {
      {1.0, 2.0, 3.0}, {0.49, 0.8, 0.54}, {0.7}};

  neural_network::Network net("test_network.csv", {3, 3, 1});

  auto result = net.get_activations(expected[0]);

  for (std::size_t i = 0; i < expected.size(); ++i) {
    for (std::size_t j = 0; j < expected[i].size(); ++j) {
      assert(std::abs(expected[i][j] - result[i][j]) <= 0.1);
    }
  }

  return true;
}

bool test_deltas() {
  std::vector<double> input = {0.0, 0.5, 1.0};
  std::vector<std::vector<double>> expected = {
      {-0, -0.00217106, 0}, {-0.0075325, 0.0130506, 0.0308995}, {0.292846}};

  neural_network::Network net("test_network.csv", {3, 3, 1});
  auto activations = net.get_activations(input);
  auto deltas = net.get_deltas(activations, {0});

  for (std::size_t i = 0; i < deltas.size(); ++i) {
    for (std::size_t j = 0; j < deltas[i].size(); ++j) {
      // std::cout << deltas[i][j] << ' ';
      assert(std::abs(expected[i][j] - deltas[i][j]) <= 0.1);
    }
    // std::cout << '\n';
  }

  return true;
}

bool test_backprop_thread() {
  neural_network::init_stream("mnist/mnist_train.csv");

  neural_network::Network network(
      std::vector<int>{neural_network::kDataLength, 20, 20, 10});

  for (std::size_t i = 0; i < network.weights.size(); ++i) {
    for (int j = 0; j < network.weights[i].y - 1; ++j) {
      for (int k = 0; k < network.weights[i].x; ++k) {
        network.weights[i].mat[j][k] = 0.1;
      }
    }
  }

  std::vector<neural_network::Matrix> grad(network.weights.size());
  for (std::size_t layer = 0; layer < network.weights.size(); ++layer) {
    grad[layer] = neural_network::Matrix(network.weights[layer].y - 1,
                                         network.weights[layer].x);
  }

  neural_network::backpropagation_thread(&network, grad);
  // TODO: THIS WORKS BUT NEED TO FINISH TEST

  /*
  for (int j = 0; j < grad[1].y; ++j) {
    for (int k = 0; k < grad[1].x; ++k) {
      std::cout << grad[1].mat[j][k] << ' ';
    }
    std::cout << '\n';
  }
  */

  return true;
}

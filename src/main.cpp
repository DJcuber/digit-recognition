#include <neural_network/network.h>

int main() {
  neural_network::init_stream("mnist/mnist_train.csv");
  // /*
  neural_network::Network network(
      "network.csv", std::vector<int>{neural_network::kDataLength, 20, 20, 10});
  // */
  /*
  neural_network::Network network(
      std::vector<int>{neural_network::kDataLength, 20, 20, 10});
  */

  network.backpropagation(10, "mnist/mnist_train.csv");
  network.write_network("network.csv");
}

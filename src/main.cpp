#include <neural_network/network.h>

#include <filesystem>
#include <iostream>
#include <memory>

int main() {
  neural_network::Network network;
  std::destroy_at(&network);
  if (!std::filesystem::exists("network.csv")) {
    std::construct_at(
        &network, std::vector<int>{neural_network::kDataLength, 50, 50, 10});
  } else {
    std::construct_at(
        &network, "network.csv",
        std::vector<int>{neural_network::kDataLength, 50, 50, 10});
  }

  int option;
  std::cout << "1. train\n2. test\n";
  std::cin >> option;

  switch (option) {
    case 1:
      network.backpropagation(1, "mnist/mnist_train.csv");
      network.write_network("network.csv");
      break;

    case 2:
      network.test("mnist/mnist_test.csv");
  }
}

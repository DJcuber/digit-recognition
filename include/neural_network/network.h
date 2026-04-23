#ifndef NETWORK_H_
#define NETWORK_H_

#include <neural_network/matrix.h>

#include <mutex>
#include <semaphore>
#include <string>
#include <vector>

namespace neural_network {
inline constexpr int kDataLength = 784;
inline constexpr int kDatasetSize = 60000;
inline constexpr int kBatchSize = 100;
inline constexpr int kThreads = 10;

inline constexpr double kLearningRate = 0.01;
inline constexpr double kRegularization = 0.1;

struct Data {
  int value;
  std::vector<double> image;
};

class Network {
 public:
  std::vector<int> layers;
  std::vector<Matrix> weights;

  std::mutex grad_mu;
  std::counting_semaphore<kThreads> threads_sem{kThreads};

  Network();
  Network(const std::vector<int>& _layers);
  Network(const std::string& _network_file, const std::vector<int>& _layers);

  void write_network(const std::string& _network_file);

  std::vector<std::vector<double>> get_activations(
      const std::vector<double>& input);
  std::vector<std::vector<double>> get_deltas(
      const std::vector<std::vector<double>>& activations,
      const std::vector<double>& desired);

  void backpropagation(unsigned epochs, const std::string& _training_file);
  void test(const std::string& _test_file);
};

void init_stream(const std::string& _training_file);
Data next_line();

void backpropagation_thread(Network* network, std::vector<Matrix>& grad);
void test_thread(Network* network, volatile int& passed);

};  // namespace neural_network

#endif

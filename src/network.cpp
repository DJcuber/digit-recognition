#include <neural_network/network.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

// WARN:
#include <iostream>

static std::string training_filename;
static std::ifstream training_file;
static std::random_device rand_device;

static std::mutex file_mu;

static constexpr double sigmoid(double z) { return 1 / (1 + std::exp(-z)); }

neural_network::Data neural_network::next_line() {
  neural_network::Data data{0,
                            std::vector<double>(neural_network::kDataLength)};

  std::string line;
  {
    const std::lock_guard<std::mutex> lock(file_mu);
    if (!std::getline(training_file, line)) {
      training_file = std::ifstream(training_filename);
      std::getline(training_file, line);
    }
  }
  int l = 0, r = 0;

  while (line[r] != ',') ++r;
  data.value = std::stoi(line.substr(l, r - l));
  l = r + 1;
  r = l;

  for (int i = 0; i < neural_network::kDataLength - 2; ++i) {
    while (line[r] != ',') ++r;
    data.image[i] = std::stoi(line.substr(l, r - l)) / 255.0;
    l = r + 1;
    r = l;

    data.image[neural_network::kDataLength - 2] = std::stoi(line.substr(l));
  }

  return data;
}

void neural_network::init_stream(const std::string& _training_filename) {
  training_filename = _training_filename;
  training_file = std::ifstream(training_filename);
}

neural_network::Network::Network(const std::vector<int>& _layers)
    : layers(_layers) {
  if (this->layers.size() <= 1) {
    throw std::runtime_error("Invalid network");
  }

  this->weights = std::vector<neural_network::Matrix>(this->layers.size() - 1);

  constexpr int range = 100000;
  std::default_random_engine e1(rand_device());
  std::uniform_int_distribution<int> uniform_dist(0, 2 * range);

  for (std::size_t i = 0; i < this->weights.size(); ++i) {
    this->weights[i] =
        neural_network::Matrix(this->layers[i + 1] + 1, this->layers[i] + 1);
    for (int j = 0; j < this->weights[i].y - 1; ++j) {
      for (int k = 0; k < this->weights[i].x; ++k) {
        int rand = uniform_dist(rand_device);
        this->weights[i].mat[j][k] =
            (rand > range ? rand - 2 * range : rand) / (double)range;
      }
    }
    for (int k = 0; k < this->weights[i].x - 1; ++k) {
      this->weights[i].mat.back()[k] = 0;
    }
    this->weights[i].mat.back().back() = 1;
  }
}

neural_network::Network::Network(const std::string& _network_file,
                                 const std::vector<int>& _layers)
    : layers(_layers) {
  if (this->layers.size() <= 1) throw std::runtime_error("Invalid network");

  std::ifstream file(_network_file);
  this->weights = std::vector<neural_network::Matrix>(this->layers.size() - 1);

  for (std::size_t i = 0; i < this->weights.size(); ++i) {
    this->weights[i] =
        neural_network::Matrix(this->layers[i + 1] + 1, this->layers[i] + 1);

    for (int j = 0; j < this->weights[i].y - 1; ++j) {
      std::string line;
      std::getline(file, line);
      int l = 0, r = 0;

      for (int k = 0; k < this->weights[i].x - 1; ++k) {
        while (line[r] != ',' && line[r] != '\n' && line[r] != EOF) r++;
        this->weights[i].mat[j][k] = std::stof(line.substr(l, r - l));
        l = r + 1;
        r = l;
      }
      this->weights[i].mat[j].back() = std::stof(line.substr(l));
    }
    for (int k = 0; k < this->weights[i].x - 1; ++k) {
      this->weights[i].mat.back()[k] = 0;
    }
    this->weights[i].mat.back().back() = 1;
  }
  file.close();
}

void neural_network::Network::write_network(const std::string& _network_file) {
  std::ofstream file(_network_file);
  for (std::size_t i = 0; i < this->weights.size(); ++i) {
    for (int j = 0; j < this->weights[i].y - 1; ++j) {
      for (int k = 0; k < this->weights[i].x - 1; ++k) {
        file << this->weights[i].mat[j][k] << ',';
      }
      file << this->weights[i].mat[j].back() << '\n';
    }
  }
  file.close();
}

std::vector<std::vector<double>> neural_network::Network::get_activations(
    const std::vector<double>& input) {
  std::vector<std::vector<double>> activations(this->layers.size());
  activations[0] = input;
  activations[0].push_back(1);
  for (std::size_t i = 1; i < this->layers.size(); ++i) {
    activations[i] = this->weights[i - 1].multiply(activations[i - 1]);
    for (std::size_t j = 0; j < activations[i].size() - 1; ++j) {
      activations[i][j] = sigmoid(activations[i][j]);
    }
  }
  for (auto& v : activations) {
    v.pop_back();
  }
  return activations;
}

std::vector<std::vector<double>> neural_network::Network::get_deltas(
    const std::vector<std::vector<double>>& activations,
    const std::vector<double>& desired) {
  std::vector<std::vector<double>> deltas = activations;
  for (std::size_t i = 0; i < activations.back().size(); ++i) {
    auto node = activations.back()[i];
    deltas.back()[i] = 2 * (node - desired[i]) * node * (1 - node);
  }

  for (int layer = (int)activations.size() - 2; layer >= 0; --layer) {
    for (std::size_t curr_node = 0; curr_node < activations[layer].size();
         ++curr_node) {
      deltas[layer][curr_node] = 0.0;
      for (std::size_t next_node = 0; next_node < activations[layer + 1].size();
           ++next_node) {
        deltas[layer][curr_node] +=
            deltas[layer + 1][next_node] *
            this->weights[layer].mat[next_node][curr_node];
      }
      auto node = activations[layer][curr_node];
      deltas[layer][curr_node] *= node * (1 - node);
    }
  }

  return deltas;
}

void neural_network::Network::backpropagation(
    unsigned epochs, const std::string& _training_filename) {
  using namespace neural_network;

  init_stream(_training_filename);

  for (std::size_t batch = 0; batch < epochs * (kDatasetSize / kBatchSize);
       ++batch) {
    std::vector<Matrix> grad(this->weights.size());
    for (std::size_t layer = 0; layer < this->weights.size(); ++layer) {
      grad[layer] = Matrix(this->weights[layer].y - 1, this->weights[layer].x);
    }

    // Sums the gradient of each of the data points of the batch
    std::vector<std::thread> threads(kBatchSize);
    for (int i = 0; i < kBatchSize; ++i) {
      this->threads_sem.acquire();
      threads[i] = std::thread(backpropagation_thread, this, std::ref(grad));
    }
    for (auto& v : threads) {
      v.join();
    }

    // Regularization
    for (std::size_t i = 0; i < this->weights.size(); ++i) {
      for (int j = 0; j < this->weights[i].y - 1; ++j) {
        for (int k = 0; k < this->weights[i].x - 1; ++k) {
          this->weights[i].mat[j][k] =
              this->weights[i].mat[j][k] *
                  (1.0 - kLearningRate * kRegularization / kBatchSize) -
              kLearningRate * grad[i].mat[j][k];
        }
        this->weights[i].mat[j].back() -= kLearningRate * grad[i].mat[j].back();
      }
    }

    if ((batch * kBatchSize) % 10000 == 0 && (batch != 0)) {
      std::cout << batch * kBatchSize << '\n';
    }
  }
  std::cout << "Done!\n";
}

void neural_network::backpropagation_thread(neural_network::Network* network,
                                            std::vector<Matrix>& grad) {
  auto input = neural_network::next_line();
  auto activations = network->get_activations(input.image);

  std::vector<double> desired(10);
  desired[input.value] = 1.0;

  auto deltas = network->get_deltas(activations, desired);

  {
    std::lock_guard<std::mutex> lock(network->grad_mu);
    for (std::size_t i = 0; i < grad.size(); ++i) {
      for (int j = 0; j < grad[i].y; ++j) {
        for (int k = 0; k < grad[i].x - 1; ++k) {
          grad[i].mat[j][k] += deltas[i + 1][j] * activations[i][k];
        }
        grad[i].mat[j].back() += deltas[i + 1][j];
      }
    }
  }

  network->threads_sem.release();
}

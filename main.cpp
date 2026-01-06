#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <filesystem>

#define IMAGE_SIZE 28
#define FILE_LINE_LENGTH 785

#define DATASET_SIZE 60000
#define BATCH_SIZE 100
#define LEARNING_RATE 0.5
#define REGULARIZATION 0.27


std::ifstream trainingFile;
std::random_device rand_device;

std::vector<double> nextLine() {
  std::vector<double> data(FILE_LINE_LENGTH);
  std::string line;
  std::getline(trainingFile, line);
  int l = 0, r = 0;
    
  while (line[r] != ',') r++;
  data[FILE_LINE_LENGTH-1] = std::stoi(line.substr(l, r-l));
  l = r+1;
  r = l;

  for (int i = 0; i < FILE_LINE_LENGTH-2; i++) {
    while (line[r] != ',') r++;
    data[i] = std::stoi(line.substr(l, r-l)) / (double) 255;
    l = r+1;
    r = l;
  }
  data[FILE_LINE_LENGTH-2] = std::stoi(line.substr(l));

  return data;
}

void writeNetwork(std::string filename, std::vector<std::vector<std::vector<double>>> network) {
  std::ofstream networkFile(filename);
  for (int i = 0; i < network.size(); i++) {
    for (int j = 0; j < network[i].size(); j++) {
      for (int k = 0; k < network[i][j].size()-1; k++) {
        networkFile << network[i][j][k] << ',';
      }
      networkFile << network[i][j].back() << '\n';
    }
  }

  networkFile.close();
}

std::vector<std::vector<std::vector<double>>> readNetwork(std::string filename, std::vector<int> nodes) {
  std::vector<std::vector<std::vector<double>>> network;
  std::ifstream networkFile("network.csv");
  std::string line;
  
  for (int i = 0; i < nodes.size()-1; i++) {
    network.push_back({});
    for (int j = 0; j < nodes[i+1]; j++) {
      network[i].push_back({});
      std::getline(networkFile, line);
      int l = 0, r = 0;
      for (int k = 0; k < nodes[i]; k++) {
        while (line[r] != ',' && line[r] != '\n' && line[r] != EOF) r++;
        network[i][j].push_back(std::stof(line.substr(l, r-l)));
        l = r + 1;
        r = l;
      }
      network[i][j].push_back(std::stof(line.substr(l)));
    }
  }

  networkFile.close();

  return network;
}

std::vector<std::vector<double>> genRandMatrix(int inputs, int nodes) {
  int range = 100000;
  std::default_random_engine e1(rand_device());
  std::uniform_int_distribution<int> uniform_dist(0, 2*range);

  std::vector<std::vector<double>> mat(nodes, std::vector<double>(inputs+1));
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < inputs+1; j++) {
      int rand = uniform_dist(rand_device);
      mat[i][j] = (rand > range ? rand - 2 * range : rand) / (double) range;
    }
  }

  return mat;
}

void printImage(std::vector<double> image) {
  for (int i = 0; i < IMAGE_SIZE; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      if (image[IMAGE_SIZE*i + j] > 0)
        std::cout << '#';
      else
        std::cout << '.'; 
    }
    std::cout << '\n';
  }
}

std::vector<std::vector<std::vector<double>>> createNetwork(std::vector<int> nodes) {
  std::vector<std::vector<std::vector<double>>> network;

  for (int i = 0; i < nodes.size()-1; i++) {
    network.push_back(genRandMatrix(nodes[i], nodes[i+1]));
  }

  return network;
}

double sigmoid(double z) {
  return 1 / (1 + expf(-z));
}

std::vector<std::vector<double>> getActivations(std::vector<std::vector<std::vector<double>>> network, std::vector<double> input) {
  std::vector<std::vector<double>> activations(network.size()+1);
  for (auto v : input) {
    activations[0].push_back(v);
  }

  for (int i = 0; i < network.size(); i++) {
    for (int j = 0; j < network[i].size(); j++) {
      double sum = 0;
      for (int k = 0; k < network[i][j].size()-1; k++) {
        sum += activations[i][k]*network[i][j][k];
      }
      sum += network[i][j].back();
      activations[i+1].push_back(sigmoid(sum));
    }
  }
  return activations;
}

std::vector<std::vector<double>> getDelta(std::vector<std::vector<std::vector<double>>> network, std::vector<std::vector<double>> activations, std::vector<double> desired) {
  std::vector<std::vector<double>> delta = activations;


  for (int i = 0; i < activations.back().size(); i++) {
    // Find regularization error
    delta[delta.size()-1][i] = 2*(activations.back()[i]-desired[i])*activations.back()[i]*(1-activations.back()[i]);
  }

  for (int i = delta.size()-2; i >= 0; i--) {
    for (int j = 0; j < delta[i].size(); j++) {
      delta[i][j] = 0.0f;
      for (int k = 0; k < delta[i+1].size(); k++) { 
        delta[i][j] += delta[i+1][k]*network[i][k][j];
      }
      delta[i][j] *= activations[i][j]*(1-activations[i][j]);
    }
  }

  return delta;
}

void test(std::vector<std::vector<std::vector<double>>> network) { 
  trainingFile = std::ifstream("mnist/mnist_test.csv");
  int test_length = 10000;
  int correct = 0;
  for (int test = 0; test < test_length; test++) {
    auto image = nextLine();
    int ans = image.back();
    image.pop_back();
    auto activations = getActivations(network, image);
    double maxi = activations.back()[0];
    int predict = 0;
    //std::cout << maxi << ' ';
    for (int i = 1; i < 10; i++) {
      //std::cout << activations.back()[i] << ' ';
      if (activations.back()[i] > maxi) {
        maxi = activations.back()[i];
        predict = i;
      }
    }
    if (predict == ans) correct++;
  }

  std::cout << "Correct: " << correct << ", Total: " << test_length << ", Accuracy: " << (double) correct / test_length << '\n';
}

void backpropigation(std::vector<std::vector<std::vector<double>>>& network, int epochs) {
  trainingFile = std::ifstream("mnist/mnist_train.csv");
  for (int batch = 0; batch < epochs*DATASET_SIZE/BATCH_SIZE; batch++) {
  //for (int batch = 0; batch < 1; batch++) {
    std::vector<std::vector<std::vector<double>>> slope(network.size());
    for (int i = 0; i < network.size(); i++) {
      for (int j = 0; j < network[i].size(); j++) {
        slope[i].push_back({});
        for (int k = 0; k < network[i][j].size(); k++) {
          slope[i][j].push_back(0.0f);
        }
      }
    }
    for (int sample = 0; sample < BATCH_SIZE; sample++) {
    //for (int sample = 0; sample < 1; sample++) {
      auto image = nextLine();
      int number = image[FILE_LINE_LENGTH-1];
      std::vector<double> desired(10);
      desired[number] = 1.0f;
      image.pop_back();
      auto activations = getActivations(network, image);
      auto delta = getDelta(network, activations, desired);

      //for (int i = 0; i < delta[0].size(); i++) std::cout << (delta[0][i] > 0.0f) << ' ';
      /*
      for (auto a : delta) {
        for (auto b : a) {
          std::cout << b << ' ';
        }
        std::cout << '\n';
      }
      std::cout << '\n';
      */
      /*
      for (auto a : activations) {
        for (auto b : a) {
          std::cout << b << ' ';
        }
        std::cout << '\n';
      }
      */


      for (int i = 0; i < delta.size()-1; i++) {
        for (int j = 0; j < delta[i].size(); j++) {
          for (int k = 0; k < delta[i+1].size(); k++) {
            slope[i][k][j] += delta[i+1][k]*activations[i][j];
          }
        }
      }
      
      for (int i = 0; i < slope.size(); i++) {
        for (int j = 0; j < delta[i+1].size(); j++) {
          slope[i][j].back() += delta[i+1][j];
        }
      }
    }
    for (int i = 0; i < network.size(); i++) {
      for (int j = 0; j < network[i].size(); j++) {
        for (int k = 0; k < network[i][j].size()-1; k++) {
          network[i][j][k] = network[i][j][k] * (1 - LEARNING_RATE * REGULARIZATION / BATCH_SIZE) - LEARNING_RATE * slope[i][j][k];
          //network[i][j][k] -= LEARNING_RATE * slope[i][j][k];
        }
        network[i][j].back() -= LEARNING_RATE * slope[i][j].back();
      }
    }

    if ((batch * BATCH_SIZE) % 10000 == 0 && batch != 0) std::cout << batch * BATCH_SIZE << '\n';
    if (batch % epochs == 0) {
      trainingFile.close();
      trainingFile = std::ifstream("mnist/mnist_train.csv");
    }
  }
  trainingFile.close();
  std::cout << "done\n";
}

void custom_network() {
  std::vector<std::vector<std::vector<double>>> network = {
    {
      {-0.1, -0.23, 0.96},
      {0.05, -0.44, 0.24}
    },
    {
      {-0.048, 0.35, 0.64}
    }
  };

  /*
  for (auto a : network) {
    for (auto b : a) {
      for (auto c : b) 
        std::cout << c << ' ';
      std::cout << '\n';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
  */
  
  for (int a = 0; a < 100000; a++) { 
  //for (int i = 0; i < 1; i++) {
    std::vector<double> input = {(a%2 == 0 ? 1 : 0), (a%4 > 1 ? 1 : 0)};

    auto activations = getActivations(network, input);

    std::vector<double> desired(1);
    desired[0] = (double)((int)input[0] ^ (int)input[1]);
    auto delta = getDelta(network, activations, desired);

    //std::cout << delta[1][0] << " ";
    //std::cout << (desired[0] - activations.back()[0]) * (desired[0] - activations.back()[0]) << ", ";
    //std::cout << network[1][0][0] << ' ';

    for (int i = 0; i < delta.size()-1; i++) {
      for (int j = 0; j < delta[i].size(); j++) {
        for (int k = 0; k < delta[i+1].size(); k++) {
          network[i][k][j] -= delta[i+1][k] * activations[i][j];
        }
      }
    }

    for (int i = 0; i < network.size(); i++) {
      for (int j = 0; j < delta[i+1].size(); j++) {
        network[i][j].back() -= delta[i+1][j];
      }
    }
  }
    
  /*
  auto activations = getActivations(network, {0, 0});
  for (auto a : activations) {
    for (auto b : a) {
      std::cout << b << ' ';
    }
    std::cout << '\n';
  }
  */

  /*
  for (auto a : network) {
    for (auto b : a) {
      for (auto c : b) 
        std::cout << c << ' ';
      std::cout << '\n';
    }
    std::cout << '\n';
  }
  */

  ///*
  for (int i = 0; i < 4; i++) {
    std::vector<double> input = {(i%4 > 1 ? 1 : 0), (i%2 == 0 ? 0 : 1)};
    auto activations = getActivations(network, input);
    std::cout << activations.back()[0] << '\n';
  }
  //*/
}

void visual(std::vector<std::vector<std::vector<double>>> network) {
  trainingFile = std::ifstream("mnist/mnist_test.csv");
  int test_length = 10000;
  for (int test = 0; test < test_length; test++) {
    auto image = nextLine();
    int ans = image.back();
    image.pop_back();
    auto activations = getActivations(network, image);
    double maxi = activations.back()[0];
    int predict = 0;
    std::cout << maxi << ' ';
    for (int i = 1; i < 10; i++) {
      std::cout << activations.back()[i] << ' ';
      if (activations.back()[i] > maxi) {
        maxi = activations.back()[i];
        predict = i;
      }
    }
    std::cout << '\n';
    printImage(image);
    std::cout << predict << '\n';
    int temp;
    std::cin >> temp;
  }
}

int main() {
  std::vector<std::vector<std::vector<double>>> network;
  if (!std::filesystem::exists("network.csv")) {
    network = createNetwork({FILE_LINE_LENGTH-1, 20, 20, 10});
    writeNetwork("network.csv", network);
  }
  else
    network = readNetwork("network.csv", {FILE_LINE_LENGTH-1, 20, 20, 10});

  int option;
  std::cout << "1. train network\n2. test network\n3. custom network\n4. visual\n";
  std::cin >> option;

  switch (option) {
    case 1:  
      backpropigation(network, 1);
      writeNetwork("network.csv", network);
      break;
    case 2:
      test(network);
      break;
    case 3:
      custom_network();
      break;
    case 4:
      visual(network);
      break;
  }

  return 0;
}

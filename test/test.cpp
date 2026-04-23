#include <test/network.h>

int main() {
  test_next_line();
  test_network_constructor();
  test_activations();
  test_deltas();
  test_backprop_thread();
}

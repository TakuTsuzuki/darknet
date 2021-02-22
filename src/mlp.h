#ifndef DARKNET_MLP
#define DARKNET_MLP

#include "darknet.h"

void train_iris_classifier();
void test_mlp_hidden_layer_forward();
void test_train_iris_classifier();

network **make_mlp_network();
network *make_mlp_single_network();
list *make_sample_network_config();

#endif //DARKNET_MLP
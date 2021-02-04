#ifndef DARKNET_MLP
#define DARKNET_MLP

#include "darknet.h"

void train_iris_classifier();
void test_mlp_hidden_layer_forward();

network **make_mlp_network();
network *make_mlp_single_network();
list *make_cfg();
list *make_data_cfg();
data load_iris_data();


#endif //DARKNET_MLP
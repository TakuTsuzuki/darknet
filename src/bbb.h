#ifndef DARKNET_BBB_H
#define DARKNET_BBB_H

#include "darknet.h"

void train_iris_bbb_classifier();

network **make_bbb_network();
network *make_bbb_single_network();
list *make_sample_bbb_network_config();

#endif //DARKNET_BBB_H

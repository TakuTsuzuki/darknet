#ifndef DARKNET_CONFIG_H
#define DARKNET_CONFIG_H
#include "list.h"

list *make_data_config(int num_config, char *configs[][2]);
list *make_network_config(int num_categories, char **categories, int *num_configs, char *configs[][2]);

#endif //DARKNET_CONFIG_H

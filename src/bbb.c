#include <stdio.h>
#include <string.h>
#include "bbb.h"
#include "network.h"
#include "list.h"
#include "option_list.h"
#include "parser.h"
#include "data.h"
#include "iris_loader.h"
#include "config.h"

#ifdef ACCELERATOR
#include "connected_layer.h"
#endif

char *BBB_DATA_CFG[][2] = {
        {"classes", "3"},
        {"train", "train_path"},
        {"valid", "test_path"},
        {"labels", "label_path"},
        {"backup", "backup_path"},
        {"top", "2"}
};
int NUM_BBB_DATA_CFG = 6;

/* modified train_classifier @ examples/classifier.c */
network **make_bbb_network()
{
    int ngpus = 1;
    network **nets = calloc(ngpus, sizeof(network*));

    list *sections = make_sample_bbb_network_config();
    nets[0] = make_mlp_single_network(sections);

    network *net = nets[0];

    list *options = make_data_config(NUM_BBB_DATA_CFG, BBB_DATA_CFG);
    // int tag = option_find_int_quiet(options, "tag", 0);
    // char *label_list = option_find_str(options, "labels", "data/labels.list");
    // char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    // int classes = option_find_int(options, "classes", 2);
    free_list(options);
    options = NULL;

    return nets;
}

int NUM_BBB_NETWORK_CFG_TYPES = 4;
char *BBB_NETWORK_CFG_TYPES[] = {
        "[net]",
        "[bayesconnected]",
        "[bayesconnected]",
        "[softmax]"
};
int NUM_BBB_NETWORK_NET_CFG[4] = {16, 18, 20, 21};
char *BBB_NETWORK_NET_CFG[][2] = {
        // [net]
        // {"batch", "8"},
        {"batch", "1"},
        {"subdivisions", "1"},
        {"height", " 4"},
        {"width", " 4"},
        {"channels", "3"},
        {"max_crop", "4"},
        {"min_crop", "4"},
        {"hue", ".1"},
        {"saturation", ".75"},
        {"exposure", ".75"},
        {"learning_rate", "0.00001"},
        {"policy", "poly"},
        {"power", "4"},
        {"max_batches", "100"},
        {"momentum", "0.9"},
        {"decay", "0.0005"},

        // [bayesconnected]
        {"output", "10"},
        {"activation", "relu"},

        // [bayesconnected]
        {"output", "3"},
        {"activation", "linear"},

        // [softmax]
        {"groups", "1"}
};

list *make_sample_bbb_network_config() {
    return make_network_config(NUM_BBB_NETWORK_CFG_TYPES, BBB_NETWORK_CFG_TYPES, NUM_BBB_NETWORK_NET_CFG, BBB_NETWORK_NET_CFG);
}

/*
void train_iris_classifier()
{
    int N = 75;
    list *sections = make_sample_network_config();
    printf("started making MLP network.\n");
    network *net = make_mlp_single_network(sections);
    printf("finished making MLP network.\n");

    float avg_loss = -1;
    data train;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0) {
        printf("started loading data.\n");
        train = load_iris_data(1);
        printf("finished loading data.\n");

        printf("started training.\n");
        float loss = train_network(net, train);
        printf("finished training.\n");

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), *net->seen);
    }

    // free_list(sections);
    // free_network(net);
}
*/

void train_iris_bbb_classifier()
{
    int N = 75;
    list *sections = make_sample_bbb_network_config();
    printf("started making BBB network.\n");
    network *net = make_mlp_single_network(sections);
    printf("finished making BBB network.\n");

    int sampling = 1;
    int num_sample = 3;
    float avg_loss = -1;
    data train;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0) {
        printf("started loading data.\n");
        train = load_iris_data(1);
        printf("finished loading data.\n");

        printf("started training.\n");
        // float loss = train_network(net, train);
        float loss = train_network_bbb(net, train, sampling, num_sample);
        printf("finished training.\n");

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), *net->seen);
    }

    // free_list(sections);
    // free_network(net);
}

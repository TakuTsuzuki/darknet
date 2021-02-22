#include <stdio.h>
#include <string.h>
#include "mlp.h"
#include "network.h"
#include "list.h"
#include "option_list.h"
#include "parser.h"
#include "data.h"
#include "iris_loader.h"
#include "config.h"
#include "connected_layer.h"

char *DATA_CFG[][2] = {
        {"classes", "3"},
        {"train", "train_path"},
        {"valid", "test_path"},
        {"labels", "label_path"},
        {"backup", "backup_path"},
        {"top", "2"}
};
int NUM_DATA_CFG = 6;

/* modified train_classifier @ examples/classifier.c */
network **make_mlp_network()
{
    int ngpus = 1;
    network **nets = calloc(ngpus, sizeof(network*));

    list *sections = make_sample_network_config();
    nets[0] = make_mlp_single_network(sections);

    network *net = nets[0];

    list *options = make_data_config(NUM_DATA_CFG, DATA_CFG);
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

int NUM_NETWORK_CFG_TYPES = 4;
char *NETWORK_CFG_TYPES[] = {
        "[net]",
        "[connected]",
        // "[dropout]",
        "[connected]",
        "[softmax]"
};
int NUM_NETWORK_NET_CFG[4] = {16, 18, 20, 21};
char *NETWORK_NET_CFG[][2] = {
        // [net]
        // {"batch", "8"},
        {"batch", "1"},
        {"subdivisions", "1"},
        {"height", "4"},
        {"width", "4"},
        {"channels", "3"},
        {"max_crop", "4"},
        {"min_crop", "4"},
        {"hue", ".1"},
        {"saturation", ".75"},
        {"exposure", ".75"},
        {"learning_rate", "0.1"},
        {"policy", "poly"},
        {"power", "4"},
        // {"max_batches", "5000"},
        {"max_batches", "100"},
        {"momentum", "0.9"},
        {"decay", "0.0005"},

        // [connected]
        {"output", "10"},
        {"activation", "relu"},

        // [dropout]
        // {"probability", ".5"},

        // [connected]
        {"output", "3"},
        {"activation", "linear"},

        // [softmax]
        {"groups", "1"}
};

/*
int NUM_NETWORK_CFG_TYPES = 5;
char *NETWORK_CFG_TYPES[] = {
        "[net]",
        "[connected]",
        "[dropout]",
        "[connected]",
        "[softmax]"
};
int NUM_NETWORK_NET_CFG[5] = {16, 18, 19, 21, 22};
char *NETWORK_NET_CFG[][2] = {
        // [net]
        // {"batch", "8"},
        {"batch", "1"},
        {"subdivisions", "1"},
        {"height", "4"},
        {"width", "4"},
        {"channels", "3"},
        {"max_crop", "4"},
        {"min_crop", "4"},
        {"hue", ".1"},
        {"saturation", ".75"},
        {"exposure", ".75"},
        {"learning_rate", "0.1"},
        {"policy", "poly"},
        {"power", "4"},
        // {"max_batches", "5000"},
        {"max_batches", "100"},
        {"momentum", "0.9"},
        {"decay", "0.0005"},

        // [connected]
        {"output", "10"},
        {"activation", "relu"},

        // [dropout]
        {"probability", ".5"},

        // [connected]
        {"output", "3"},
        {"activation", "linear"},

        // [softmax]
        {"groups", "1"}
};
*/

list *make_sample_network_config() {
    return make_network_config(NUM_NETWORK_CFG_TYPES, NETWORK_CFG_TYPES, NUM_NETWORK_NET_CFG, NETWORK_NET_CFG);
}

void forward_hidden_layer(network *net, data d) {
    // assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;
        net->train = 1;
        forward_network(net);
    }
}

void forward_hidden_layer_accelerator(network *net, data d) {
    // assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        *net->seen += net->batch;
        net->train = 1;
        layer l = net->layers[0];
#ifdef ACCELERATOR
        forward_connected_layer_accelerator(l, *net);
#else
        forward_connected_layer(l, *net);
#endif
    }
}

void print_layers(network *net) {
    for (int i = 0; i < net->n; i++) {
        layer l = net->layers[i];
        printf("layer %d\n", i);
        for (int j = 0; j < l.outputs; j++) {
            printf("output[%d] = %f\n", j, l.output[j]);
        }
    }
}

void print_layer(layer l) {
    for (int i = 0; i < l.outputs; i++) {
        printf("%d: output=%f, bias=%f\n", i, l.output[i], l.biases[i]);
    }
    // for (int i = 0; i < l.outputs * l.inputs; i++) {
    //    printf("weight[%d] = %f\n", i, l.weight[i]);
    // }
}

void test_mlp_hidden_layer_forward() {
    list *sections_accelerator = make_sample_network_config();
    list *sections = make_sample_network_config();
    printf("started making MLP network.\n");
    network *net = make_mlp_single_network(sections);
    net->layers[0].forward = forward_connected_layer;
    net->layers[1].forward = forward_connected_layer;
    network *net_accelerator = make_mlp_single_network(sections_accelerator);
#ifdef ACCELERATOR
    net_accelerator->layers[0].forward = forward_connected_layer_accelerator;
    net_accelerator->layers[1].forward = forward_connected_layer_accelerator;
#endif
    for (int i = 0; i < net->layers[0].inputs * net->layers[0].outputs; i++) {
        net_accelerator->layers[0].weights[i] = net->layers[0].weights[i];
    }
    for (int i = 0; i < net->layers[1].inputs * net->layers[1].outputs; i++) {
        net_accelerator->layers[1].weights[i] = net->layers[1].weights[i];
    }
    printf("finished making MLP network.\n");

    data train;
    printf("started loading data.\n");
    train = load_iris_data(1);
    printf("finished loading data.\n");

    printf("started forward test.\n");
    forward_hidden_layer(net, train);
    // forward_hidden_layer_accelerator(net_accelerator, train);
    forward_hidden_layer(net_accelerator, train);
    printf("finished forward test.\n");

    // print output
    printf("CPU\n");
    print_layer(net->layers[0]);
    print_layer(net->layers[1]);

    printf("Accelerator\n");
    print_layer(net_accelerator->layers[0]);
    print_layer(net_accelerator->layers[1]);

    // free_list(sections);
    // free_network(net);
}

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

void test_train_iris_classifier()
{
    int N = 75;
    list *sections = make_sample_network_config();
    list *sections_cpu = make_sample_network_config();

    printf("started making MLP network.\n");
    network *net = make_mlp_single_network(sections);
    network *net_cpu = make_mlp_single_network(sections_cpu);
    net_cpu->layers[0].forward = forward_connected_layer;
    net_cpu->layers[0].backward = backward_connected_layer;
    net_cpu->layers[2].forward = forward_connected_layer;
    net_cpu->layers[2].backward = backward_connected_layer;
    printf("finished making MLP network.\n");

    float avg_loss = -1;
    float avg_loss_cpu = -1;
    data train;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0) {
        printf("started loading data.\n");
        train = load_iris_data(1);
        printf("finished loading data.\n");

        printf("started training.\n");
        float loss = train_network(net, train);
        float loss_cpu = train_network(net_cpu, train);
        printf("finished training.\n");

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("Accelerator\n");
        printf("%ld, %.3f: %f, %f avg, %f rate, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), *net->seen);
        print_layers(net);

        if(avg_loss_cpu == -1) avg_loss_cpu = loss_cpu;
        avg_loss_cpu = avg_loss_cpu*.9 + loss_cpu*.1;
        printf("CPU\n");
        printf("%ld, %.3f: %f, %f avg, %f rate, %ld images\n", get_current_batch(net_cpu), (float)(*net_cpu->seen)/N, loss_cpu, avg_loss_cpu, get_current_rate(net_cpu), *net_cpu->seen);
        print_layers(net_cpu);
    }

    // free_list(sections);
    // free_network(net);
}



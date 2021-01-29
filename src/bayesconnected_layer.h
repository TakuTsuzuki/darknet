#ifndef BAYESCONNECTED_LAYER_H
#define BAYESCONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#include "distribution.h"

layer make_bayesconnected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void forward_bayesconnected_layer(layer l, network net);
void backward_bayesconnected_layer(layer l, network net);
void update_bayesconnected_layer(layer l, update_args a);
void sampling_bayesconnected_layer(layer l, bool sampling);

#endif


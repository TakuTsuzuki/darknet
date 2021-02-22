#include "../src/mlp.h"
#include <stdio.h>

int main(int argc, char **argv) {
    printf("start forward sample.\n");
    test_mlp_hidden_layer_forward();
    return 0;
}
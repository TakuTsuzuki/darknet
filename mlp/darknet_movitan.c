#include "darknet_movitan.h"
// #include "common_lib/dataset.h"
// #include "common_lib/functional.h"
// #include "common_lib/utils.h"
#include "include/movitan.h"
#include "include/movitan_utils.h"
#include "include/movitan_params.h"

#include "darknet_movitan.h"

void darknet_movitan_hidden_forward(layer _l, float *x) {
    printf("start forward.\n");
    layer *l = &_l;
    // initializaiton
    //init_array(l->u, l->batch_size, l->n_out);
    //init_array(l->y, l->batch_size, l->n_out);
    const uint32_t col_num = l->inputs;
    const uint32_t row_num = l->outputs;
    // const uint32_t batch_size = l->batch;
    const uint32_t batch_size = 1;

    const uint32_t vec_col_size_full    = col_num/DIM;
    const uint32_t vec_col_size_partial = col_num%DIM;
    const uint32_t vec_col_size = vec_col_size_full + (vec_col_size_partial != 0 ? 1 : 0);

    const uint32_t vec_row_size_full    = row_num/DIM;
    const uint32_t vec_row_size_partial = row_num%DIM;
    const uint32_t vec_row_size = vec_row_size_full + (vec_row_size_partial != 0 ? 1 : 0);

    if( batch_size > 1){
        printf("not support yet minibatch");
        exit(1);
    }
   
    printf("vec_col: full=%d, partial=%d, size=%d\n", vec_col_size_full, vec_col_size_partial, vec_col_size);
    printf("vec_row: full=%d, partial=%d, size=%d\n", vec_row_size_full, vec_row_size_partial, vec_row_size);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // u = Wx+b
        // load vector A
        movitan_load_vector(x, A_SP_ADDR_START, vec_col_size_full, vec_col_size_partial);
        printf("step 1");

        // load vector D
        movitan_load_vector(l->biases, D_SP_ADDR_START, vec_row_size_full, vec_row_size_partial);
        printf("step 2");
        // load matrix B
        movitan_load_matrix_blocked(l->weights, B_SP_ADDR_START, col_num,
                                    vec_col_size_full, vec_col_size_partial, vec_col_size,
                                    vec_row_size_full, vec_row_size_partial, vec_row_size);
        printf("step 3");

        movitan_vec_col_mat_col_mul(A_SP_ADDR_START, B_SP_ADDR_START, C_SP_ADDR_START,
                                    vec_col_size_full, vec_col_size_partial, vec_col_size,
                                    vec_row_size_full, vec_row_size_partial, vec_row_size);
        printf("step 4");

        // add bias
        movitan_fadd(C_SP_ADDR_START, D_SP_ADDR_START, C_SP_ADDR_START,vec_row_size);

        //printf("Relu Start\n");
        // do relu
        movitan_relu(C_SP_ADDR_START, D_SP_ADDR_START, vec_row_size);

        // store u & y
        movitan_store_vector(l->delta, C_SP_ADDR_START, vec_row_size_full, vec_row_size_partial);
        movitan_store_vector(l->output, D_SP_ADDR_START, vec_row_size_full, vec_row_size_partial);
    }
}

/*
float train_network_datum_movitan() {

}

float train_network_movitan() {
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum_movitan(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}
 */

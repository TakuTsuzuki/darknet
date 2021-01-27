#include "bayesconnected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_bayesconnected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam) 
{ 
    int i; 
    layer l = {0}; 
    l.learning_rate_scale = 1; 
    l.type = BAYESCONNECTED; 
    
    l.inputs = inputs; 
    l.outputs = outputs; // outputs は出力層の大きさ 
    l.batch=batch; 
    l.batch_normalize = batch_normalize; 
    l.h = 1; 
    l.w = 1; 
    l.c = inputs; 
    l.out_h = 1; 
    l.out_w = 1; 
    l.out_c = outputs; 

    l.output = calloc(batch*outputs, sizeof(float)); // output は出力層の配列 
    l.delta = calloc(batch*outputs, sizeof(float)); 
    l.weight_updates = calloc(inputs*outputs, sizeof(float)); 
    l.bias_updates = calloc(outputs, sizeof(float)); 
    l.weights = calloc(outputs*inputs, sizeof(float)); 
    l.biases = calloc(outputs, sizeof(float)); 

    l.forward = forward_bayesconnected_layer; 
    l.backward = backward_bayesconnected_layer; 
    l.update = update_bayesconnected_layer; 
    l.sampling = sampling_bayesconnected_layer; 

    // added for BBB 
    l.weights_mu = calloc(outputs*inputs, sizeof(float)); //重みガウス分布の平均  
    l.weights_rho = calloc(outputs*inputs, sizeof(float)); //重みガウス分布の共分散  
    l.biases_mu = calloc(outputs, sizeof(float)); //バイアス項の平均  
    l.biases_rho = calloc(outputs, sizeof(float)); //バイアス項の共分散  
    l.weight_updates_mu = calloc(inputs*outputs, sizeof(float)); //重みガウス分布の平均の勾配  
    l.weight_updates_rho = calloc(inputs*outputs, sizeof(float)); //重みガウス分布の共分散の勾配  
    l.bias_updates_mu = calloc(outputs, sizeof(float)); //バイアス項の平均の勾配  
    l.bias_updates_rho = calloc(outputs, sizeof(float)); //バイアス項の共分散の勾配  
    l.weight_prior_updates = calloc(outputs*inputs, sizeof(float)); //重みガウス分布の事前分布の勾配  
    l.bias_prior_updates = calloc(outputs, sizeof(float)); //バイアス項の事前分布の勾配  
    l.weights_eps = calloc(outputs*inputs, sizeof(float)); // for using sampling 
    l.biases_eps = calloc(outputs, sizeof(float)); // for using sampling 
    // end BBB 

    // initialization 
    float scale = sqrt(2./inputs); 

    init_arrayuniform(l.weights, outputs*inputs, scale); 
    init_arrayzero(l.biases, outputs); 

    //initialization ADDED for BBB 
    // init random uniform  
    init_arrayuniform(l.weights_mu, outputs*inputs, scale); 
    init_arrayuniform(l.weights_rho, outputs*inputs, scale); 
    init_arrayuniform(l.biases_mu, outputs, scale); 
    init_arrayuniform(l.biases_rho, outputs, scale); 

    // init zero 
    // y(activationかける前のarray), grad_x に対応する配列は存在しない 
    init_arrayzero(l.output, batch*outputs); 
    init_arrayzero(l.weights_eps, outputs*inputs); 
    init_arrayzero(l.weight_updates, outputs*inputs); 
    init_arrayzero(l.weight_updates_mu, outputs*inputs); 
    init_arrayzero(l.weight_updates_rho, outputs*inputs); 
    init_arrayzero(l.weight_prior_updates,outputs*inputs); 
    init_arrayzero(l.biases_eps, outputs); 
    init_arrayzero(l.bias_updates, outputs); 
    init_arrayzero(l.bias_updates_mu, outputs); 
    init_arrayzero(l.bias_updates_rho, outputs); 
    init_arrayzero(l.bias_prior_updates, outputs); 

    l.log_prior = 0; 
    l.log_variational_posterior = 0; 

    init_scale_mixture_gaussian( 
    &(l.weights_prior),  
    0.0, expf(0.0), // mu1, sigma1 
    0.0, expf(-6.0), // mu2, sigma2 
    0.5 // alpha 
    );  

    init_scale_mixture_gaussian( 
    &(l.biases_prior),  
    0.0, expf(0.0), // mu1, sigma1 
    0.0, expf(-6.0), // mu2, sigma2 
    0.5 // alpha 
    );  

    /*
    // Check the initialization 
    assert(fabs(l.weights_prior.sigma1 - expf(0.0)) < EPS); 
    assert(fabs(l.weights_prior.sigma2 - expf(-6.0)) < EPS); 
    assert(fabs(l.biases_prior.sigma1 - expf(0.0)) < EPS); 
    assert(fabs(l.biases_prior.sigma2 - expf(-6.0)) < EPS); 
    // end BBB
    */ 

    if(adam){ 
        l.m = calloc(l.inputs*l.outputs, sizeof(float)); 
        l.v = calloc(l.inputs*l.outputs, sizeof(float)); 
        l.bias_m = calloc(l.outputs, sizeof(float)); 
        l.scale_m = calloc(l.outputs, sizeof(float)); 
        l.bias_v = calloc(l.outputs, sizeof(float)); 
        l.scale_v = calloc(l.outputs, sizeof(float)); 
    } 

    if(batch_normalize){ 
        l.scales = calloc(outputs, sizeof(float)); 
        l.scale_updates = calloc(outputs, sizeof(float)); 
        for(i = 0; i < outputs; ++i){ 
            l.scales[i] = 1; 
        } 
        l.mean = calloc(outputs, sizeof(float)); 
        l.mean_delta = calloc(outputs, sizeof(float)); 
        l.variance = calloc(outputs, sizeof(float)); 
        l.variance_delta = calloc(outputs, sizeof(float)); 
        l.rolling_mean = calloc(outputs, sizeof(float)); 
        l.rolling_variance = calloc(outputs, sizeof(float)); 
        l.x = calloc(batch*outputs, sizeof(float)); 
        l.x_norm = calloc(batch*outputs, sizeof(float)); 
    } 

    l.activation = activation; 
    fprintf(stderr, "bayesconnected %4d -> %4d\n", inputs, outputs); 
    return l; 
} 

//ADDED for BBB 
void sampling_bayesconnected_layer(layer l, bool sampling) 
{ 
    float eps; 
    if (sampling == true) { 
        for (int i = 0; i < l.inputs * l.outputs; i++) { 
            eps = rand_normal(); 
            l.weights[i] = l.weights_mu[i] + log1pf(expf(l.weights_rho[i])) * eps; 
            l.weights_eps[i] = eps; 
        } 
        for (int i = 0; i < l.outputs; i++) { 
            eps = rand_normal(); 
            l.biases[i] = l.biases_mu[i] + log1pf(expf(l.biases_rho[i])) * eps; 
            l.biases_eps[i] = eps; 
        } 
    } else { 
        for (int i = 0; i < l.inputs * l.outputs; i++) { 
            l.weights[i] = l.weights_mu[i]; 
        } 
        for (int i = 0; i < l.outputs; i++) { 
            l.biases[i] = l.biases_mu[i]; 
        } 
    }
    // Calculate log_prior and variational_log_posteior
    l.log_prior = 0;
    l.log_variational_posterior = 0;
    for (int i = 0; i < l.inputs * l.outputs; i++) {
        l.log_prior += compute_log_prob_scale_mixture_gaussian(
            &(l.weights_prior), l.weights[i]);
        l.log_variational_posterior += _compute_log_prob_gaussian(
            l.weights_mu[i], log1pf(expf(l.weights_rho[i])), l.weights[i]);
    }

    for (int i = 0; i < l.outputs; i++) {
        l.log_prior += compute_log_prob_scale_mixture_gaussian(
            &(l.biases_prior), l.biases[i]);
        l.log_variational_posterior += _compute_log_prob_gaussian(
            l.biases_mu[i], log1pf(expf(l.biases_rho[i])), l.biases[i]);
    }    
}

void update_bayesconnected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    // self.b -= l * self.grad_b
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    // self.grad_b = momentum * self.grad_b
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_bayesconnected_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_bayesconnected_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    
    _accumulate_grad(l);
}


void denormalize_bayesconnected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights_mu[i*l.inputs + j] *= scale;
            l.weights_rho[i*l.inputs + j] *= scale;
        }
        l.biases_mu[i] -= l.rolling_mean[i] * scale;
        l.biases_rho[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_bayesconnected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases_mu ");
    print_statistics(l.biases_mu, l.outputs);
    printf("Weights_mu ");
    print_statistics(l.weights_mu, l.outputs);
}

void _accumulate_grad(layer l) {

    float pi = M_PI;
    float m1 = l.weights_prior.mu1;
    float m2 = l.weights_prior.mu2;
    float s1 = l.weights_prior.sigma1;
    float s2 = l.weights_prior.sigma2;
    float a  = l.weights_prior.alpha;
    
    for(int out_idx = 0; out_idx < l.outputs; out_idx++) {           
        for(int in_idx = 0; in_idx < l.inputs; in_idx++) {
            int w_idx = out_idx * l.inputs + in_idx;
            
            float w  = l.weights[w_idx];
            float m  = l.weights_mu[w_idx];
            float r  = l.weights_rho[w_idx];
            float s  = movitan_log1p(movitan_exp(r));
            float gw = l.weight_updates[w_idx];
            float e  = l.weights_eps[w_idx];

            // (3) equation in the paper
            float g1     = expf(-powf(w,2)/(2.0*s1*s1));
            float g2     = expf(-powf(w,2)/(2.0*s2*s2));

            float dprior_dw = -w*((a/powf(s1,3))*g1+((1.0-a)/powf(s2,3))*g2)/((a/s1)*g1+((1.0-a)/s2)*g2)/l.batch;
        
            float dpos_dw   = -e/s/l.batch;
            float dpos_dmu  = e/s/l.batch; 
            float dpos_ds   = (powf(e,2)-1)/s/l.batch;

            float dloss_dw  = (dpos_dw - dprior_dw) + gw;
            float dloss_dmu = dloss_dw + dpos_dmu;
            float dloss_ds  = dloss_dw*e + dpos_ds;  

            // (4) equation in the paper
            float ds_dr  = 1 / (1 + expf(-r));
            float dloss_drho = dloss_ds * ds_dr;

            // accumulate each gradient for sampling
            l.weight_updates_mu[w_idx]    += dloss_dmu;
            l.weight_updates_rho[w_idx]   += dloss_drho;
            l.weight_prior_updates[w_idx] += dprior_dw;
        }
    }

    for(int b_idx = 0; b_idx < l.outputs; b_idx++) {
        float b  = l.biases[b_idx];
        float m  = l.biases_mu[b_idx];
        float r  = l.biases_rho[b_idx];
        float s  = log1pf(expf(r));
        float gb = l.bias_updates[b_idx];
        float e  = l.biases_eps[b_idx];

        // (3) equation in the paper
        float g1     = expf(-powf(b,2)/(2.0*s1*s1));
        float g2     = expf(-powf(b,2)/(2.0*s2*s2));
        
        float dprior_db = -b*(a/powf(s1,3)*g1+(1-a)/powf(s2,3)*g2)/(a/s1*g1+(1-a)/s2*g2)/l.batch;
 
        float dpos_db   = -e/s/l.batch;
        float dpos_dmu  = e/s/l.batch; 
        float dpos_ds   = (powf(e,2)-1)/s/l.batch;

        float dloss_db  = (dpos_db - dprior_db) + gb;
        float dloss_dmu = dloss_db + dpos_dmu;
        float dloss_ds  = dloss_db*e + dpos_ds;  

        // (4) equation in the paper
        float ds_dr  = 1 / (1 + expf(-r));
        float dloss_drho = dloss_ds * ds_dr;

        // accumulate each gradient for sampling
        l.bias_updates_mu[b_idx]    += dloss_dmu;
        l.bias_updates_rho[b_idx]   += dloss_drho;
        l.bias_prior_updates[b_idx] += dprior_db;
    }
}

#include "distribution.h" 

void init_scale_mixture_gaussian(ScaleMixtureGaussian *smg, float mu1, float sigma1, float mu2, float sigma2,float alpha) 
{ 
    smg->mu1 = mu1; 
    smg->sigma1 = sigma1; 
    smg->mu2 = mu2; 
    smg->sigma2 = sigma2; 
    smg->alpha = alpha; 
} 

float _compute_prob_gaussian(float mu, float sigma, float x) 
{ 
    if (!(sigma >= 0)) { 
    printf("%f\n", sigma); 
    } 
    // assert(sigma >= 0); 
    return expf(-powf(x - mu, 2) / (2 * powf(sigma, 2))) / (sqrtf(2 * M_PI) * sigma); 
} 

 

float _compute_log_prob_gaussian(float mu, float sigma, float x) 
{ 
    return logf(_compute_prob_gaussian(mu, sigma, x)); 
} 

 

float compute_log_prob_gaussian(Gaussian *dist, float x) 
{ 
    float sigma = log1p(expf(dist->rho)); 
    return _compute_log_prob_gaussian(dist->mu, sigma, x); 
} 

 

float compute_log_prob_scale_mixture_gaussian(ScaleMixtureGaussian *dist, float x) 
{ 
    float prob1 = _compute_prob_gaussian(dist->mu1, dist->sigma1, x); 
    float prob2 = _compute_prob_gaussian(dist->mu2, dist->sigma2, x); 
    return logf(dist->alpha * prob1 + (1.0 - dist->alpha) * prob2); 
}
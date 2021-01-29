#include "darknet.h" 

void init_scale_mixture_gaussian(ScaleMixtureGaussian *smg, float mu1, float sigma1, float mu2, float sigma2, float alpha); 
float compute_log_prob_gaussian(Gaussian *dist, float x); 
float compute_log_prob_scale_mixture_gaussian(ScaleMixtureGaussian *dist, float x); 

//おかしいので直す
float _compute_log_prob_gaussian(float mu, float sigma, float x) ;
float _compute_prob_gaussian(float mu, float sigma, float x) ; 
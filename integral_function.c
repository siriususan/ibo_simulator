#include <stddef.h>
#include <tgmath.h>
#include <math.h>

const double INV_SQRT_2 = 0.7071067811865475;		// 1/sqrt(2)
const double INV_SQRT_2_PI = 0.3989422804014327;	// 1/sqrt(2*pi)

int i;		//target location
int size;	//size of arrays
double *dp;		//dprime
double *pp;		//posterior probabilities

void set(int size_, void *dp_, void *pp_)
{
	size = size_;
	dp = (double *)dp_;
	pp = (double *)pp_;
}

void set_target(int target_)
{
	i = target_;
}

double pdf(double x)
{
	return INV_SQRT_2_PI * exp(-x*x);
}

double cdf(double x)
{
	return 0.5 + 0.5 * erf(x * INV_SQRT_2);
}

double function(int n, double args[n])
{
	int j;
	double prod = 1;
	for (j = 0; j < size; ++j) {
		if (j == i)
			continue;
		prod *= cdf(
			(
				-2.0*log(pp[j]/pp[i]) +
				dp[j]*dp[j] + 2.0*dp[i]*args[0] +
				dp[i]*dp[i]
			) / (
				2.0*dp[j]
			)
		);
	}
	return pdf(args[0]) * prod;
}

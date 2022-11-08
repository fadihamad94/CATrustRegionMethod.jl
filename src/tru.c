/* tru.c */
/* TRU C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_tru.h"

// Custom userdata struct
struct userdata_type_tru {
	int n;
	double (*eval_f)(const double x[]);
	double * (*eval_g)(const double x[]);
	double * (*eval_h)(const double x[]);
	int status;
	int iter;
	int total_function_evaluation;
	int total_gradient_evaluation;
	int total_hessian_evaluation;
	double* solution;
};

// Function prototypes
int fun( int n, const double x[], double *f, const void * );
int grad( int n, const double x[], double g[], const void * );
int hess( int n, int ne, const double x[], double hval[], const void * );

struct userdata_type_tru tru(double x[], double g[], struct userdata_type_tru userdata, int print_level, int maxit, double initial_radius, bool subproblem_direct){
	// Derived types
	void *data;
	struct tru_control_type control;
	struct tru_inform_type inform;

	int status;

	// Initialize TRU
	tru_initialize( &data, &control, &status );
	control.f_indexing = false; // C sparse matrix indexing
	control.print_level = print_level;
	control.maxit = maxit;
	control.norm=-1;
	control.start_print=0;
	control.stop_print=maxit;
	control.initial_radius = initial_radius;
	control.subproblem_direct = subproblem_direct;

	tru_import( &control, &data, &status, userdata.n, "dense",
		userdata.n*(userdata.n+1)/2, NULL, NULL, NULL );
	tru_solve_with_mat( &data, &userdata, &status,
		userdata.n, x, g, userdata.n*(userdata.n+1)/2, fun, grad, hess, NULL );
	tru_information( &data, &inform, &status);

	userdata.status = inform.status;
	userdata.iter = inform.iter;
	userdata.total_function_evaluation = inform.f_eval;
	userdata.total_gradient_evaluation = inform.g_eval;
	userdata.total_hessian_evaluation = inform.h_eval;
	userdata.solution = x;

	// Delete internal workspace
	tru_terminate( &data, &control, &inform );
	return userdata;
}

// Objective function
int fun( int n, const double x[], double *f, const void *userdata ){
	struct userdata_type_tru *myuserdata = (struct userdata_type_tru *) userdata;
  *f = myuserdata->eval_f(x);
  return 0;
}

// Gradient of the objective
int grad( int n, const double x[], double g[], const void *userdata ){
	struct userdata_type_tru *myuserdata = (struct userdata_type_tru *) userdata;
  double* temp_g = myuserdata->eval_g(x);
  for (int i = 0; i < n; i++)
	{
		g[i] = temp_g[i];
  }
  return 0;
}

// Hessian of the objective
int hess( int n, int ne, const double x[], double hval[], const void *userdata ){
	struct userdata_type_tru *myuserdata = (struct userdata_type_tru *) userdata;
  double* temp_hval = myuserdata->eval_h(x);
  for(int i =0; i < ne; i++)
	{
		hval[i] = temp_hval[i];
  }
  return 0;
}

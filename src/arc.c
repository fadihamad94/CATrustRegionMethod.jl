/* arct.c */
/* Full test for the ARC C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_arc.h"

// Custom userdata struct
struct userdata_type_arc {
    int n;
    double (*eval_f)(const double x[]);
    double * (*eval_g)(const double x[]);
    double * (*eval_h)(const double x[]);
    int status;
    int iter;
    int total_function_evaluation;
    int total_gradient_evaluation;
    int total_hessian_evaluation;
    int total_inner_iterations_or_factorizations;
    double * solution;
};

// Function prototypes
int fun( int n, const double x[], double *f, const void * );
int grad( int n, const double x[], double g[], const void * );
int hess( int n, int ne, const double x[], double hval[], const void * );

struct userdata_type_arc arc(double x[], double g[], struct userdata_type_arc userdata, int print_level, int maxit, double initial_weight, double stop_g_absolute, double stop_g_relative, double stop_s,  double eta_too_successful, double eta_1, double eta_2, bool subproblem_direct, int max_inner_iterations_or_factorizations, double clock_time_limit){
	// Derived types
	void *data;
  struct arc_control_type control;
  struct arc_inform_type inform;
	int status;
	int eval_status;
  double u[userdata.n], v[userdata.n];
	// Initialize ARC
	arc_initialize( &data, &control, &status );
	control.f_indexing = false; // C sparse matrix indexing
	control.print_level = print_level;
	control.maxit = maxit;
	control.norm=-1;
	control.start_print=0;
	control.stop_print=maxit;
	control.subproblem_direct=subproblem_direct;
	control.initial_weight = initial_weight;
	control.eta_too_successful = eta_too_successful;
	control.eta_very_successful = eta_2;
	control.eta_successful = eta_1;
	control.clock_time_limit = clock_time_limit;
	control.stop_g_absolute = stop_g_absolute;
	control.stop_g_relative = stop_g_relative;
	control.stop_s = stop_s;
	if(subproblem_direct){
		int max_factorizations = max_inner_iterations_or_factorizations;
		control.rqs_control.max_factorizations = max_factorizations;
	}else{
		int max_iterations = max_inner_iterations_or_factorizations;
		control.glrt_control.itmax = max_iterations;
	}
	arc_import( &control, &data, &status, userdata.n, "dense",
			   userdata.n*(userdata.n+1)/2, NULL, NULL, NULL );

	arc_solve_with_mat( &data, &userdata, &status,
		   						userdata.n, x, g, userdata.n*(userdata.n+1)/2, fun, grad, hess, NULL );

	arc_information( &data, &inform, &status);

	userdata.status = inform.status;
	userdata.iter = inform.iter;
	userdata.total_function_evaluation = inform.f_eval;
	userdata.total_gradient_evaluation = inform.g_eval;
	userdata.total_hessian_evaluation = inform.h_eval;
	if(subproblem_direct){
		userdata.total_inner_iterations_or_factorizations = inform.rqs_inform.factorizations;
	}else{
		userdata.total_inner_iterations_or_factorizations = inform.glrt_inform.iter;
	}
	userdata.solution = x;
	// Delete internal workspace
	arc_terminate( &data, &control, &inform );
	return userdata;
}

// Objective function
int fun( int n, const double x[], double *f, const void *userdata ){
    struct userdata_type_arc *myuserdata = (struct userdata_type_arc *) userdata;
    *f = myuserdata->eval_f(x);
    return 0;
}

// Gradient of the objective
int grad( int n, const double x[], double g[], const void *userdata ){
    struct userdata_type_arc *myuserdata = (struct userdata_type_arc *) userdata;
    double* temp_g = myuserdata->eval_g(x);

    for (int i = 0; i < n; i++)
    {
      g[i] = temp_g[i];
    }
    return 0;
}

// Hessian of the objective
int hess( int n, int ne, const double x[], double hval[],
         const void *userdata ){
    struct userdata_type_arc *myuserdata = (struct userdata_type_arc *) userdata;
    double* temp_hval = myuserdata->eval_h(x);
    for(int i =0; i < ne; i++)
    {
        hval[i] = temp_hval[i];
    }
    return 0;
}

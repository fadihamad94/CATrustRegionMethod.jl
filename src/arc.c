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
    double time_clock_factorize;
    double time_clock_preprocess;
    double time_clock_solve;
    double time_clock_analyse;

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
	//printf("Initialize ARC\n");
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
	//printf("%f: \n", control.stop_g_absolute);
	//printf("%e: \n", control.stop_g_relative);
	//printf("%e: \n", control.stop_s);
	control.stop_g_absolute = stop_g_absolute;
	control.stop_g_relative = stop_g_relative;
	control.stop_s = stop_s;
	//printf("%f: \n", control.stop_g_absolute);
        //printf("%e: \n", control.stop_g_relative);
        //printf("%e: \n", control.stop_s);
	//printf("Start print:\n");
	//printf("%d:\n", control.start_print);
	//printf("Stop print:\n");
        //printf("%d:\n", control.stop_print);
	//printf("Importing ARC\n");
	if(subproblem_direct){
		int max_factorizations = max_inner_iterations_or_factorizations;
		control.rqs_control.max_factorizations = max_factorizations;
	}else{
		int max_iterations = max_inner_iterations_or_factorizations;
                control.glrt_control.itmax = max_iterations;
	}
	arc_import( &control, &data, &status, userdata.n, "dense",
			   userdata.n*(userdata.n+1)/2, NULL, NULL, NULL );
	//printf("Solving ARC\n");
	arc_solve_with_mat( &data, &userdata, &status,
		   						userdata.n, x, g, userdata.n*(userdata.n+1)/2, fun, grad, hess, NULL );
	/*
	double H_dense[userdata.n*(userdata.n+1)/2];
        double f = 0.0;
        fun(userdata.n, x, &f, &userdata);
	printf("Function:\n");
        printf("%f\n", f);
        hess(userdata.n, userdata.n*(userdata.n+1)/2, x, H_dense,
                           &userdata);
	printf("Hessian:\n");
        printf("%f\n", H_dense[0]);
        int iteration;
	for(iteration = 1; iteration <= 100; iteration = iteration + 1){
	printf("\n------------------------------ITERATION---------------------------------\n");
	printf("%d\n", iteration);
	arc_solve_reverse_with_mat(&data, &status, &eval_status,
                                       userdata.n, x, f, g, userdata.n*(userdata.n+1)/2, H_dense, u, v );
	printf("Status:\n");
	printf("%d\n", status);
	
	if(status == 0){ // successful termination
	     printf("successful termination\n"); 
             break;
        }else if(status < 0){ // error exit
	     printf(" error exit\n");
             break;
        }else if(status == 2){ // evaluate f
	     printf("evaluate f\n");
             eval_status = fun(userdata.n, x, &f, &userdata);
        }else if(status == 3){ // evaluate g
             printf("evaluate g\n");
             eval_status = grad(userdata.n, x, g, &userdata);
        }else if(status == 4){ // evaluate H
	     printf("evaluate H\n");
             eval_status = hess(userdata.n, userdata.n*(userdata.n+1)/2, x, H_dense, 
                                                 &userdata); 
        }else{
             printf(" the value %1i of status should not occur\n", 
                          status);
             break;
        }
	}
	*/
	//printf("Information ARC\n");
	arc_information( &data, &inform, &status);

        userdata.status = inform.status;
	userdata.iter = inform.iter;
	userdata.total_function_evaluation = inform.f_eval;
	userdata.total_gradient_evaluation = inform.g_eval;
	userdata.total_hessian_evaluation = inform.h_eval;
	if(subproblem_direct){
		userdata.total_inner_iterations_or_factorizations = inform.rqs_inform.factorizations;
	}else{
		userdata.total_inner_iterations_or_factorizations = inform.glrt_inform.iter + inform.glrt_inform.iter_pass2;
	}
	userdata.solution = x;
	userdata.time_clock_factorize = inform.rqs_inform.time.clock_factorize;
	userdata.time_clock_preprocess = inform.rqs_inform.time.clock_assemble;
	userdata.time_clock_solve = inform.rqs_inform.time.clock_solve;
        userdata.time_clock_analyse = inform.rqs_inform.time.clock_analyse;
	// Delete internal workspace
	//printf("Termination ARC\n");
	arc_terminate( &data, &control, &inform );
	return userdata;
}

// Objective function
int fun( int n, const double x[], double *f, const void *userdata ){
    struct userdata_type_arc *myuserdata = (struct userdata_type_arc *) userdata;
    *f = myuserdata->eval_f(x);
    //printf("%f\n", *f);
    //printf("$f\n", f);
    return 0;
}

// Gradient of the objective
int grad( int n, const double x[], double g[], const void *userdata ){
    //printf("Computing Gradient\n");
    struct userdata_type_arc *myuserdata = (struct userdata_type_arc *) userdata;
    //g = myuserdata->eval_g(x);
    double* temp_g = myuserdata->eval_g(x);
   
    for (int i = 0; i < n; i++)
    {
      g[i] = temp_g[i];
    }
    
    //g[0] = 2.0 * ( x[0] + x[2] + 4.0 ) - sin(x[0]);
    //g[1] = 2.0 * ( x[1] + x[2] );
    //g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] );
    //printf("%f\n", g[0]);
    //printf("%f\n", g[1]);
    //printf("%f\n", g[2]);
    return 0;
}

// Hessian of the objective
int hess( int n, int ne, const double x[], double hval[],
         const void *userdata ){
    //printf("Computing Hessian\n");
    struct userdata_type_arc *myuserdata = (struct userdata_type_arc *) userdata;
    double* temp_hval = myuserdata->eval_h(x);
    //hval = myuserdata->eval_h(x);
    for(int i =0; i < ne; i++)
    {
        hval[i] = temp_hval[i];
    }
    //printf("%f\n", hval[0]);
    //printf("%f\n", hval[1]);
    //printf("%f\n", hval[2]);
    return 0;
}


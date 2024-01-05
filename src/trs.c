/* trs.c */
/* TRS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_trs.h"

#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

//struct history_type{
//	double lambda;
//	double x_norm;
//};

// Custom userdata struct
struct userdata_type_trs {
	int status;
	int factorizations;
	bool hard_case;
	double multiplier;
	//double obj;
	//double lambda;
	//double lambda_0;
	//double lambda_100;
	//double x_norm;
        //double x_norm_100;
        //double x_norm_len_history;
	//int len_history;
	//struct history_type history[100];
	//double* solution;
	//double time_clock_factorize;
    	//double time_clock_preprocess;
    	//double time_clock_solve;
    	//double time_clock_analyse;

};

struct userdata_type_trs trs(int n, double f, double x[], double g[], double H_dense[], double radius, int print_level, int max_factorizations, bool use_initial_multiplier, double initial_multiplier, bool use_stop_args, double stop_normal, double stop_hard, const char* outputFilePath){
	// Derived types

	void *data;
	struct trs_control_type control;
	struct trs_inform_type inform;
	
	int status;
	int H_dense_ne = n*(n+1)/2;

	// Initialize TRS
	trs_initialize( &data, &control, &status );
	control.f_indexing = false; // C sparse matrix indexing
	control.print_level = print_level;
	control.max_factorizations = max_factorizations;

	if(use_initial_multiplier)
	{
		control.initial_multiplier = initial_multiplier;
	}

	if(use_stop_args)
	{
		control.stop_normal = stop_normal;
		control.stop_absolute_normal = stop_normal * radius;
		control.stop_hard = stop_hard;
	}
  
	trs_import( &control, &data, &status, n, "dense", 0, NULL, NULL, NULL );

	// solve the problem
	trs_solve_problem( &data, &status, n, radius, f, g, H_dense_ne, H_dense, x, 0, NULL, 0, 0, NULL, NULL );
	trs_information( &data, &inform, &status );
  
	struct userdata_type_trs userdata;
	userdata.status = inform.status;
	userdata.factorizations = inform.factorizations;
	//userdata.obj = inform.obj;
	
	//userdata.solution = x;
	userdata.hard_case = inform.hard_case;
	userdata.multiplier = inform.multiplier;
	//userdata.lambda = inform.history[inform.len_history - 1].lambda;
	//userdata.lambda_0 = inform.history[0].lambda;
	//userdata.lambda_100 = inform.history[99].lambda;
	//userdata.x_norm = inform.x_norm;
	//userdata.x_norm_100 = inform.history[99].x_norm;
	//userdata.x_norm_len_history = inform.history[inform.len_history].x_norm;

	if(print_level >= 0){
		FILE *file = fopen(outputFilePath, "a");  // Open file in append mode
		if (file != NULL) {
			 fprintf(file, "%d,%d,%f,%f,%f,%f,%d,%d\n", inform.status,inform.hard_case ? 1 : 0,inform.x_norm,radius,inform.multiplier,inform.history[inform.len_history].lambda,inform.len_history,inform.factorizations);
			 fclose(file);
		}

	}

	//userdata.len_history = inform.len_history;
	//struct history_type history[100];
	//for(int i =0; i < 100; i++){
	//	struct trs_history_type temp_1;
	//      temp_1.lambda = inform.history[i].lambda;
	//	temp_1.x_norm = inform.history[i].x_norm;
	//	struct history_type temp_2;
	//	temp_2.lambda = temp_1.lambda;
	//	temp_2.x_norm = temp_1.x_norm;
	//	history[i] = temp_2;
	//}
	
	//userdata.time_clock_factorize = inform.time.clock_factorize;
	//userdata.time_clock_preprocess = inform.time.clock_assemble;
	//userdata.time_clock_solve = inform.time.clock_solve;
        //userdata.time_clock_analyse = inform.time.clock_analyse;

	// Delete internal workspace
	trs_terminate( &data, &control, &inform );
	return userdata;
}

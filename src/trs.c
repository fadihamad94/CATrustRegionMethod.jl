/* trs.c */
/* TRS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_trs.h"

#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

// Custom userdata struct
struct userdata_type_trs {
	int status;
	int factorizations;
	double obj;
	bool hard_case;
	double multiplier;
	double x_norm;
	double* solution;
};

struct userdata_type_trs trs(int n, double f, double x[], double g[], double H_dense[], double radius, int print_level, int max_factorizations, bool use_initial_multiplier, double initial_multiplier){
  // Derived types
  void *data;
  struct trs_control_type control;
  struct trs_inform_type inform;

  int status;
  int H_dense_ne = n*(n+1)/2;

	// Initialize TRS
	//printf("!---------Initializing--------------!\n");
	trs_initialize( &data, &control, &status );
	control.f_indexing = false; // C sparse matrix indexing
	control.print_level = print_level;
	control.max_factorizations = max_factorizations;

	if(use_initial_multiplier)
	{
		control.initial_multiplier = initial_multiplier;
	}
  // import the control parameters and structural data
  //printf("--------------Importing--------------\n");
  trs_import( &control, &data, &status, n, "dense", 0, NULL, NULL, NULL );
  // solve the problem
  //printf("----------------Solving--------------\n");
  trs_solve_problem( &data, &status, n, radius, f, g, H_dense_ne, H_dense, x, 0, NULL, 0, 0, NULL, NULL );
  //printf("--------------Information------------\n");
  trs_information( &data, &inform, &status );
  //printf("+++++++++++++++++++Done+++++++++++\n");
  //printf("status: %d \n", inform.status);
  //printf("iter: %d \n", inform.factorizations);
  //printf("obj: %f \n", inform.obj);
  //printf("multiplier: %f \n", inform.multiplier);
  //printf("x_norm: %f \n", inform.x_norm);
  //printf("x[0]: %f \n", x[0]);
  //printf("x[1]: %f \n", x[1]);
  //printf("+++++++++++++++++++Done+++++++++++\n");
	struct userdata_type_trs userdata;
	userdata.status = inform.status;
	userdata.factorizations = inform.factorizations;
	userdata.obj = inform.obj;
	//printf("===============REACHED HERE==================\n");
	//for(int i = 0; i < n; i++) printf("%f: ", x[i]);
	//printf("===============REACHED HERE==================\n");
	userdata.solution = x;
	//printf("===============REACHED HERE==================\n");
	userdata.hard_case = inform.hard_case;
	//printf("===============REACHED HERE==================\n");
	userdata.multiplier = inform.multiplier;
	//printf("multiplier: %f \n", userdata.multiplier);
	//printf("===============REACHED HERE==================\n");
	userdata.x_norm = inform.x_norm;
	//printf("x_norm: %f \n", userdata.x_norm);
	//printf("===============REACHED HERE==================\n");
  // Delete internal workspace
  //printf("----------Termination-------------\n");
  trs_terminate( &data, &control, &inform );
  //printf("===============DONE==================\n");
	return userdata;
}


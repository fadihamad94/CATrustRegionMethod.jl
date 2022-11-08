/* trs.c */
/* TRS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_trs.h"

// Custom userdata struct
struct userdata_type_trs {
	int status;
	int factorizations;
	double obj;
	bool hard_case;
	double multiplier;
	double x_norm;
};

struct userdata_type_trs trs(int n, double f, double x[], double g[], double H_dense[], double radius, int print_level, int max_factorizations){
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

  // import the control parameters and structural data
  trs_import( &control, &data, &status, n, "dense", 0, NULL, NULL, NULL );

	// solve the problem
  trs_solve_problem( &data, &status, n, radius, f, g, H_dense_ne, H_dense, x, 0, NULL, 0, 0, NULL, NULL );

  trs_information( &data, &inform, &status );

	struct userdata_type_trs userdata;
	userdata.status = inform.status;
	userdata.factorizations = inform.factorizations;
	userdata.obj = inform.obj;
	userdata.hard_case = inform.hard_case;
	userdata.multiplier = inform.multiplier;
	userdata.x_norm = inform.x_norm;

	// Delete internal workspace
  trs_terminate( &data, &control, &inform );
	return userdata;
}

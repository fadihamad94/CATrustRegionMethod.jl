/* gltr.c */
/* GLTR C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_gltr.h"

// Custom userdata struct
struct userdata_type_gltr {
	int status;
	int iter;
	double obj;
	bool hard_case;
	double multiplier;
	double mnormx;
};

void matrix_product(int n, double g[], double H_dense[], double vector[])
{
	double hessian[n][n];
	int count = 0;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j <= i; j++)
		{
			hessian[i][j] = H_dense[count];
			hessian[j][i] = hessian[i][j];
			count += 1;
		}
	}

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			vector[i] += hessian[i][j] * g[j];
		}
	}
}

struct userdata_type_gltr gltr(int n, double f, double x[], double g[], double H_dense[], double radius, int print_level, int max_iterations){
  // Derived types
  void *data;
  struct gltr_control_type control;
  struct gltr_inform_type inform;

  double r[n];
	double vector[n];

  int status;

  // Initialize gltr
  gltr_initialize( &data, &control, &status );
  control.f_indexing = false; // C sparse matrix indexing
  control.print_level = print_level;
  control.itmax = max_iterations;
	control.unitm = true;

	gltr_import_control( &control, &data, &status );

	status = 1;
	for( int i = 0; i < n; i++) r[i] = g[i];

	while(true){ // reverse-communication loop
		gltr_solve_problem( &data, &status, n, radius, x, r, vector );
		if ( status == 0 ) { // successful termination
				break;
		} else if ( status < 0 ) { // error exit
				break;
		} else if ( status == 3 ) { // form the Hessian-vector product
				matrix_product(n, g, H_dense, vector);
		} else if ( status == 5 ) { // restart
				for( int i = 0; i < n; i++) r[i] = g[i];
		}else{
				printf(" the value %1i of status should not occur\n", status);
				break;
		}
	}

	gltr_information( &data, &inform, &status );

	struct userdata_type_gltr userdata;
	userdata.status = inform.status;
	userdata.iter = inform.iter;
	userdata.obj = inform.obj;
	userdata.hard_case = inform.hard_case;
	userdata.multiplier = inform.multiplier;
	userdata.mnormx = inform.mnormx;

  // Delete internal workspace
  gltr_terminate( &data, &control, &inform );
	return userdata;
}

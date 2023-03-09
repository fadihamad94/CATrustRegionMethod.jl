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

void  matrix_product(int n, double H_dense[], double vector[])
{
	double hessian[n][n];
	double h_vector[n];
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
		h_vector[i] = 0;
		for(int j = 0; j < n; j++)
		{
			h_vector[i] += hessian[i][j] * vector[j];
		}
	}
	for(int i = 0; i < n; i++) vector[i] = h_vector[i];
}

struct userdata_type_gltr gltr(int n, double x[], double f, double g[], double H_dense[], double radius, int print_level, int max_iterations, double stop_relative, double stop_absolute, bool steihaug_toint){
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
  control.steihaug_toint = steihaug_toint;
  if(stop_relative >= 0) {
	  if(print_level  > 0){
	  	printf("control.stop_relative is %.9f", control.stop_relative);
	 	printf("stop_relative is %.9f", stop_relative);
	  }
	  control.stop_relative = stop_relative;
	  if(print_level  > 0){
	  	printf("control.stop_relative is %.9f", control.stop_relative);
	  }
  }
  if(stop_absolute >= 0) {
          control.stop_absolute = stop_absolute;
  }
  //control.out = 1;
  //printf("conrol.out &f", control.out);
  //printf("control.stop_relative is %.9f", control.stop_relative);
  //printf("control.stop_absolute is %.9f", control.stop_absolute);
  gltr_import_control( &control, &data, &status );
  status = 1;
  for( int i = 0; i < n; i++) r[i] = g[i];

  double initial_g[n];
  for( int i = 0; i < n; i++) initial_g[i] = g[i];
  //for( int i = 0; i < n; i++) printf("gradient value is %f.\n", g[i]);

  while(true){ // reverse-communication loop
	gltr_solve_problem( &data, &status, n, radius, x, r, vector );
	if ( status == 0 ) { // successful termination
		//printf("the value %li of status is to exit.\n", status);
		break;
	} else if ( status < 0 ) { // error exit
		break;
	} else if ( status == 3 ) { // form the Hessian-vector product
		 matrix_product(n, H_dense, vector);
	} else if ( status == 5 ) { // restart
		//printf("the value %li of status is printed for debugging.\n", status);
		for( int i = 0; i < n; i++) r[i] = initial_g[i];
		//for( int i = 0; i < n; i++) printf("gradient value is %f.\n", initial_g[i]);
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


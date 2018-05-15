/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.07
* Implemented by Rui Xia (rxiacn@gmail.com)
* Last updated on 2011-03-06.
*********************************************************************/

#include <cstdlib>
#include <iostream>
#include <string>
#include <string.h>
#include "NBEM.h"

using namespace std;


void print_help() {
	cout << "\n***** OpenPR-NBEM Un-supervised Learning Module *****\n\n"
		<< "usage: nbem [options] initial_model_file unlabeled_file model_file\n\n"
		<< "options: -h        -> help\n"
		<< "         -n int    -> Maximal iteration steps (default: 20)\n" 
		<< "         -m float  -> Minimal increase rate of loglikelihood (default: 1e-4)\n" 
		<< endl;
}

void read_parameters(int argc, char *argv[], char *init_file, char *unlabel_file, char *model_file,
						int *max_iter, double *eps_thrd) {
	// set default options
	*max_iter = 20;
	*eps_thrd = 1e-4;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 'n':
				*max_iter = atoi(argv[++i]);
				break;
			case 'm':
				*eps_thrd = atof(argv[++i]);
				break;
			default:
				cout << "Unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+2)>=argc) {
		cout << "Not enough parameters!" << endl;
		print_help();
		exit(0);
	}
	strcpy(init_file, argv[i]);
	strcpy(unlabel_file, argv[i+1]);
	strcpy(model_file, argv[i+2]);
}

int main(int argc, char *argv[])
{
	char init_file[200];
	char unlabel_file[200];
	char model_file[200];
	int max_iter;
	double eps_thrd;
	read_parameters(argc, argv, init_file, unlabel_file, model_file, &max_iter, &eps_thrd);
    NBEM nbem;
    nbem.init_em_usl(init_file, unlabel_file);
    nbem.learn_em_usl(max_iter, eps_thrd, 2);
	nbem.save_model(model_file, nbem.usamp_class_prb, nbem.usamp_feat_class_prb);
	return 1;
}

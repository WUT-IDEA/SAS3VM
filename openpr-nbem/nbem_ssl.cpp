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
	cout << "\n***** OpenPR-NBEM Semi-supervised Learning Module *****\n\n"
		<< "usage: nbem_ssl [options] labeled_file unlabeled_file model_file\n\n"
		<< "options: -h        -> help\n"
		<< "         -l float  -> The turnoff weight for unlabeled set (default 1)\n"
		<< "         -n int    -> Maximal iteration steps (default: 20)\n" 
		<< "         -m float  -> Minimal increase rate of loglikelihood (default: 1e-4)\n"
		<< endl;
}

void read_parameters(int argc, char *argv[], char *label_file, char *unlabel_file, char *model_file,
						int *max_iter, double *eps_thrd, float *lambda) {
	// set default options
	*max_iter = 20;
	*eps_thrd = 1e-4;
	*lambda = 1.0;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 'l':
				*lambda = atof(argv[++i]);
				break;
			case 'n':
				*max_iter = atoi(argv[++i]);
				break;
			case 'm':
				*eps_thrd = atof(argv[++i]);
				break;
			default:
				cout << "Error: unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+2)>=argc) {
		cout << "Error: not enough parameters!" << endl;
		print_help();
		exit(0);
	}
	strcpy(label_file, argv[i]);
	strcpy(unlabel_file, argv[i+1]);
	strcpy(model_file, argv[i+2]);
}

int main(int argc, char *argv[])
{
	char label_file[100];
	char unlabel_file[200];
	char model_file[200];
	float lambda;
	int max_iter;
	double eps_thrd;
    read_parameters(argc, argv, label_file, unlabel_file, model_file, &max_iter, &eps_thrd, &lambda);
    NBEM nbem;
    nbem.init_em_ssl(label_file, unlabel_file);
    nbem.learn_em_ssl(max_iter, eps_thrd, lambda, 2);
	nbem.save_model(model_file, nbem.comb_class_prb, nbem.comb_feat_class_prb);
	return 1;
}

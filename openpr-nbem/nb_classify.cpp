/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.07
* Implemented by Rui Xia (rxiacn@gmail.com)
* Last updated on 2011-03-06.
*********************************************************************/

#include <cstdlib>
#include <iostream>
#include <string>
#include <string.h>
#include "NB.h"

using namespace std;


void print_help() {
	cout << "\n***** OpenPR-NB Classification Module *****\n\n"
		<< "usage: nb_classify [options] test_file model_file output_file\n\n"
		<< "options: -h        -> help\n"	
		<< "         -f [0..2] -> 0: only output class label (default)\n"
		<< "                   -> 1: output class label with log-likelihood\n"
		<< "                   -> 2: output class label with probability\n"
		<< endl;
}

void read_parameters(int argc, char *argv[], char *test_file, char *model_file, 
						char *output_file, int *output_format) {
	// set default options
	*output_format = 0;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 'f':
				*output_format = atoi(argv[++i]);
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
	strcpy(test_file, argv[i]);
	strcpy(model_file, argv[i+1]);
	strcpy(output_file, argv[i+2]);
}

int main(int argc, char *argv[])
{
	char test_file[200];
	char model_file[200];
	char output_file[200];
	int output_format;
	read_parameters(argc, argv, test_file, model_file, output_file, &output_format);
    NB nb;
	nb.load_model(model_file, nb.samp_class_prb, nb.samp_feat_class_prb);
	float acc= nb.classify_test_data(test_file, output_file, 1, output_format, nb.samp_class_prb, nb.samp_feat_class_prb);
	cout << "Accuracy: " << acc << endl;
	return 1;
}

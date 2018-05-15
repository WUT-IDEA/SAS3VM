/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.07
* Implemented by Rui Xia (rxiacn@gmail.com)
* Last updated on 2011-03-06.
*********************************************************************/

#pragma once

#include <sstream>
#include "NB.h"

using namespace std;

class NBEM : public NB
{
public:
	NBEM();
	~NBEM();

protected:
	vector<sparse_feat> usamp_feat_vec;
    vector< vector<float> > usamp_prb_vec;
    
	vector<float> usamp_class_freq;
    vector< vector<float> > usamp_feat_class_freq;

public:
	vector<float> usamp_class_prb;
    vector< vector<float> > usamp_feat_class_prb;    
	vector<float> comb_class_prb;
    vector< vector<float> > comb_feat_class_prb;

public:
	void load_unlabel_data(string unlabel_file);
	void init_em_ssl(string train_file, string unlabel_file);
	void init_em_usl(string init_file, string unlabel_file);
	void learn_em_ssl(int max_iter, double eps_thrd, float lambda, int output_format);
	void learn_em_usl(int max_iter, double eps_thrd, int output_format);
	
	void learn_em_sslt(int max_iter, double eps_thrd, float lambda, int output_format, string test_file, string output_file);
	void learn_em_uslt(int max_iter, double eps_thrd, int output_format, string test_file, string output_file);

protected:
	void alloc_uniform();
	
	void count_usamp_class_freq();
    void count_usamp_feat_class_freq();	
	void calc_usamp_class_prb();
    void calc_usamp_feat_class_prb();
    
    void calc_comb_class_prb(float lambda);
    void calc_comb_feat_class_prb(float lambda);

	void predict_usamp_prb(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	
	double calc_samp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	double calc_usamp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	double calc_comb_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	double calc_logsum(vector<float> &logp_vec);

};

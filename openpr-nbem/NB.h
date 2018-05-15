/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.07
* Implemented by Rui Xia (rxiacn@gmail.com)
* Last updated on 2011-03-06. 
*********************************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <math.h>

using namespace std;

struct sparse_feat
{
	vector<int> id_vec;
	vector<int> value_vec;
};

class NB
{
public:
	int class_set_size;
	int feat_set_size;
	
	vector<sparse_feat> samp_feat_vec;
    vector<int> samp_class_vec;
    
	vector<int> samp_class_freq;
	vector< vector<int> > samp_feat_class_freq;
	vector<float> samp_class_prb;
    vector< vector<float> > samp_feat_class_prb;
     
public:
    NB();
    ~NB();
	void save_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb);
    void load_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb);
    
	void load_train_data(string training_file);
    void learn(int event_model);
    
	vector<float> predict_logp_bern(sparse_feat samp_feat, vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	vector<float> predict_logp_mult(sparse_feat samp_feat, vector<float> &class_prb, vector< vector<float> > &feat_class_prb);
	
	vector<float> score_to_prb(vector<float> &score, float t = -10);
	int score_to_class(vector<float> &score);
	
	float classify_test_data(string test_file, string output_file, int event_model, int output_format, vector<float> &class_prb, vector< vector<float> > &feat_class_prb);

protected:
	vector<string> string_split(string terms_str, string spliting_tag);	
	
	void read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec);
	
	float calc_acc(vector<int> &true_class_vec, vector<int> &pred_class_vec);

	void count_samp_class_freq();
	void count_samp_feat_class_freq_bern();
    void count_samp_feat_class_freq_mult(); 
    
	void calc_samp_class_prb();
	void calc_samp_feat_class_prb_bern();
    void calc_samp_feat_class_prb_mult();

};



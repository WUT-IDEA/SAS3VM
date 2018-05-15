/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.07
* Implemented by Rui Xia (rxiacn@gmail.com)
* Last updated on 2011-03-06.
*********************************************************************/

#include "NBEM.h"

NBEM::NBEM()
{
}

NBEM::~NBEM()
{
}

void NBEM::load_unlabel_data(string unlabel_file)
{
	cout << "Loading unlabeled data..." << endl;
	vector<int> usamp_class_vec; //meaningless
	read_samp_file(unlabel_file, usamp_feat_vec, usamp_class_vec);
}

void NBEM::init_em_ssl(string train_file, string unlabel_file)
{
	load_train_data(train_file);
	load_unlabel_data(unlabel_file);
	count_samp_class_freq();
	calc_samp_class_prb();
	count_samp_feat_class_freq_mult();
	calc_samp_feat_class_prb_mult();
}

void NBEM::learn_em_ssl(int max_iter, double eps_thrd, float lambda, int output_format)
{
	double logl, logl_pre;
	// Initial E-step
	predict_usamp_prb(samp_class_prb, samp_feat_class_prb);
	logl = calc_comb_logl(samp_class_prb, samp_feat_class_prb);
	cout << "\nEM learning..." << endl;	
	cout << "Initial loglikelihood: " << logl << endl;	
	for (int i = 1; i <= max_iter; i++) {
		cout << "\nIter: " << i << endl;
		// M-step
		count_usamp_class_freq();
		calc_comb_class_prb(lambda);
		count_usamp_feat_class_freq();
		calc_comb_feat_class_prb(lambda);
		// E-step
		vector< vector<float> > usamp_prb_vec_pre = usamp_prb_vec;
		predict_usamp_prb(comb_class_prb, comb_feat_class_prb);
		logl_pre = logl;
		logl = calc_comb_logl(comb_class_prb, comb_feat_class_prb);
		cout << "Loglikelihood: " << logl << ", increasing " << (logl_pre-logl)/logl_pre <<endl;	
		if ((logl_pre-logl)/logl_pre < eps_thrd) { 
			cout << "Reach convergence!" << endl;
			break;
		}
	}
	samp_feat_vec.clear();
	samp_class_vec.clear();
	usamp_feat_vec.clear();
	usamp_prb_vec.clear();
}

// semi-supervised learning with test 
void NBEM::learn_em_sslt(int max_iter, double eps_thrd, float lambda, int output_format, string test_file, string output_file)
{
	float acc = classify_test_data(test_file, output_file, 2, output_format, samp_class_prb, samp_feat_class_prb);
	cout << "Initial Acc: " << acc << endl;
	double logl, logl_pre;
	// Initial E-step
	predict_usamp_prb(samp_class_prb, samp_feat_class_prb);
	logl = calc_comb_logl(samp_class_prb, samp_feat_class_prb);
	cout << "\nEM learning..." << endl;	
	cout << "Initial loglikelihood: " << logl << endl;	
	for (int i = 1; i <= max_iter; i++) {
		cout << "\nIter: " << i << endl;
		// M-step
		count_usamp_class_freq();
		calc_comb_class_prb(lambda);
		count_usamp_feat_class_freq();
		calc_comb_feat_class_prb(lambda);
		// E-step
		vector< vector<float> > usamp_prb_vec_pre = usamp_prb_vec;
		predict_usamp_prb(comb_class_prb, comb_feat_class_prb);
		logl_pre = logl;
		logl = calc_comb_logl(comb_class_prb, comb_feat_class_prb);
		cout << "Loglikelihood: " << logl << ", increasing " << (logl_pre-logl)/logl_pre <<endl;	
		if ((logl_pre-logl)/logl_pre < eps_thrd) { 
			cout << "Reach convergence!" << endl;
			break;
		}
		float acc = classify_test_data(test_file, output_file, 2, output_format, comb_class_prb, comb_feat_class_prb);
		cout << "Acc: " << acc << endl;	
	}
	float final_acc = classify_test_data(test_file, output_file, 2, output_format, comb_class_prb, comb_feat_class_prb);
	cout << "\nFinal acc: " << final_acc << endl;	
	samp_feat_vec.clear();
	samp_class_vec.clear();
	usamp_feat_vec.clear();
	usamp_prb_vec.clear();
}


void NBEM::init_em_usl(string init_file, string unlabel_file)
{
	load_unlabel_data(unlabel_file);
	//alloc_uniform();
	load_model(init_file, usamp_class_prb, usamp_feat_class_prb);
}


void NBEM::learn_em_usl(int max_iter, double eps_thrd, int output_format)
{
	double logl, logl_pre;
	// Initial E-step
	predict_usamp_prb(usamp_class_prb, usamp_feat_class_prb);
	logl = calc_usamp_logl(usamp_class_prb, usamp_feat_class_prb);
	cout << "\nEM learning..." << endl;	
	cout << "Initial loglikelihood: " << logl << endl;	
	for (int i = 1; i <= max_iter; i++) {
		cout << "\nIter: " << i << endl;
		// M-step
		count_usamp_class_freq();
		calc_usamp_class_prb();
		count_usamp_feat_class_freq();
		calc_usamp_feat_class_prb();
		// E-step
		predict_usamp_prb(usamp_class_prb, usamp_feat_class_prb);
		logl_pre = logl;
		logl = calc_usamp_logl(usamp_class_prb, usamp_feat_class_prb);
		cout << "Loglikelihood: " << logl << ", increasing " << (logl_pre-logl)/logl_pre <<endl;	
		if ((logl_pre-logl)/logl_pre < eps_thrd) { 
			cout << "Reach convergence!" << endl;
			break;
		}
	}
	usamp_feat_vec.clear();
	usamp_prb_vec.clear();
}

// un-supervised learning with test 
void NBEM::learn_em_uslt(int max_iter, double eps_thrd, int output_format, string test_file, string output_file)
{
	float acc = classify_test_data(test_file, output_file, 2, output_format, usamp_class_prb, usamp_feat_class_prb);
	cout << "Initial Acc: " << acc << endl;
	double logl, logl_pre;
	// Initial E-step
	predict_usamp_prb(usamp_class_prb, usamp_feat_class_prb);
	logl = calc_usamp_logl(usamp_class_prb, usamp_feat_class_prb);
	cout << "\nEM learning..." << endl;	
	cout << "Initial loglikelihood: " << logl << endl;	
	for (int i = 1; i <= max_iter; i++) {
		cout << "\nIter: " << i << endl;
		// M-step
		count_usamp_class_freq();
		calc_usamp_class_prb();
		count_usamp_feat_class_freq();
		calc_usamp_feat_class_prb();
		// E-step
		predict_usamp_prb(usamp_class_prb, usamp_feat_class_prb);
		logl_pre = logl;
		logl = calc_usamp_logl(usamp_class_prb, usamp_feat_class_prb);
		cout << "Loglikelihood: " << logl << ", increasing " << (logl_pre-logl)/logl_pre <<endl;	
		if ((logl_pre-logl)/logl_pre < eps_thrd) { 
			cout << "Reach convergence!" << endl;
			break;
		}
		float acc= classify_test_data(test_file, output_file, 2, output_format, usamp_class_prb, usamp_feat_class_prb);
		cout << "Acc: " << acc << endl;	
	}
	float final_acc= classify_test_data(test_file, output_file, 2, output_format, usamp_class_prb, usamp_feat_class_prb);
	cout << "\nFinal acc: " << final_acc << endl;
	usamp_feat_vec.clear();
	usamp_prb_vec.clear();
}


void NBEM::alloc_uniform()
{
	usamp_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		usamp_class_prb.push_back(1.0/class_set_size);
	}
	usamp_feat_class_prb.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 1.0/feat_set_size);
		usamp_feat_class_prb.push_back(temp_vec);
	}
}

void NBEM::count_usamp_class_freq()
{
	// allocate
	usamp_class_freq.clear();
	for (int j = 0; j < class_set_size; j++) {
		usamp_class_freq.push_back(0.0);
	}
	// count expected freq
	for (size_t i = 0; i < usamp_prb_vec.size(); i++) {
		for (int j = 0; j < class_set_size; j++) {
			usamp_class_freq[j] += usamp_prb_vec[i][j];
		}
	}	
}
	
void NBEM::calc_usamp_class_prb()
{
	// allocate
	usamp_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		usamp_class_prb.push_back(0.0);
	}
	// freq to prb
    for (int j = 0; j < class_set_size; j++) {
		usamp_class_prb[j] = (float)(1+usamp_class_freq[j])/(class_set_size+usamp_prb_vec.size());
    }
}

void NBEM::count_usamp_feat_class_freq()
{
	// allocate
	usamp_feat_class_freq.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 0.0);
		usamp_feat_class_freq.push_back(temp_vec);
	}
	// count expected freq
    for (size_t i = 0; i != usamp_feat_vec.size(); i++) {
        sparse_feat usamp_feat = usamp_feat_vec[i];
        vector<float> usamp_prb = usamp_prb_vec[i];
		for (int j = 0; j < class_set_size; j++) {
			for (size_t k = 0; k < usamp_feat.id_vec.size(); k++) {
				int feat_id = usamp_feat.id_vec[k];
				int feat_value = usamp_feat.value_vec[k];
				usamp_feat_class_freq[feat_id-1][j] += usamp_prb[j]*feat_value;
			}
		}
    }
}
    
void NBEM::calc_usamp_feat_class_prb()
{
	// allocate
	usamp_feat_class_prb.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 0.0);
		usamp_feat_class_prb.push_back(temp_vec);
	}    
    // column sum
    vector<float> usamp_feat_class_sum(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            usamp_feat_class_sum[j] += usamp_feat_class_freq[k][j];
        }
    }
    // feaq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            usamp_feat_class_prb[k][j] = (float)(1 + usamp_feat_class_freq[k][j])/(feat_set_size + usamp_feat_class_sum[j]); // with Laplace smoothing
        }
    } 
}

void NBEM::calc_comb_class_prb(float lambda)
{
	// allocate
	comb_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		comb_class_prb.push_back(0.0);
	}
	// comb prb
    for (int j = 0; j < class_set_size; j++) {
		comb_class_prb[j] = (1+samp_class_freq[j]+lambda*usamp_class_freq[j])/(class_set_size+samp_class_vec.size()+lambda*usamp_prb_vec.size());
    }
}

void NBEM::calc_comb_feat_class_prb(float lambda)
{
	// allocate
	vector< vector<float> > comb_feat_class_freq;
	comb_feat_class_prb.clear();
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec(class_set_size, 0.0);
		comb_feat_class_freq.push_back(temp_vec);
		comb_feat_class_prb.push_back(temp_vec);
	}
	// comb freq
	for (int k = 0; k < feat_set_size; k++) {
		for (int j = 0; j < class_set_size; j++) {
			comb_feat_class_freq[k][j] = samp_feat_class_freq[k][j] + lambda*usamp_feat_class_freq[k][j];
		}
	}
    // column sum
    vector<float> comb_feat_class_sum(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            comb_feat_class_sum[j] += comb_feat_class_freq[k][j];
        }
    }
    // freq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
			// with Laplace smoothing
			comb_feat_class_prb[k][j] = (float)(1 + comb_feat_class_freq[k][j])/(feat_set_size + comb_feat_class_sum[j]);
        }
    } 
}

void NBEM::predict_usamp_prb(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	usamp_prb_vec.clear();
	for (size_t i = 0; i != usamp_feat_vec.size(); i++) {
		vector<float> logp = predict_logp_mult(usamp_feat_vec[i], class_prb, feat_class_prb);
		vector<float> prb = score_to_prb(logp);
		usamp_prb_vec.push_back(prb);
	}
}

double NBEM::calc_samp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	double samp_logl;
	samp_logl = 0;
	for (size_t i = 0; i < samp_feat_vec.size(); i++) {
		sparse_feat samp_feat = samp_feat_vec[i];
		int samp_class = samp_class_vec[i];
		samp_logl += log(class_prb[samp_class-1]);
		for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
			samp_logl += log(feat_class_prb[feat_id-1][samp_class-1])*feat_value; 
		}
	}
	return samp_logl;
}

double NBEM::calc_usamp_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	double usamp_logl = 0;
	for (size_t i = 0; i < usamp_feat_vec.size(); i++) {
		sparse_feat usamp_feat = usamp_feat_vec[i];
		vector<float> logp_vec = predict_logp_mult(usamp_feat, class_prb, feat_class_prb);
		double logsum = calc_logsum(logp_vec);
		usamp_logl += logsum;
	}
	return usamp_logl;
}


double NBEM::calc_comb_logl(vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
	double samp_logl = calc_samp_logl(class_prb, feat_class_prb);
	double usamp_logl = calc_usamp_logl(class_prb, feat_class_prb);
	return (samp_logl+usamp_logl);
}

// Compute log of sum without overflow
double NBEM::calc_logsum(vector<float> &logp_vec)
{
	float max_logp = logp_vec[0];
	for (size_t j = 1; j < logp_vec.size(); j++) {
		if (logp_vec[j] > max_logp) {
			max_logp = logp_vec[j];
		}
	}
	double logsum = 0;
	double delta_sum = 0;
	for (size_t j = 0; j < logp_vec.size(); j++) {
		delta_sum += exp(logp_vec[j]-max_logp); //
	}
	logsum = log(delta_sum)+max_logp; 
	return logsum;
}



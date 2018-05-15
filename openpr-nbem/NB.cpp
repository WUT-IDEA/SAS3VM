/********************************************************************
* The NBEM (Naive Bayes Expectation-Maximization) Toolkit V1.07
* Implemented by Rui Xia (rxiacn@gmail.com)
* Last updated on 2011-03-06.
*********************************************************************/

#include "NB.h"

NB::NB()
{
}

NB::~NB()
{
}

void NB::save_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb)
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
    for (int j = 0; j < class_set_size; j++) {
        fout << samp_class_prb[j] << " ";
    }
    fout << endl;
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            fout << samp_feat_class_prb[k][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

void NB::load_model(string model_file, vector<float> &samp_class_prb, vector< vector<float> > &samp_feat_class_prb)
{
    cout << "Loading model..." << endl;
    samp_class_prb.clear();
    samp_feat_class_prb.clear();
    ifstream fin(model_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << model_file << endl;
    }
    string line_str;
    // load class_prb
    getline(fin, line_str);
    vector<string> frist_line_vec = string_split(line_str, " ");
    for (vector<string>::iterator it = frist_line_vec.begin(); it != frist_line_vec.end(); it++) {
        float prb = (float)atof(it->c_str());
        samp_class_prb.push_back(prb);
        
    }
    // load feat_class_prb
    while (getline(fin, line_str)) {
        vector<float> prb_vec;
        vector<string> line_vec = string_split(line_str, " ");
        for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
            float prb = (float)atof(it->c_str());
            prb_vec.push_back(prb);
        }
        samp_feat_class_prb.push_back(prb_vec);
    }
    fin.close();
    feat_set_size = (int)samp_feat_class_prb.size();
    class_set_size = (int)samp_feat_class_prb[0].size();
}

void NB::read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec) {
    ifstream fin(samp_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
	int k = 0;
	string line_str;
    while (getline(fin, line_str)) {
		if (k == 0 && line_str[0] == '#'){
			vector<string> class_feat_size = string_split(line_str.substr(1), " ");
			class_set_size =  (int)atoi(class_feat_size[0].c_str());
			feat_set_size =  (int)atoi(class_feat_size[1].c_str());			
		} 
		else {
			size_t class_pos = line_str.find_first_of("\t");
			int class_id = atoi(line_str.substr(0, class_pos).c_str());
			samp_class_vec.push_back(class_id);
			string terms_str = line_str.substr(class_pos+1);
			sparse_feat samp_feat;
			if (terms_str != "") {
				vector<string> fv_vec = string_split(terms_str, " ");
				for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) {
					size_t feat_pos = it->find_first_of(":");
					int feat_id = atoi(it->substr(0, feat_pos).c_str());
					int feat_value = (int)atof(it->substr(feat_pos+1).c_str());
					if (feat_value != 0) {
						samp_feat.id_vec.push_back(feat_id);
						samp_feat.value_vec.push_back(feat_value);              
					}
				}
			}
			samp_feat_vec.push_back(samp_feat);		
		}
		k++;
    }
    fin.close();
}

void NB::load_train_data(string training_file)
{
    cout << "Loading training data..." << endl;
    read_samp_file(training_file, samp_feat_vec, samp_class_vec);
}

void NB::count_samp_class_freq()
{
	// allocate
	samp_class_freq.clear();
	for (int j = 0; j < class_set_size; j++) {
		samp_class_freq.push_back(0);
	}
	// count freq
    for (vector<int>::iterator it_i = samp_class_vec.begin(); it_i != samp_class_vec.end(); it_i++) {
		int samp_class = *it_i;
        samp_class_freq[samp_class-1]++;
    }
 }
    
void NB::calc_samp_class_prb()
{
	// allocate
	samp_class_prb.clear();
	for (int j = 0; j < class_set_size; j++) {
		samp_class_prb.push_back(0.0);
	}
    // freq to prb
    for (int j = 0; j < class_set_size; j++) {
		samp_class_prb[j] = (float)(1+samp_class_freq[j])/(class_set_size+samp_class_vec.size());
    }
}

void NB::count_samp_feat_class_freq_mult()
{
	// allocate
	samp_feat_class_freq.clear();	
	for (int k = 0; k < feat_set_size; k++) {
		vector<int> temp_vec1(class_set_size, 0);
		samp_feat_class_freq.push_back(temp_vec1);
	}
    // count freq
    for (size_t i = 0; i < samp_feat_vec.size(); i++) {
        sparse_feat samp_feat = samp_feat_vec[i];
        int samp_class = samp_class_vec[i];
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
            samp_feat_class_freq[feat_id-1][samp_class-1] += feat_value;
        }
    }
}

void NB::calc_samp_feat_class_prb_mult()
{
	// allocate
	samp_feat_class_prb.clear();	
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec2(class_set_size, 0.0);
		samp_feat_class_prb.push_back(temp_vec2);
	}	
    // column sum
    vector<int> samp_feat_class_sum(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            samp_feat_class_sum[j] += samp_feat_class_freq[k][j];
        }
    }
    // freq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
			// with Laplace smoothing
			samp_feat_class_prb[k][j] = (float)(1 + samp_feat_class_freq[k][j])/(feat_set_size + samp_feat_class_sum[j]);
        }
    } 
}

void NB::count_samp_feat_class_freq_bern()
{
	// allocate
	samp_feat_class_freq.clear();	
	for (int k = 0; k < feat_set_size; k++) {
		vector<int> temp_vec1(class_set_size, 0);
		samp_feat_class_freq.push_back(temp_vec1);
	}
    // count freq
    for (size_t i = 0; i != samp_feat_vec.size(); i++) {
        sparse_feat samp_feat = samp_feat_vec[i];
        int samp_class = samp_class_vec[i];
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            samp_feat_class_freq[feat_id-1][samp_class-1] += 1;
        }
    }	
}
	
void NB::calc_samp_feat_class_prb_bern()
{
	// allocate
	samp_feat_class_prb.clear();	
	for (int k = 0; k < feat_set_size; k++) {
		vector<float> temp_vec2(class_set_size, 0.0);
		samp_feat_class_prb.push_back(temp_vec2);
	}
    // column sum
    vector<int> samp_feat_class_sum(class_set_size, 0);
    for (vector<int>::iterator it_i = samp_class_vec.begin(); it_i != samp_class_vec.end(); it_i++) {
		int samp_class = *it_i;
        samp_feat_class_sum[samp_class-1]++;
    }
    // freq to prb
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
			// with Laplace smoothing
            samp_feat_class_prb[k][j] = (float)(1 + samp_feat_class_freq[k][j])/(2 + samp_feat_class_sum[j]);
        }
    }
}

void NB::learn(int event_model)
{
    cout << "Learning..." << endl;
    count_samp_class_freq();
    calc_samp_class_prb();
    if (event_model == 0) {
		count_samp_feat_class_freq_bern();
        calc_samp_feat_class_prb_bern();
    }
    else {
		count_samp_feat_class_freq_mult();
        calc_samp_feat_class_prb_mult();
    }
    samp_feat_vec.clear();
    samp_class_vec.clear();
}

vector<float> NB::predict_logp_bern(sparse_feat samp_feat, vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
    vector<int> feat_vec_out;
    int i = 0, k = 0;
    while (k < feat_set_size) {
		if ((k+1) != samp_feat.id_vec[i]) {
			feat_vec_out.push_back(k+1);
		}
		else if (i < (int)samp_feat.id_vec.size()-1) {
			i++;
		}
		k++;
    }
    vector<float> logp(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        double logp_samp_given_class = 0.0;
        for (size_t k1 = 0; k1 < samp_feat.id_vec.size(); k1++) {
            int feat_id = samp_feat.id_vec[k1];
            logp_samp_given_class += log(feat_class_prb[feat_id-1][j]); 
        }
        for (size_t k0 = 0; k0 < feat_vec_out.size(); k0++) {
            int feat_id = feat_vec_out[k0];
            logp_samp_given_class += log(1-feat_class_prb[feat_id-1][j]);
        }
        double logp_samp_and_class = logp_samp_given_class + log(class_prb[j]);
        logp[j] = (float)logp_samp_and_class;
    }
    return logp;
}

vector<float> NB::predict_logp_mult(sparse_feat samp_feat, vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
    vector<float> logp(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        double logp_samp_given_class = 0.0;
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
            logp_samp_given_class += log(feat_class_prb[feat_id-1][j])*feat_value;
        }
        double logp_samp_and_class = logp_samp_given_class + log(class_prb[j]);
        logp[j] = (float)logp_samp_and_class;
    }
    return logp;
}

vector<float> NB::score_to_prb(vector<float> &score, float t)
{
	float m = score[0];
	for (int j = 1; j < class_set_size; j++) {
		if (score[j] > m) {
			m = score[j];	
		}
	}
    vector<float> prb(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        float denom = 0.0;
        if (score[j]-m > t) {
			for (int i = 0; i < class_set_size; i++) {
				if (score[i]-m > t) {
					denom += exp(score[i]-m);			
				}
			}
			prb[j] = exp(score[j]-m)/denom;
        }
        else {
			prb[j] = 0;
        }
    }
    return prb;
}

int NB::score_to_class(vector<float> &score)
{
    int pred_class = 0; 
    float max_score = score[0];
    for (int j = 1; j < class_set_size; j++) {
        if (score[j] > max_score) {
            max_score = score[j];
            pred_class = j;
        }
    }
    return ++pred_class;
}

float NB::classify_test_data(string test_file, string output_file, int event_model, int output_format, vector<float> &class_prb, vector< vector<float> > &feat_class_prb)
{
    cout << "Classifying test file..." << endl;
    vector<sparse_feat> test_feat_vec;
    vector<int> test_class_vec;
    vector<int> pred_class_vec;
    read_samp_file(test_file, test_feat_vec, test_class_vec);
    ofstream fout(output_file.c_str());
    for (size_t i = 0; i < test_class_vec.size(); i++) {
        sparse_feat samp_feat = test_feat_vec[i];
        vector<float> pred_score;
        if (event_model == 0) {
            pred_score = predict_logp_bern(samp_feat, class_prb, feat_class_prb);
        }
        else {
            pred_score = predict_logp_mult(samp_feat, class_prb, feat_class_prb);
        }       
        int pred_class = score_to_class(pred_score);
        pred_class_vec.push_back(pred_class);
        fout << pred_class << "\t";
        if (output_format == 1) {
            for (int j = 0; j < class_set_size; j++) {
                fout << pred_score[j] << ' '; 
            }       
        }
        else if (output_format == 2) {
            vector<float> pred_prb = score_to_prb(pred_score);
            for (int j = 0; j < class_set_size; j++) {
                fout << pred_prb[j] << ' '; 
            }
        }
        fout << endl;       
    }
    fout.close();
    float acc = calc_acc(test_class_vec, pred_class_vec);
    return acc; 
}

float NB::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
    size_t len = test_class_vec.size();
    if (len != pred_class_vec.size()) {
        cerr << "Error: two vectors should have the same lenght." << endl;
        exit(0);
    }
    int err_num = 0;
    for (size_t id = 0; id != len; id++) {
        if (test_class_vec[id] != pred_class_vec[id]) {
            err_num++;
        }
    }
    return 1 - ((float)err_num) / len;
}

vector<string> NB::string_split(string terms_str, string spliting_tag)
{
    vector<string> feat_vec;
    size_t term_beg_pos = 0;
    size_t term_end_pos = 0;
    while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) {
        if (term_end_pos > term_beg_pos) {
            string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
            feat_vec.push_back(term_str);
        }
        term_beg_pos = term_end_pos + 1;
    }
    if (term_beg_pos < terms_str.size()) {
        string end_str = terms_str.substr(term_beg_pos);
        feat_vec.push_back(end_str);
    }
    return feat_vec;
}

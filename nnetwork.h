#ifndef NNETWORK_H
#define NNETWORK_H

class NNetwork
{
private:
	vector<double> weightsH;
	vector<double> outHidden;
	vector<double> weightsO;
	vector<uint8_t> inputs;
	double output;
	double logistic();

public:
	NNetwork();
	double process(const std::string& jpegFile);
};	


#endif

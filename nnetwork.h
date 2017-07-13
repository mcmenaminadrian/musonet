#ifndef NNETWORK_H
#define NNETWORK_H

class NNetwork
{
private:
	const int logisticTableSize 100;
	const double logisticTableMax 10.0;
	int row_stride;
	unsigned int widthJPEG;
	unsigned int heightJPEG;
	vector<double> weightsH; //weights into hidden layer from input
	vector<double> biasHidden; //weights of bias into hidden layer
	vector<double> biasOutput; //weight(s) of bias into output
	vector<double> outHidden; //output of hidden layer
	vector<double> weightsO; //weights of output from hidden to output
	vector<uint8_t> inputs; //values of input neurons
	vector<uint8_t> jpegBuffer; //representation of JPEG
	vector<double> logisticValues; //logistic function table
	vector<double> logisticDifferential; //logistic function differential
	double tableFactor;
	void primeLogisticTable();
	double logistic(const double& inValue) const;
	double calculateDotProduct(bool isRow, int number) const;
	void calculateHiddenValues();
	void loadJPEG(const std::string& jpegFile);
	void loadInputs();
	double calculateOutputValue() const;

public:
	NNetwork();
	double process(const std::string& jpegFile);
};	


#endif

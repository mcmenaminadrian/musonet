#ifndef NNETWORK_H
#define NNETWORK_H

class NNetwork
{
private:
	const int logisticTableSize;
	const double logisticTableMax;
	int row_stride;
	unsigned int widthJPEG;
	unsigned int heightJPEG;
	std::vector<double> weightsH; //weights into hidden layer from input
	std::vector<double> biasHidden; //weights of bias into hidden layer
	std::vector<double> biasOutput; //weight(s) of bias into output
	std::vector<double> outHidden; //output of hidden layer
	std::vector<double> weightsO; //weights of output from hidden to output
	std::vector<double> inputs; //values of input neurons
	std::vector<uint8_t> jpegBuffer; //representation of JPEG
	std::vector<double> logisticValues; //logistic function table
	std::vector<double> logisticDifferential; //logistic func differential
	std::vector<std::vector<double> > desired; //outcome sought in training
	std::vector<double> outGradients; //gradient of output from inputs
	std::vector<double> hiddenGradients; // and for hidden layer
	std::vector<double> biasGradients; // and for bias at hidden level 
	double tableFactor;
	void primeLogisticTable();
	double logisticDifferentialFunc(const double& inValue) const;
	double logistic(const double& inValue) const;
	double dotProduct(const bool isRow, const int number) const;
	void calculateHiddenValues();
	void loadJPEG(const std::string& jpegFile);
	void loadData(const std::string& dataFile);
	void loadWeights();
	void writeWeights() const;
	double calculateOutputValue() const;
	void storeScannedLine(JSAMPROW jRow);
	void processInputs(const int startRow, const int startCol);
	void gradientOutputLayer(const double& a, const double& d);
	void gradientHiddenLayer(const double& a, const double& d);
	void tryCorrection(const double& eta);

public:
	NNetwork();
	void process(const std::string& jpegFile, const std::string& dataFile);
};	


#endif

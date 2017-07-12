#ifndef NNETWORK_H
#define NNETWORK_H

class NNetwork
{
private:
	const int logisticTableSize 100;
	const double logisticTableMax 10.0;
	vector<double> weightsH;
	vector<double> biasHidden;
	vector<double> biasOutput;
	vector<double> outHidden;
	vector<double> weightsO;
	vector<uint8_t> inputs;
	vector<double> logisticValues;
	vector<double> logisticDifferential;
	double tableFactor;
	void primeLogisticTable();
	double logistic(const double& inValue) const;
	double calculateDotProduct(bool isRow, int number) const;
	void calculateHiddenValues();

public:
	NNetwork();
	double process(const std::string& jpegFile);
};	


#endif

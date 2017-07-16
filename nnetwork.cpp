#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include "stdint.h"
#include "jpeglib.h"
#include "nnetwork.h"


using namespace std;

//==========logistic function stuff==========

double NNetwork::logisticDifferentialFunc(const double& inValue) const
{
	double result = inValue * (1.0 - inValue);
	return result;
}

double NNetwork::logistic(const double& inValue) const
{

	double result = 1.0 / (1.0 + exp(-inValue));
	return result;
}


//==========save and load weights==========

void NNetwork::writeWeights() const
{
	ofstream weightStream("nnetweights.txt");
	for (unsigned int i = 0; i < weightsH.size(); i++) {
		weightStream << weightsH.at(i);
		weightStream << " ";
	}
	weightStream << endl;
	for (unsigned int i = 0; i < biasHidden.size(); i++) {
		weightStream << biasHidden.at(i);
		weightStream << " ";
	}
	weightStream << endl;
	for (unsigned int i = 0; i < weightsO.size(); i++) {
		weightStream << weightsO.at(i);
		weightStream << " ";
	}
	weightStream << endl;
	weightStream << biasOutput.at(0);
	weightStream << endl;
	weightStream.close();
}

void NNetwork::loadWeights()
{

	ifstream weightStream("nnetweights.txt");
	double x;
	char c;
	for (int i = 0; i < 20000; i++) {
		weightStream >> x;
		weightsH.push_back(x);
	}
	weightStream >> c;
	for (int i = 0; i < 200; i++) {
		weightStream >> x;
		biasHidden.push_back(x);
	}
	weightStream >> c;
	for (int i = 0; i < 200; i++) {
		weightStream >> x;
		weightsO.push_back(x);
	}
	weightStream >> c >> x;
	biasOutput.push_back(x);
	weightStream.close();
/*		

	double factor = RAND_MAX;
	for (int i = 0; i < 20000; i++) {
		double x = rand();
		x /= factor;
		weightsH.push_back(x);
	}
	for (int j = 0; j < 200; j++) {
		double y = rand();
		y /= factor;
		weightsO.push_back(y);
	}
	//bias weights for hidden layer
	for (int k = 0; k < 200; k++) {
		double z = rand();
		z /= factor;
		biasHidden.push_back(z);
	}
	double zz = rand();
	zz /= factor;
	biasOutput.push_back(zz);
*/
}

//==========calculate outputs==========

//200 hidden neurons
//one for each row in input
//one for each column in input
//20000 weights as each input neuron linked to 2 hidden neurons
//in hidden weights 0 - 99 weights for row 0
//100 - 199 weights for column 0 etc
//and for hidden neuron 0, all of row 0 connected
//and for hidden neuron 1, all of column 0 connected etc 

double NNetwork::dotProduct(const bool isRow, const int number) const
{
	double result = 0.0;
	if (isRow) {
		for (int i = 0; i < 100; i++) {
			result += inputs.at(number * 100 + i) *
				weightsH.at(number * 200 + i);
		}
	} else {
		for (int i = 0; i < 100; i++) {
			result += inputs.at(i * 100 + number) *
				weightsH.at(100 + number * 200 + i);
		}
	}
	return result;
}

void NNetwork::calculateHiddenValues()
{
	outHidden.clear();
	for (int i = 0; i < 200; i++) {
		if (i%2 == 0) {
			//row
			outHidden.push_back(logistic(dotProduct(true, i/2)
				+ biasHidden.at(i)));
		} else {
			//column
			outHidden.push_back(logistic(dotProduct(false, i/2)
				+ biasHidden.at(i)));
		}
	}
}

double NNetwork::calculateOutputValue() const
{
	double result = 0.0;
	for (unsigned int i = 0; i < outHidden.size(); i++) {
		double neuronResult = outHidden.at(i) * weightsO.at(i);
		result += neuronResult;
	}
	result += biasOutput.at(0);
	return logistic(result);
}

void NNetwork::gradientOutputLayer(const double& actual, const double& desired)
{
	outGradients.clear();
	double missedBy = desired - actual;
	for (int i = 0; i < 200; i++) {
		outGradients.push_back(2 * missedBy *
			logisticDifferentialFunc(actual) * outHidden.at(i));
	}
	//for bias
	outGradients.push_back(-2 * missedBy * 
		logisticDifferentialFunc(actual) * biasOutput.at(0));
}

void NNetwork::gradientHiddenLayer(const double& actual, const double& desired)
{
	hiddenGradients.clear();
	double missedBy = desired - actual;
	for (int i = 0; i < 20000; i++) {
		hiddenGradients.push_back( -2 * missedBy *
			weightsO.at(i/100) * 
			logisticDifferentialFunc(outHidden.at(i / 100)) *
			inputs.at(i/2));
	}
	biasGradients.clear();
	for (int i = 0; i < 200; i++) {
		biasGradients.push_back(-2 * missedBy * weightsO.at(i) *
			logisticDifferentialFunc(outHidden.at(i)));
	}
}

void NNetwork::tryCorrection(const double& eta)
{

	for (int i = 0; i < 200; i++) {
		weightsO.at(i) -= (eta * outGradients.at(i));
	}
	biasOutput.at(0) -= (eta * outGradients.at(200));
	for (int i = 0; i < 20000; i++) {
		weightsH.at(i) -= (eta * hiddenGradients.at(i));
	}
	for (int i = 0; i < 200; i++) {
		biasHidden.at(i) -= (eta * biasGradients.at(i));
	}
}

//==========image handling==========

void NNetwork::storeScannedLine(JSAMPROW sampledLine)
{
	for (int i = 0; i < row_stride; i++) {
		unsigned char x = *(sampledLine + i);
		jpegBuffer.push_back(x);
	}
}

void NNetwork::loadJPEG(const string & jpegFile)
{
	//load the jpeg
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE* inFile;
	JSAMPARRAY buffer;

	cout << "Opening " << jpegFile << endl;

	if ((inFile = fopen(jpegFile.c_str(), "rb")) == NULL) {
		fprintf(stderr, "cannot open %s\n", jpegFile.c_str());
		return;
	}

	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, inFile);

	jpeg_read_header(&cinfo, TRUE);
	cinfo.out_color_space = JCS_GRAYSCALE;
	jpeg_start_decompress(&cinfo);
	row_stride = cinfo.output_width * cinfo.output_components;
	buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr)
		&cinfo, JPOOL_IMAGE, row_stride, 1);

	while (cinfo.output_scanline < cinfo.output_height) {
		jpeg_read_scanlines(&cinfo, buffer, 1);
		storeScannedLine(buffer[0]);
	}
	widthJPEG = cinfo.output_width;
	heightJPEG = cinfo.output_height;

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(inFile);

	return;
}

void NNetwork::processInputs(const int startRow, const int startCol)
{
	inputs.clear();
	for (unsigned int i = 0; i < 100; i++) {
		for (int j = 0; j < 100; j++) {
			double jpegValue = jpegBuffer.at(
				(startRow + i) * widthJPEG + j + startCol);
			inputs.push_back(jpegValue/2550);
		}
	}
}

void NNetwork::loadData(const string& dataFile)
{
	//default set every square to false
	for (unsigned int i = 0; i < heightJPEG / 100; i++) {
		desired.push_back(vector<double>(widthJPEG/100, 0.0));
	}
	//load the file
	ifstream dataStream(dataFile);
	string line;
	while (getline(dataStream, line)) {
		istringstream iss(line);
		unsigned int xCoord, yCoord;
		char c;
		iss >> xCoord >> c >> yCoord;
		(desired.at(yCoord/100)).at(xCoord/100) = 1.0;
	}
}


void NNetwork::process(const string& jpegFile, const string& dataFile)
{
	jpegBuffer.clear();
	loadJPEG(jpegFile);
	loadData(dataFile);
	double totalError = 0.0;
	int cases = 0;
	for (int j = 0; j < 10000; j++) {
	for (unsigned i = 0; i < (heightJPEG / 100) * 100; i+= 100) {
		for (unsigned int j = 0; j < (widthJPEG / 100) * 100; j+=100) {
			processInputs(i, j);
			calculateHiddenValues();
//			cout << "For JPEG beginning at ( " << i << "," << j;
//			cout << "output value is ";
			double outputValue = calculateOutputValue();
//		       	cout << outputValue;
			double desiredValue = 
				(desired.at(i / 100)).at(j / 100);
//			cout << " sought " << desiredValue;
			double error = outputValue - desiredValue;
			gradientOutputLayer(outputValue, desiredValue);
			gradientHiddenLayer(outputValue, desiredValue);
			tryCorrection(0.5);
			error = error * error;
			totalError += error;
//			cout << " Squared error is " << error << endl;
			cases++;
		}
	}
	cout << "Mean error is " << totalError / cases << endl;
	cout << "ITERATION: " << j << endl;
}
	writeWeights();
}

NNetwork::NNetwork():logisticTableSize(100), logisticTableMax(10.0)
{
	//test with same sequence
//	srand(100);
	//proper psuedo random sequence
	srand(time(0));
	loadWeights();
}


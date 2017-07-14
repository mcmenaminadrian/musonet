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


void NNetwork::primeLogisticTable()
{
	tableFactor = (logisticTableSize - 1) / logisticTableMax;
	int j = 0;
	for (double i = 0; i < logisticTableSize; i++) {
		logisticValues.push_back(1.0/(1.0 + exp(-i/tableFactor)));
		if (j > 0) {
			logisticDifferential.push_back(logisticValues.at(j) -
				logisticValues.at(j - 1));
		}
		j++;
	}
}

double NNetwork::logistic(const double& inValue) const
{
	double indexD;
	int index;

	if (inValue >= 0.0) {
		indexD = inValue * tableFactor;
		index = indexD;
		if (index >= logisticTableSize - 1) {
			return logisticValues.at(logisticTableSize - 1);
		} else {
			return logisticValues.at(index) +
				logisticDifferential.at(index) * 
				(indexD - index);
		}
	} else {
		indexD = -inValue * tableFactor;
		index = indexD;
		if (index >= logisticTableSize - 1) {
		       return 1.0 - logisticValues.at(logisticTableSize - 1);
		} else {
			return 1.0 - (logisticValues.at(index) +
				logisticDifferential.at(index) * 
				(indexD - index));
		}
	}
}


//this is just a filler for now
void NNetwork::loadWeights()
{
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
}


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
		result += outHidden.at(i) * weightsO.at(i);
	}
	result += biasOutput.at(0);
	return logistic(result);
}

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
			inputs.push_back(jpegBuffer.at(
				(startRow + i) * widthJPEG + j + startCol));
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
	for (unsigned i = 0; i < (heightJPEG / 100) * 100; i+= 100) {
		for (unsigned int j = 0; j < (widthJPEG / 100) * 100; j+=100) {
			processInputs(i, j);
			calculateHiddenValues();
			cout << "For JPEG beginning at ( " << i << "," << j;
			cout << " ) output value is ";
			double outputValue = calculateOutputValue();
		       	cout << outputValue << endl;
			double error = outputValue -
				(desired.at(i / 100)).at(j / 100);
			error = error * error;
			totalError += error;
			cout << "Squared error is " << error << endl;
		}
	}
	cout << "Total error is " << totalError << endl;
}

NNetwork::NNetwork():logisticTableSize(100), logisticTableMax(10.0)
{
	//test with same sequence
//	srand(100);
	//proper psuedo random sequence
	srand(time(0));

	primeLogisticTable();
	loadWeights();
}


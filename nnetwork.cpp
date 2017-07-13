#include <iostream>
#include <cstdlib>
#include <ctime>
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
	for (int i = 0; i < 20000; i++) {
		double x = rand()/MAX_RAND;
		weightsH.push_back(x);
	}
	for (int j = 0; j < 200; j++) {
		double y = rand()/MAX_RAND;
		weightsO.push_back(y);
	}
	//bias weights for hidden layer
	for (int k = 0; k < 200; k++) {
		double z = rand() / MAX_RAND;
		biasHidden.push_back(z);
	}
	double zz = rand() / MAX_RAND;
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
			result += input.at(number * 100 + i) *
				weightsH(number * 200 + i);
		}
	} else {
		for (int i = 0; i < 100; i++) {
			result += input.at(i * 100 + number) *
				weightsH(100 + number * 200 + i);
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
				+ biasHidden(i)));
		} else {
			//column
			outHidden.push_back(logistic(dotProduct(false, i/2)
				+ biasHidden(i)));
		}
	}
}

double NNetwork::calculateOutputValue() const
{
	double result = 0.0;
	for (int i = 0; i < outHidden.size(); i++) {
		result += outHidden.at(i) * weightsO.at(i);
	}
	result += biasOuput(0);
	return logistic(result);
}

void NNetwork::storeScannedLine(JSAMPROW sampledLine)
{
	for (unsigned int i = 0; i < row_stride; i++) {
		unsigned char x = *(sampledLine + i);
		jpegBuffer.push_back(x);
	}
}

void NNetwork::loadJPEG(const string & jpegFile)
	//load the jpeg
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE* inFile;
	JSAMPARRAY buffer;

	cout << "Opening " << jpegFile << endl;

	if (((inFile = jpegFile.c_str(), "rb")) == NULL){
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
	setByteWidth(row_stride);
	buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr)
		&cinfo, JPOOL_IMAGE, row_stride, 1);

	while (cinfo.output_scanline < cinfo.output_height) {
		jpeg_read_scanlines(&cinfo, buffer, 1);
		storeScannedLine(buffer[0]);
	}
	cInfo = cinfo;
	widthJPEG = cinfo.output_width;
	heightJPEG = cinfo.output_height;

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(inFile);

	return;
}

void NNetwork::processInputs(const int& startRow, const int& startCol)
{
	inputs.clear();
	for (unsigned int i = 0; i < 100; i++) {
		for (int j = 0; j < 100; j++) {
			inputs.push_back(jpegBuffer.at(
				(startRow + i) * widthJPEG + j + startCol));
		}
	}
}

double NNetwork::process(const string& jpegFile)
{
	jpegBuffer.clear();
	loadJPEG(jpegFile);
	for (unsigned i = 0; i < heightJPEG; i+= 100) {
		for (unsigned int j = 0; j < widthJPEG; j+=100) {
			loadInputs(i, j);
			calculateHiddenValues();
			cout << "For JPEG beginning at ( " << i << "," << j;
			cout << " ) output value is ";
		       	cout << calculateOutputValue() << endl;
		}
	}
}



NNetwork::NNetwork()
{
	//test with same sequence
	srand(100);
	//proper psuedo random sequence
	//srand(time(0));

	primeLogisticTable();
	loadWeights();
}


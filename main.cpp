#include <iostream>
#include <string>
#include "jpegimage.h"
#include "nnetwork.h"

//MUSONET - copyright Adrian McMenamin <adrianmcmenamin@gmail.com>, 2017
//MUSONET is licenced for use and distrubution under the terms of the
//GNU General Public License version 3 or any later version at your
//discretion. Please note this is experimental software and no
//guarantee or waranty can be offered as to its performance.
//


using namespace std;

int main(int argc, char *argv[])
{
	//for now we take the last paramter to be the image file
	string fileName(argv[argc - 1]);
	JpegImage testPEG(fileName);
	NNetwork neuralNet();
	cout << neuralNet.process(testPEG) << endl;
	return 1;
}


#include <iostream>
#include <string>
#include <vector>
#include "stdint.h"
#include "jpeglib.h"
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
	//for now we take the last paramter to be the data file
	string testJPEG(argv[argc - 2]);
	string testData(argv[argc -1]);
	NNetwork neuralNet;
	neuralNet.process(testJPEG, testData);
	return 1;
}


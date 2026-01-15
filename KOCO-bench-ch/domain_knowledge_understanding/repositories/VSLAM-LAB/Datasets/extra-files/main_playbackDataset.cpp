/* Copyright (c) 2016, Jakob Engel
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation and/or 
 * other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors 
 * may be used to endorse or promote products derived from this software without 
 * specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include "opencv2/opencv.hpp"


#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <iomanip> 

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include "BenchmarkDatasetReader.h"

int main( int argc, char** argv )
{
	setlocale(LC_ALL, "");

	std::string dataset = argv[1];
        std::string outputPath = argv[2];


	DatasetReader* reader = new DatasetReader(dataset);

	Eigen::Matrix3f K_rect = reader->getUndistorter()->getK_rect();
	Eigen::Vector2i dim_rect = reader->getUndistorter()->getOutputDims();

	
        std::ofstream calibFile(outputPath + "/calibration.txt");
        calibFile << K_rect(0,0) << " " << K_rect(1,1) << " " << K_rect(0,2) << " " << K_rect(1,2) << std::endl;
        calibFile.close();
        
	Eigen::Matrix3f K_org= reader->getUndistorter()->getK_org();
	Eigen::Vector2i dim_org = reader->getUndistorter()->getInputDims();
	float omega = reader->getUndistorter()->getOmega();

	for(int i=0; i < reader->getNumImages(); i++)
	{
		ExposureImage* I = reader->getImage(i, true, false, false, false);
		std::ostringstream oss;
    		oss << std::setw(5) << std::setfill('0') << i;
		cv::imwrite(outputPath + "/rgb/" + oss.str() + ".jpg", cv::Mat(I->h, I->w, CV_32F, I->image));
		delete I;
	}

	return 0;
}


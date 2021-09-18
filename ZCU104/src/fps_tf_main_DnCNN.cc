/*
## © Copyright (C) 2016-2020 Xilinx, Inc
##
## Licensed under the Apache License, Version 2.0 (the "License"). You may
## not use this file except in compliance with the License. A copy of the
## License is located at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
##
## Modified by Fernando Carrió - 2021
## Throughput test for the DnCNN network
*/

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

// header file OpenCV for image processing
#include <opencv2/opencv.hpp>

// header files for DNNDK APIs
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

int threadnum;

#define KERNEL_CONV "DnCNN"
#define CONV_INPUT_NODE "dncnn_block1_Conv2D"
#define CONV_OUTPUT_NODE "dncnn_blockOutput_Conv2D"
const string baseImageInputPath = "./InputImages/";
const string baseImageOutputPath = "./OutputImages/";

#ifdef SHOWTIME
#define _T(func)                                                              \
    {                                                                         \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
    }
#else
#define _T(func) func;
#endif


/*List all images's name in path.*/
void ListImages(std::string const &path, std::vector<std::string> &images) {
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
          (ext == "bmp") ||  (ext == "BMP") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
//        cout << name << endl;
      }
    }
  }
  sort(images.begin(), images.end()); 

  closedir(dir);
}

void normalize_image(const cv::Mat& image, int8_t* data, float scale) {
    for(int j = 0; j < image.rows; ++j) {
      for(int k = 0; k < image.cols; ++k) {
	      data[j*image.cols+k] = ((float(image.at<uint8_t>(j,k))))*scale;
      }
     }
}

void create_image(cv::Mat& image_or, cv::Mat& image_rs, int8_t* data, float scale) {
   for(int j = 0; j < image_rs.rows; ++j) {
      for(int k = 0; k < image_rs.cols; ++k) {
	image_rs.at<uint8_t>(j,k) = (float(data[j*image_rs.cols+k])*scale);
      }
     }
}



inline void set_input_image(DPUTask *task, const string& input_node, const cv::Mat& image,int ind)
{
  //Mat cropped_img;
  DPUTensor* dpu_in = dpuGetInputTensor(task, input_node.c_str());
  float scale = dpuGetTensorScale(dpu_in);
  int width = dpuGetTensorWidth(dpu_in);
  int height = dpuGetTensorHeight(dpu_in);
  int size = dpuGetTensorSize(dpu_in);
  int8_t* data = dpuGetTensorAddress(dpu_in);
  normalize_image(image, data, scale);
}


vector<string> kinds, images; //DB


void run_CNN(DPUTask *taskConv, Mat img, int ind)
{
  assert(taskConv);

  // Set image into Conv Task with mean value
  cv::Mat image_or = img.clone();
  set_input_image(taskConv, CONV_INPUT_NODE, img, ind);

  dpuRunTask(taskConv);


  int8_t *output = new int8_t[320*320];
  int8_t output2;
  output2 = dpuGetOutputTensorInHWCInt8(taskConv,CONV_OUTPUT_NODE,output,102400,0);


  DPUTensor* dpu_out = dpuGetOutputTensor(taskConv,CONV_OUTPUT_NODE);

  float scale = dpuGetTensorScale(dpu_out);
  cv::Mat image3(320, 320, CV_8UC1);
  create_image(image_or,image3,output,scale);
  std::stringstream id;
  id << std::setw(2) << std::setfill('0') << ind;
  imwrite(baseImageOutputPath+images.at(ind),image3);  
  //imwrite(baseImageOutputPath+KERNEL_CONV+"_"+id.str()+"_HR.png",image3);
}


/**
 * @brief Run DPU CONV Task for Keras Net
 *
 * @param taskConv - pointer to CONV Task
 *
 * @return none
 */
void superResolution(DPUKernel *kernelConv)
{

  //  vector<string> kinds, images;

  /*Load all image names */
  ListImages(baseImageInputPath, images);
  if (images.size() == 0) {
    cerr << "\nError: Not images exist in " << baseImageInputPath << endl;
    return;
  } else {
//    cout << "total image : " << images.size() << endl;
  }

  /* ************************************************************************************** */
  //DB added multi-threding code

#define DPU_MODE_NORMAL 0
#define DPU_MODE_PROF   1
#define DPU_MODE_DUMP   2

  thread workers[threadnum];
  auto _start = system_clock::now();

  for (auto i = 0; i < threadnum; i++)
  {
  workers[i] = thread([&,i]()
  {

    /* Create DPU Tasks for CONV  */
    DPUTask *taskConv = dpuCreateTask(kernelConv, DPU_MODE_NORMAL); // profiling not enabled


    for(unsigned int ind = i  ;ind < images.size();ind+=threadnum)
      {
        cv::Mat img(320, 320, CV_8UC1);
        img = imread(baseImageInputPath + images.at(ind), IMREAD_GRAYSCALE);
	      run_CNN(taskConv, img, ind);
      }
    // Destroy DPU Tasks & free resources
    dpuDestroyTask(taskConv);
  });
  }

  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }

  auto _end = system_clock::now();
  auto duration = (duration_cast<microseconds>(_end - _start)).count();
  cout << "[Time]" << duration << "us" << endl;
  cout << "[FPS]" << images.size()*1000000.0/duration  << endl;

}

/**
 * @brief Entry for running DnCNN network
 *
 * @return 0 on success, or error message dispalyed in case of failure.
 */
int main(int argc, char *argv[])
{

  DPUKernel *kernelConv;

  if(argc == 2) {
    threadnum = stoi(argv[1]);
    cout << "now running " << argv[0] << " " << argv[1] << endl;
  }
  else
      cout << "now running " << argv[0] << endl;

  /* Attach to DPU driver and prepare for running */
  dpuOpen();

  /* Create DPU Kernel */
  kernelConv = dpuLoadKernel(KERNEL_CONV); //DB

  /* run Super Resolution  */
  superResolution(kernelConv);

  /* Destroy DPU Kernel  */
  dpuDestroyKernel(kernelConv);

  /* Dettach from DPU driver & release resources */
  dpuClose();

  return 0;
}

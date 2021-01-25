#ifndef __DEEPTOOLKIT_BATCH_OPS__
#define __DEEPTOOLKIT_BATCH_OPS__

#define __PY_SSIZE_T_CLEAN__
#include <Python/Python.h>

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Path operation definitions.
bool path_exists(const char* path);
std::vector<std::string> get_paths(const char *path);

// Image batch operations.
std::vector<cv::Mat> batch_image_read(const char *path);
std::vector<cv::Mat> batch_image_resize_with_size(const std::vector<cv::Mat>& images,
                                                  cv::Size& size(int& rows, int& cols));
std::vector<cv::Mat> batch_image_resize_with_factor(const std::vector<cv::Mat>& images,
                                                    const double& dx, const double& dy);

#endif

//
// Created by Amogh Joshi on 1/25/21.
//

#include "batch_ops.h"
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace cv;

namespace deeptoolkit
{
    bool path_exists(const char* path)
    {
        /**
         * Determine if a provided path exists.
         * @param: The provided path.
         */
        if (path == nullptr)
            return false;

        DIR *pathDir;
        bool pathExists = false;

        pathDir = opendir(path);

        if (pathDir != nullptr)
        {
            pathExists = true;
            (void) closedir(pathDir);
        }

        return pathExists;
    }

    std::vector<string> get_paths(const char *path)
    {
        /**
         * Get a list of paths within the provided directory path.
         * @param path: The provided directory path.
         */
        string provided_path = path;
        vector<string> paths;
        assert(path_exists(path));

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(path)) != nullptr)
        {
            while ((ent = readdir(dir)) != nullptr)
            {
                const char *fp_short = ent->d_name;
                string fp = string(path) + "/" + string(fp_short);

                /* Determine if object is a . or .. relative directory path. */
                struct stat buf{};
                stat(fp.c_str(), &buf);
                if (S_ISDIR(buf.st_mode))
                    continue;

                paths.emplace_back(fp);
            }
            closedir(dir);
        }
        else
        {
            const char* msg = "Error while trying to iterate over files in provided directory";
            perror(msg);
        }

        return paths;
    }

    std::vector<Mat> batch_image_read(const char* path)
    {
        /**
         * Reads a batch of images from a provided image directory.
         * @param path: The provided image directory.
         */
        assert(path_exists(path)); // Confirm that provided path exists.

        /* Create list of images and paths. */
        vector<Mat> images;
        vector<string> paths = get_paths(path);

        for (string path: paths)
        {
            const char *pass_path = path.c_str();
            Mat image = imread(pass_path, 0);
            if (!image.empty()) {
                /* Load image and add to vector. */
                images.push_back(image);
            } else {
                /* Failed to read image path. */
                cout << "Failed to read image " << pass_path << endl;
            }
        }

        return images;
    }

    std::vector<Mat> base_batch_image_resize_with_size(const std::vector<Mat>& images,
                                                       Size& size(int& rows, int& cols))
    {
        /**
         * Resizes a provided batch of images to a certain provided size.
         * @param images: The provided batch of images.
         * @param size: What to resize the images to.
         */

        /* Create list of resized images. */
        vector<Mat> resized_images;

        for (const Mat& image: images)
        {
            Mat resized;
            cv::resize(image, resized, reinterpret_cast<Size_<int> &&>(size), 0, 0, cv::INTER_LINEAR);
            resized_images.push_back(resized);
            cout << resized.size << endl;
        }

        return resized_images;
    }

    std::vector<cv::Mat> base_batch_image_resize_with_factor(const std::vector<cv::Mat>& images,
                                                             const double& dx, const double& dy)
    {
        /**
         * Resizes a provided batch of images to a certain provided size.
         * @param images: The provided batch of images.
         * @param size: What to resize the images to.
         */

        /* Create list of resized images. */
        vector<Mat> resized_images;

        for (const Mat& image: images)
        {
            Mat resized;
            cv::resize(image, resized, Size(), dx, dy, cv::INTER_LINEAR);
            resized_images.push_back(resized);
            cout << resized.size << endl;
        }

        return resized_images;
    }
}



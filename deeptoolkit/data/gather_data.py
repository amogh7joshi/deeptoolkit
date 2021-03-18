#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import asyncio
import aiohttp

from tqdm import tqdm

from deeptoolkit.internal.asynchronous import fetch_image_batch

def gather_images(image_urls, output_files = None, bypass = False, redownload = False):
   """Gathers images from provided urls asynchronously, for high efficiency.

   Given a provided set of image urls, this method will download all of
   the images to either the provided output filenames, or the function will
   automatically download the images to a `Images` directory in the current
   working directory, each image being titled its name in the URL.

   This is primarily for batch downloading large sets of images, especially
   when building large datasets. As opposed to individually downloading
   images one at a time, making use of this function can speed up download
   anywhere from 30-50 times the original speed.

   Examples:

   # >>> image_urls = ["url_1", "url_2", "url_3"]
   # >>> output_files = ["URLImages/image1.jpg", "URLImages/image2.jpg", "URLImages/image3.jpg"]

   Arguments:
      - image_urls: The URLs to the images you want to download.
      - output_files: An optional argument for the output filenames.
      - bypass: Whether you want to skip images that have failed to download
                or raise an error to check what went wrong.
      - redownload: If you are re-downloading a batch, setting this to True will
                    clear and redownload a new set of images, and not just
                    skip the images that already exist at the determined paths.
   """
   # First, determine if output paths have been provided.
   if output_files is not None:
      # Check to make sure there are enough paths.
      if len(output_files) != len(image_urls):
         raise IndexError(f"The number of output files ({len(output_files)}) "
                          f"does not match the number of image URLS {(len(image_urls))}.")

      # Otherwise, make sure that the file endings of the output paths
      # match the file endings of the actual images from the URLs.
      for indx, (image_url, output_path) in enumerate(zip(image_urls, output_files)):
         # Get the file endings of both.
         url_extension = os.path.splitext(os.path.basename(image_url))[1]
         path_extension = os.path.splitext(os.path.basename(output_path))[1]

         # Ensure that they are equal, and if not, modify the output path.
         if url_extension != path_extension:
            # Create the new filepath.
            file_body = os.path.splitext(os.path.basename(output_path))[0]
            new_filename = "".join([file_body, url_extension])
            output_files[indx] = new_filename
   else:
      # If no output paths have been provided, then make output paths in
      # an `Images` directory in the user's current working directory.
      current_directory = os.getcwd()
      image_directory = os.path.join(current_directory, 'Images')

      # Create the new image path if it doesn't exist.
      if not os.path.exists(image_directory):
         os.makedirs(image_directory)

      # Create the output paths.
      output_files = []
      for image_url in image_urls:
         # Get the image name.
         image_name = os.path.basename(image_url)

         # Create the full path.
         output_files.append(os.path.join(image_directory, image_name))

   # Convert the flattened lists of urls and paths into image batches.
   url_batches = []
   path_batches = []
   for indx in range(0, len(image_urls), 50):
      try: # Try adding a full batch of 50.
         url_batches.append(image_urls[indx: indx + 50])
         path_batches.append(output_files[indx: indx + 50])
      except IndexError: # Otherwise, just the remaining urls/paths.
         url_batches.append(image_urls[indx:])
         path_batches.append(output_files[indx:])

   # Construct an inner asynchronous function to dispatch to the main
   # image fetching method, which the current function will execute.
   async def dispatch_to_gather_images():
      for url_batch, file_batch in tqdm(
            zip(url_batches, path_batches), desc = "Downloading All Images"):
         async with aiohttp.ClientSession() as sess:
            await fetch_image_batch(sess, url_batch, file_batch, bypass, redownload)

   # Execute the image gathering method.
   asyncio.run(dispatch_to_gather_images())


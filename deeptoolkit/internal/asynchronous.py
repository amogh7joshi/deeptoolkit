#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import asyncio
import aiohttp
import aiofiles

async def fetch_single_file(session, url, fname, bypass):
   """Utility function to download a single file from a given
   URL as a coroutine (for high efficiency).

   Used in the `deeptoolkit.data` module, to fetch a piece of
   image data from a given source.
   """
   async with session.get(url) as resp:
      # Ensure that a valid response has been received.
      if resp.status != 200:
         if bypass: # Allow the skipping of certain files.
            return False
         else: # Otherwise, raise an error.
            raise PermissionError(
               "Received a non-200 status code, implying that an error "
               f"has occured for the image at {url}. Check the URL.")
      else:
         # Read the page content and save it to the output file.
         async with aiofiles.open(fname, 'wb') as save_file:
            await save_file.write(await resp.read())

async def fetch_image_batch(session, urls, fnames, bypass, redownload):
   """Utility function to fetch a batch of images from a set of given
   URLs as coroutines (for high efficiency).

   Used in the `deeptoolkit.data` module, to fetch batches of image
   data from provided sources.
   """
   tasks = []

   # Create all of the different tasks for each image.
   for media_url, fname in zip(urls, fnames):
      # Check whether the image already exists, for the case
      # where a user is re-downloading a batch of images.
      if not redownload:
         if os.path.exists(fname):
            continue

      # Otherwise, create the coroutines.
      task = asyncio.create_task(fetch_single_file(
         session, media_url, fname, bypass))
      tasks.append(task)

   # Gather and execute the different coroutines.
   return await asyncio.gather(*tasks)





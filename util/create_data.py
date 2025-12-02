# define preprocessing function to map DICOM file to standard image (sitk version)
import os, glob, re
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from collections import defaultdict
from util.helpers import *

MAX_FILES = 50000
delimiter = '_Dx-'

def create_metadata_sitk(path):
    """
    Reads DICOM files, extracts metadata, and groups slices by SeriesInstanceUID 
    using SimpleITK (sitk).

    Args:
        path (str or Path): The root directory containing patient folders with DICOM files.

    Returns:
        tuple: (modality_count, series)
            modality_count (defaultdict): Count of files per Modality.
            series (defaultdict): Dictionary grouping slices by series key (Modality_UID).
    """
    series = defaultdict(list)
    modality_count = defaultdict(int)
    dicomFiles = {}
    fileCount = 0

    # 1. Collect all desired DICOM file paths and their sample names
    for folder_path in Path(path).iterdir():
        if folder_path.is_dir():
            try:
                sampleName = folder_path.name.split(delimiter)[1]
            except IndexError:
                continue

            for dcmFile in folder_path.rglob('*.dcm'):
                if fileCount >= MAX_FILES:
                    break
                dicomFiles[str(dcmFile)] = sampleName
                fileCount += 1
            if fileCount >= MAX_FILES:
                break

    # 2. Extract metadata for collected files using SimpleITK
    for filePath, sampleName in dicomFiles.items():
        try:
            # Use SimpleITK to read metadata (and image data, which we discard here)
            image = sitk.ReadImage(filePath)
            
            # Use SimpleITK's GetMetaData methods
            modality = image.GetMetaData('0008|0060') if image.HasMetaDataKey('0008|0060') else None
            seriesUID = image.GetMetaData('0020|000e') if image.HasMetaDataKey('0020|000e') else None
            inst = image.GetMetaData('0020|0013') if image.HasMetaDataKey('0020|0013') else None
            ipp_str = image.GetMetaData('0020|0032') if image.HasMetaDataKey('0020|0032') else None
            miscUID = image.GetMetaData('0008|0018') if image.HasMetaDataKey('0008|0018') else None # SOPInstanceUID
            
            # --- Metadata Validation ---
            if modality != 'CT':
                continue # Skip non-CT files

            if seriesUID is None:
                continue # Skip files without a SeriesInstanceUID
            
            # Extract Z-coordinate from ImagePositionPatient (IPP) string "x\y\z"
            z = float(ipp_str.split('\\')[-1]) if ipp_str else None
            
            # Group slices by "series"
            series_key = f"{modality}_{seriesUID}"
            series[series_key].append((sampleName, filePath, inst, z, miscUID))
            modality_count[modality] += 1

        except Exception as e:
            print(f"Error processing metadata for {filePath}: {e}")

    return modality_count, series

def create_dataset_map(series, annotation):
  # create final dataset map
  final_dataset_map = {}

  annotated_files = 0
  delimiter = '_'

  for slice_element in series:
    sampleName, filepath, instanceNum, zCoord, _ = slice_element
    # returns path if DICOM file has corresponding XML annotation
    annotation_status = find_xml_for_slice(slice_element, annotation)

    final_dataset_map[filepath] = {
        'sampleID': sampleName,
        'instance_num': instanceNum,
        'z_coord': zCoord,
        'annotation_path': annotation_status,
        'is_annotated': (annotation_status is not None)
    }

    if annotation_status is not None:
      annotated_files += 1

  print(f"""Number of annotated files: {annotated_files} \nNumber of synchronized slices: {len(final_dataset_map)}\n""")

  return final_dataset_map

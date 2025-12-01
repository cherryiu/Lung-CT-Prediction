import argparse
import tensorflow as tf
from util.create_data import *
from util.helpers import *
from util.preprocessing import *
from util.model import *

def parse_args():
    parser = argparse.ArgumentParser('Data Preprocessing')

    parser.add_argument('--dicom-path', type=str,
                        help='path to the folder stored dicom files (.DCM)')
    parser.add_argument('--annotation-path', type=str,
                        help='path to the folder stored annotation files (.xml) or a path to a single annotation file')

    return parser.parse_args()

def main():
   args = parse_args()
   ## populate hyper-parameters
   params = {
      'data_dir': args.dicom_path,
      'annotations': args.annotation_path,
      'max_files': 100,  # remove later

      'train_ratio': 0.7,
      'val_ratio': 0.15,
      'test_ratio': 0.15,  

       'target_shape': (128, 128, 128, 1)    
   }

   _, series = create_metadata_sitk(params['data_dir'])

   #-------------- Preaparing the raw data --------------#
   # only select series that are CT scans
   filtered_series = {
      key : value 
      for key, value in series.items()
      if "CT" in key 
   }

   tensor_list = []
   mask_list = []

   for series_uid, series_arr in filtered_series.items():
      # sort best series by z coordinate if available, else by instance number
      series_arr.sort(key=sort_key)

      # create dataset map for one series
      series_dataset_map = create_dataset_map(series_arr, params['annotations'])

      # create final tensors for 3D image and corresponding mask 
      tensor, mask_volume = preprocess_data(series_dataset_map)

      # add to final list
      tensor_list.append(tensor)
      mask_list.append(mask_volume)

   dataset = tf.data.Dataset.from_tensor_slices((tensor_list, mask_list))

   #---------- Preparing the tensorflow dataset ---------#
   # Assuming 'tensor_list' and 'dataset' (the unbatched dataset) are available
   total_size = len(tensor_list)
   dataset, train_dataset, val_dataset, test_dataset = split_tf_dataset(params, dataset, total_size)
   
   # Prepare all three sets
   train_dataset_batched = prepare_dataset(train_dataset, params['batch_size'])
   val_dataset_batched = prepare_dataset(val_dataset, params['batch_size'])
   test_dataset_batched = prepare_dataset(test_dataset, params['batch_size'])


   #-------------- Create instance of CNN --------------#
   
   model = build_3d_model(params['target_shape'], params['num_classes'])
   compile_model(model)
   train_model(model, 
               params, 
               math.floor(params['train_ratio'] * total_size), 
               train_dataset_batched, test_dataset_batched, val_dataset_batched)






















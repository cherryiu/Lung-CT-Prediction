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

def run_model(params):

   _, series = create_metadata_sitk(params['data_dir'])

   #-------------- Preaparing the raw data --------------#
   # only select series that are CT scans
   filtered_series = {
      key : value 
      for key, value in series.items()
      if "CT" in key 
   }

   tensor_list = []
   label_list = []

   for series_uid, series_arr in filtered_series.items():
      # sort best series by z coordinate if available, else by instance number
      series_arr.sort(key=sort_key)

      # create dataset map for one series
      series_dataset_map = create_dataset_map(series_arr, params['annotations'])

      # create final tensors for 3D image and corresponding mask 
      tensor, label = preprocess_data(series_dataset_map)

      # --- NEW CHECK: ONLY APPEND IF THE DATA IS VALID ---
      if tensor is not None and label is not None:
         tensor_list.append(tensor)
         label_list.append(label)
      else:
         print(f"Skipping failed sample.")
      

   dataset = tf.data.Dataset.from_tensor_slices((tensor_list, label_list))

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

def get_paths():
   # Get DICOM Path
   while True:
      # Prompt user for the DICOM path
      dicom_path_str = input("Please enter the container path to the DICOM files (e.g., /data/dicom): ").strip()
      p = Path(dicom_path_str)
      print(f"--> dicom path inputted: {p}")

      # Validate existence
      if p.exists():
         break
      else:
         print(f"Path not found: {dicom_path_str}. Please verify the path and re-enter.")
      
   # Get Annotation Paths
   while True:
      # Prompt user for the Annotation path
      annotation_path_str = input("Please enter the container path to the Annotation XML files (e.g., /data/annotations): ").strip()
      p = Path(annotation_path_str)
      print(f"--> annotation path inputted: {p}")
      
      # Validate existence
      if p.exists():
         break
      else:
         print(f"Path not found: {annotation_path_str}. Please verify the path and re-enter.")
         
   return dicom_path_str, annotation_path_str

if __name__ == "__main__":
   args = parse_args()

   dicom_path = args.dicom_path
   annotation_path = args.annotation_path

   if dicom_path and annotation_path:
      print("Running in non-interactive mode.")
   else:
      print("Missing required paths:")
      dicom_path, annotation_path = get_paths()

   ## populate hyper-parameters
   params = {
      'data_dir': dicom_path,
      'annotations': annotation_path,

      'buffer_size': 2500,
      'batch_size': 4,
      'epochs': 15,
      'learning_rate': 0.0003,
      'img_size': 299,

      'train_ratio': 0.7,
      'val_ratio': 0.15,
      'test_ratio': 0.15,  

      'num_classes': 4,
      'target_shape': (64, 64, 64, 1)    
   }
   
   print("Running script.")
   run_model(params)





















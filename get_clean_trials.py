import pathlib
import os.path
import numpy as np

""" FILE NAMES """
# Interval of trials [onset, offset]
f_intervals = "trials.intervals.npy"
# Which trials were included [boolean]
f_included = "trials.included.npy"
# Timing of visual stimulation onset
f_visStim_times = "trials.visualStim_times.npy"
# Response choice: -1 for right, +1 for left, 0 for no-go
f_choice = "trials.response_choice.npy"
# Response time
f_response_times = "trials.response_times.npy"
# Feedback time 
f_feedback_times = "trials.feedback_times.npy"
# Feedback type:  -1 for white noise (miss), +1 for water reward (hit)
f_response_times = "trials.feedbackType.npy"

# Location of the project
project_path = pathlib.Path().absolute()


filenames = list([f_intervals, f_included, f_visStim_times, f_choice, f_response_times, f_feedback_times, f_response_times])


def load_trial_files (recording_name):
    data_path = os.path.join(project_path, 'data', recording_name)
    alldata = []
    for filename in filenames:
        file = os.path.join(data_path, filename)
        data = np.load(file)
        alldata.append(data)
        
    trials = {
        'interval': alldata[0],
        'included': alldata[1],
        'visStim_times': alldata[2],
        'choice': alldata[3],
        'response_time': alldata[4],
        'feedback_times': alldata[5],
        'response_times': alldata[6],
    }
    
    return trials
    

def clean_trials (recording_name):
#   Remove trials that were not included
#   INPUT: trial_intervals  - onset and offset timestamps of a N trials
#          included_trials - boolean +1 included, 0 not included
  trials = load_trial_files (recording_name)
    
  included_trials = trials['included']
  idx = np.where(included_trials == 1 )
  idx = idx[0]
  
  # Undate trials [dictionary]
  trials['interval'] = trials['interval'][idx,:]
  trials['included'] = trials['included'][idx,:]
  trials['visStim_times'] = trials['visStim_times'][idx,:]
  trials['choice'] = trials['choice'][idx,:]
  trials['response_time'] = trials['response_time'][idx,:]
  trials['feedback_times'] = trials['feedback_times'][idx,:]
  trials['response_times'] = trials['response_times'][idx,:]
      
  
  return trials



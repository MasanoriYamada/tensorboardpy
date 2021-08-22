import os
import glob
import pprint
import traceback
import pandas as pd
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path: str, ignore_path: bool = False) -> pd.DataFrame:
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        # parse scalars
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
        try:
            # parse hparams
            hparams_b = event_acc.summary_metadata['_hparams_/session_start_info'].plugin_data.content
            hparams_contents = plugin_data_pb2.HParamsPluginData.FromString(hparams_b).session_start_info.hparams
            hparams = {}
            for key in hparams_contents:
                v = eval(str(hparams_contents[key]).split(':')[-1])
                hparams[key] = v
        except:
            print("Hparams data in event file possibly corrupt: {}".format(path))
        # add hparams to scalars
        for key in hparams:
            runlog_data[key] = hparams[key]
        if not ignore_path:
            runlog_data['path'] = path
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths, ignore_path: bool):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path, ignore_path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def logdir(logdir: str, ignore_path: bool = False):
    """This is a enhanced version of
    https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
    This script exctracts variables from all logs with hparams from tensorflow event
    files ("event*"),
    """
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(os.path.join(logdir, '**', 'event*'), recursive=True)
    elif os.path.isfile(logdir):
        event_paths = [logdir]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir
            )
        )
    # Call & append
    if event_paths:
        print('events files loading...')
        pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths, ignore_path)
        return all_logs
    else:
        print("No event paths have been found.")

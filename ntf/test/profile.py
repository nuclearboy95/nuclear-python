import tensorflow as tf
from tensorflow.python.client import timeline


__all__ = ['get_profiler', 'save_profile_result']


def get_profiler():
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    return {
        'options': options,
        'run_metadata': run_metadata
    }


def save_profile_result(fpath, profiler):
    run_metadata = profiler['run_metadata']
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(fpath, 'w') as f:
        f.write(chrome_trace)

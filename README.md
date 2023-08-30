# Recording Export Utility

This Python script extracts gaze, IMU, and template data from raw Invisible recordings.
Gaze and IMU data are exported as CSV files. Template data as json.

In addition, it performs post-hoc blink and fixation detection, and exports the data
as CSV.

```
Usage: pl-rec-export [OPTIONS] [RECORDINGS]...

Options:
  -e, --export-folder TEXT      Relative export path  [default:
                                (<recording>/export)]
  -f, --force                   Overwrite an existing export
  -v, --verbose                 Show more log messages (repeat for even more)
  --blinks / --no-blinks
  --fixations / --no-fixations
  --help                        Show this message and exit.
```

## Caveats

1. Requires untouched Pupil Invisible recordings. Does not work if they were opened in
   Pupil Player
2. Does not calculate IMU `roll [deg]`/`pitch [deg]` (yet)
3. Does not handle incomplete recordings (missing/corrupted files)
4. Locally detected fixations differ from Pupil Cloud exported fixations due to not
   having access to the full 200 Hz gaze data

## Installation

1. Install Python 3.8 or higher
2. ```bash
   pip install -e https://github.com/pupil-labs/pl-rec-export#pl-rec-export`
   ```
3. ```bash
   pl-rec-export /path/to/rec
   ```

**Note:**  The [`xgboost`](https://pypi.org/project/xgboost/) Python dependency might
require you to install further non-Python dependencies. If this is the case, it will
tell you how during the install or when running the script.

## Changelog
### 1.0.2

- Python 3.8 compatibility


### 1.0.1

- Show version in cli


### 1.0.0

- Initial release

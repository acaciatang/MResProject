python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test1.MP4
python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test2.MP4
python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test3.MP4
python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test4.MP4
python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test5.MP4
python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test6.MP4
python3 ../../../TrackBEETag/Sandbox/TrackingInPython/PythonTracking.py test7.MP4

cp test1_raw.csv test1.csv
cp test2_raw.csv test2.csv
cp test3_raw.csv test3.csv
cp test4_raw.csv test4.csv
cp test5_raw.csv test5.csv
cp test6_raw.csv test6.csv
cp test7_raw.csv test7.csv

python3 ../../../TrackBEETag/Code/DrawTracks.py test1.MP4
python3 ../../../TrackBEETag/Code/DrawTracks.py test2.MP4
python3 ../../../TrackBEETag/Code/DrawTracks.py test3.MP4
python3 ../../../TrackBEETag/Code/DrawTracks.py test4.MP4
python3 ../../../TrackBEETag/Code/DrawTracks.py test5.MP4
python3 ../../../TrackBEETag/Code/DrawTracks.py test6.MP4
python3 ../../../TrackBEETag/Code/DrawTracks.py test7.MP4
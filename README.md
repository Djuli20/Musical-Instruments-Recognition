# Musical-Instruments-recognition
CNN model trained with TensorFlow that is able to recognize what musical instrument is playing
The full documentation can be found in the file Musical Instrument Documentation.docx
In the Dataset.rar can be found the dataset created by me used for training the TensorFlow model. The dataset is actually a CSV file with the 1000 FFT values per instrument in the interval 0-5000Hz. For the full WAV dataset mail me.
The Model10ep_2^17_1000benzitabel.keras model is actually the best model I was able to train. Initially I was considering 200 bands, but upgrading to 1000 bands drastically improved the accuracy.




import wave
import struct
import math
import cmath
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("Model10ep_2^17_1000benzitabel.keras")

# Your known instrument labels (index matches the encoded label)
instrument_labels = ['Acordeon', 'Trompeta', 'Vioara']

def compute_fft(x):
    """
    Calculează Transformata Fourier Rapidă (FFT) a secvenței de intrare folosind algoritmul Cooley-Tukey.
    
    Argumente:
        x: Un tuplu sau o listă de numere complexe
        
    Returnează:
        Un tuplu conținând FFT-ul secvenței de intrare
    """
    N = len(x)
    if N <= 1:
        return x
    
    even = compute_fft(tuple(x[i] for i in range(0, N, 2)))
    odd = compute_fft(tuple(x[i] for i in range(1, N, 2)))
    
    result = [0] * N
    for k in range(N // 2):
        t = cmath.exp(-2j * math.pi * k / N) * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    
    return tuple(result)

def get_frequency_bands(fft_magnitude, freqs, min_freq=0, max_freq=5000, num_bands=1000):
    """
    Împarte rezultatele FFT pe benzi de frecvență și calculează magnitudinea medie pentru fiecare bandă.
    
    Argumente:
        fft_magnitude: Tuplu cu valorile magnitudinii FFT
        freqs: Tuple cu frecvențele corespunzătoare magnitudinilor FFT
        min_freq: Frecvența minimă de luat în considerare (Hz)
        max_freq: Frecvența maximă de luat în considerare (Hz)
        num_bands: Numărul de benzi de frecvență
        
    Returnează:
        Tuplu de forma (medii_benzi, frecvențe_centrale_benzi)
    """
    fft_magnitude_list = list(fft_magnitude)
    freqs_list = list(freqs)
    band_width = (max_freq - min_freq) / num_bands
    band_averages = []
    band_center_freqs = []

    for i in range(num_bands):
        band_min = min_freq + i * band_width
        band_max = band_min + band_width
        band_indices = [j for j, freq in enumerate(freqs_list) if band_min <= freq < band_max]
        if band_indices:
            band_magnitude_values = [fft_magnitude_list[j] for j in band_indices]
            band_avg = sum(band_magnitude_values) / len(band_magnitude_values)
        else:
            band_avg = 0.0
        band_center = band_min + band_width / 2
        band_averages.append(band_avg)
        band_center_freqs.append(band_center)
    
    return band_averages, band_center_freqs

def predict_instrument(input_features, threshold=0.4):
    """
    input_features: list or numpy array of band values (same length as training data features)
    threshold: minimum confidence required to accept a prediction
    """
    # Convert and reshape input
    input_array = np.array(input_features, dtype=np.float32).reshape(1, -1, 1)

    # Predict
    prediction = model.predict(input_array)
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)

    # Check if prediction passes threshold
    if confidence >= threshold:
        predicted_label = instrument_labels[predicted_index]
        return f"🎵 Predicted: {predicted_label} (Confidence: {confidence:.2f})"
    else:
        return f"❓ Unknown Instrument (Confidence: {confidence:.2f})"

def analyze_and_predict_instrument(wav_file_path, min_freq=0, max_freq=5000, num_bands=1000, threshold=0.4):
    """
    Analyze an audio file, compute its FFT, extract frequency bands, and predict the instrument.
    
    Arguments:
        wav_file_path: Path to the WAV file
        min_freq: Minimum frequency to consider (Hz)
        max_freq: Maximum frequency to consider (Hz)
        num_bands: Number of frequency bands to extract
        threshold: Confidence threshold for instrument prediction
        
    Returns:
        Tuple containing (prediction_result, band_averages)
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("WAV file must be mono (1 channel)")
        sample_rate = wav_file.getframerate()
        if wav_file.getsampwidth() != 2:
            raise ValueError("WAV file must be 16-bit")
        
        # We need 131072 (2^17) samples for consistency with the trained model
        target_samples = 131072
        frames_to_read = target_samples
        print(f"Reading {frames_to_read} samples")
        frames = wav_file.readframes(frames_to_read)
        actual_frames_read = len(frames) // 2
        if actual_frames_read < frames_to_read:
            print(f"Warning: Only {actual_frames_read} samples available in file. Padding to {frames_to_read}.")
            frames += b'\x00\x00' * (frames_to_read - actual_frames_read)
        
        # Unpack audio data and normalize
        audio_data_tuple = struct.unpack(f'{frames_to_read}h', frames[:frames_to_read*2])
        normalized_audio_data = tuple(sample / 32768.0 for sample in audio_data_tuple)
        complex_data = tuple(complex(x, 0) for x in normalized_audio_data)
        
        # Compute FFT
        fft_result = compute_fft(complex_data)
        fft_magnitude = tuple(abs(x) for x in fft_result)
        fft_magnitude = fft_magnitude[:target_samples//2]
        freqs = tuple(i * sample_rate / target_samples for i in range(target_samples//2))
        
        # Get frequency bands
        band_averages, _ = get_frequency_bands(
            fft_magnitude, freqs, min_freq, max_freq, num_bands
        )
        
        # Make prediction using the model
        prediction_result = predict_instrument(band_averages, threshold)
        
        return prediction_result, band_averages

# Example usage
if __name__ == "__main__":
    # Analyze a WAV file and predict the instrument
    wav_file_path = "ForTest/Trompeta/Test SoloTrumpet_C3.wav"
    prediction, band_averages = analyze_and_predict_instrument(
        wav_file_path, 
        min_freq=0, 
        max_freq=5000, 
        num_bands=1000,
        threshold=0.4
    )
    print(f"Audio file analysis result: {prediction}")
    #print(f"Computed band values: {band_averages}")

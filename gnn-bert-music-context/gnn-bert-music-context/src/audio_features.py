"""
Audio Feature Extraction and Segmentation Pipeline
Handles resampling, log-mel spectrograms, chroma, MFCC, and temporal windowing.
"""

import numpy as np
import librosa
from typing import Tuple, List, Dict, Optional


class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_chroma: int = 12,
        n_mfcc: int = 20,
        hop_length: int = 512,
        n_fft: int = 2048,
        segment_duration: float = 3.0,
        target_duration: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.segment_duration = segment_duration
        self.target_duration = target_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.target_samples = int(target_duration * sample_rate)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Loads an audio file, resamples to self.sample_rate, normalizes,
        and pads or trims to target_duration.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}")

        # Pad or trim to target length
        if len(y) < self.target_samples:
            y = np.pad(y, (0, self.target_samples - len(y)), mode="constant")
        else:
            y = y[: self.target_samples]

        # Normalize per track (zero mean, unit variance or max scaling)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        return y

    def extract_log_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Computes 128-bin log-mel spectrogram.
        Returns: (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to [0, 1] range for CNN
        norm_mel = (log_mel_spec - log_mel_spec.min()) / (
            log_mel_spec.max() - log_mel_spec.min() + 1e-8
        )
        return norm_mel

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Computes 12-bin chroma features (pitch class profiles).
        Returns: (12, time_steps)
        """
        chroma = librosa.feature.chroma_cens(
            y=y, sr=self.sample_rate, hop_length=self.hop_length, n_chroma=self.n_chroma
        )
        return chroma

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """
        Computes MFCCs.
        Returns: (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )
        return mfcc

    def segment_audio(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Splits audio into fixed-length windows (e.g., 3s segments = 10 segments for 30s track).
        """
        segments = []
        num_segments = int(np.ceil(len(y) / self.segment_samples))
        for i in range(num_segments):
            start = i * self.segment_samples
            end = min(start + self.segment_samples, len(y))
            seg = y[start:end]
            if len(seg) < self.segment_samples:
                seg = np.pad(seg, (0, self.segment_samples - len(seg)), mode="constant")
            segments.append(seg)
        return segments

    def extract_segment_features(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Extracts pooled feature vectors h_i^(0) for each segment.
        Combines mean Chroma (12) + mean MFCC (20) + mean Spectral Contrast (7).
        Total node dimension = 12 + 20 + 7 = 39.
        Returns: (num_segments, feature_dim)
        """
        node_features = []
        for seg in segments:
            chroma = librosa.feature.chroma_cens(
                y=seg, sr=self.sample_rate, hop_length=self.hop_length, n_chroma=self.n_chroma
            )
            chroma_mean = np.mean(chroma, axis=1)

            mfcc = librosa.feature.mfcc(
                y=seg, sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length
            )
            mfcc_mean = np.mean(mfcc, axis=1)

            contrast = librosa.feature.spectral_contrast(
                y=seg, sr=self.sample_rate, hop_length=self.hop_length
            )
            contrast_mean = np.mean(contrast, axis=1)

            feat = np.concatenate([chroma_mean, mfcc_mean, contrast_mean])
            node_features.append(feat)

        node_features = np.array(node_features, dtype=np.float32)
        # Normalize features across nodes
        norm = np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
        node_features = node_features / norm
        return node_features

    def process_track(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Full feature extraction for a single track:
        Returns raw audio, log-mel spectrogram, and node feature matrix.
        """
        y = self.load_audio(audio_path)
        mel_spec = self.extract_log_mel_spectrogram(y)
        segments = self.segment_audio(y)
        node_feats = self.extract_segment_features(segments)

        return {
            "waveform": y,
            "mel_spectrogram": mel_spec,
            "node_features": node_feats,
            "num_nodes": len(segments),
        }

"""
TFRecord Data Manager for U-Net Training
=========================================

Comprehensive TFRecord manager for converting numpy training data to TFRecord format
with support for:
- Multiple data types (numpy arrays, scalars, strings)
- File chunking for large datasets
- Efficient serialization/deserialization
- Multi-GPU data sharding
- U-Net mask-aerial image pairs

Author: Claude Code
Date: 2026-02-11
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TFRecordConfig:
    """Configuration for TFRecord dataset."""

    # File settings
    output_dir: str = './tfrecords'
    file_prefix: str = 'litho_data'
    samples_per_file: int = 1000  # Chunk size

    # Data specifications
    image_height: int = 512
    image_width: int = 512
    mask_channels: int = 1
    aerial_channels: int = 1

    # Compression
    compression_type: str = 'GZIP'  # 'GZIP', 'ZLIB', or '' (no compression)

    # Metadata
    description: str = 'Lithography training data'
    version: str = '1.0'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TFRecordConfig':
        """Create from dictionary."""
        return cls(**d)

    def save(self, path: str) -> None:
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TFRecordConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class TFRecordWriter:
    """
    Write training data to TFRecord files with chunking support.

    Supports multiple data types per sample:
    - Numpy arrays (images, masks)
    - Scalars (float, int)
    - Strings (labels, metadata)
    """

    def __init__(self, config: TFRecordConfig):
        """
        Initialize TFRecord writer.

        Args:
            config: TFRecord configuration
        """
        self.config = config

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Statistics
        self.total_samples = 0
        self.num_files = 0

    def write_dataset(
        self,
        masks: np.ndarray,
        aerials: np.ndarray,
        metadata: Optional[Dict[str, Union[np.ndarray, float, int, str]]] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Write complete dataset to TFRecord files with chunking.

        Args:
            masks: Mask images [N, H, W] or [N, H, W, C]
            aerials: Aerial images [N, H, W] or [N, H, W, C]
            metadata: Optional per-sample metadata dict
            verbose: Print progress

        Returns:
            List of created TFRecord file paths
        """
        n_samples = len(masks)

        if len(aerials) != n_samples:
            raise ValueError(f"Masks ({len(masks)}) and aerials ({len(aerials)}) must have same length")

        if verbose:
            print(f"Writing {n_samples} samples to TFRecord...")
            print(f"  Output directory: {self.config.output_dir}")
            print(f"  Samples per file: {self.config.samples_per_file}")

        # Prepare metadata
        if metadata is None:
            metadata = {}

        # Validate metadata length if provided
        for key, value in metadata.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) != n_samples:
                    raise ValueError(f"Metadata '{key}' length ({len(value)}) != n_samples ({n_samples})")

        # Calculate number of chunks
        samples_per_file = self.config.samples_per_file
        num_chunks = (n_samples + samples_per_file - 1) // samples_per_file

        created_files = []

        # Write chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * samples_per_file
            end_idx = min(start_idx + samples_per_file, n_samples)

            # Create chunk filename
            filename = f"{self.config.file_prefix}_{chunk_idx:04d}.tfrecord"
            filepath = os.path.join(self.config.output_dir, filename)

            # Extract chunk data
            masks_chunk = masks[start_idx:end_idx]
            aerials_chunk = aerials[start_idx:end_idx]

            # Extract metadata chunk
            metadata_chunk = {}
            for key, value in metadata.items():
                if isinstance(value, (list, np.ndarray)):
                    metadata_chunk[key] = value[start_idx:end_idx]
                else:
                    # Scalar metadata applies to all samples
                    metadata_chunk[key] = value

            # Write chunk
            n_written = self._write_chunk(
                filepath,
                masks_chunk,
                aerials_chunk,
                metadata_chunk,
                verbose=verbose
            )

            created_files.append(filepath)
            self.num_files += 1
            self.total_samples += n_written

            if verbose:
                progress = 100.0 * end_idx / n_samples
                print(f"  Chunk {chunk_idx+1}/{num_chunks}: {n_written} samples ({progress:.1f}%)")

        # Save configuration
        config_path = os.path.join(self.config.output_dir, 'config.json')
        self.config.save(config_path)

        # Save dataset info
        self._save_dataset_info(n_samples, created_files)

        if verbose:
            print(f"✓ Written {self.total_samples} samples to {self.num_files} files")
            print(f"  Configuration saved to {config_path}")

        return created_files

    def _write_chunk(
        self,
        filepath: str,
        masks: np.ndarray,
        aerials: np.ndarray,
        metadata: Dict[str, Any],
        verbose: bool = False
    ) -> int:
        """
        Write a single chunk to TFRecord file.

        Args:
            filepath: Output TFRecord file path
            masks: Mask images for this chunk
            aerials: Aerial images for this chunk
            metadata: Metadata for this chunk
            verbose: Print debug info

        Returns:
            Number of samples written
        """
        # Set compression
        options = None
        if self.config.compression_type:
            options = tf.io.TFRecordOptions(
                compression_type=self.config.compression_type
            )

        n_written = 0

        with tf.io.TFRecordWriter(filepath, options=options) as writer:
            for i in range(len(masks)):
                # Extract single sample
                mask = masks[i]
                aerial = aerials[i]

                # Extract metadata for this sample
                sample_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (list, np.ndarray)):
                        sample_metadata[key] = value[i]
                    else:
                        sample_metadata[key] = value

                # Create feature dict
                features = self._create_features(mask, aerial, sample_metadata)

                # Create Example
                example = tf.train.Example(
                    features=tf.train.Features(feature=features)
                )

                # Write to file
                writer.write(example.SerializeToString())
                n_written += 1

        return n_written

    def _create_features(
        self,
        mask: np.ndarray,
        aerial: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, tf.train.Feature]:
        """
        Create TFRecord features from data.

        Args:
            mask: Single mask image
            aerial: Single aerial image
            metadata: Metadata for this sample

        Returns:
            Dictionary of TFRecord features
        """
        features = {}

        # Add mask (numpy array)
        features['mask'] = self._numpy_feature(mask)
        features['mask_shape'] = self._int64_list_feature(mask.shape)

        # Add aerial (numpy array)
        features['aerial'] = self._numpy_feature(aerial)
        features['aerial_shape'] = self._int64_list_feature(aerial.shape)

        # Add metadata
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                # Numpy array
                features[f'meta_{key}'] = self._numpy_feature(value)
                features[f'meta_{key}_shape'] = self._int64_list_feature(value.shape)
                features[f'meta_{key}_dtype'] = self._bytes_feature(str(value.dtype).encode())
            elif isinstance(value, float):
                # Float scalar
                features[f'meta_{key}'] = self._float_feature(value)
            elif isinstance(value, (int, np.integer)):
                # Int scalar
                features[f'meta_{key}'] = self._int64_feature(int(value))
            elif isinstance(value, str):
                # String
                features[f'meta_{key}'] = self._bytes_feature(value.encode())
            elif isinstance(value, bytes):
                # Bytes
                features[f'meta_{key}'] = self._bytes_feature(value)
            else:
                print(f"Warning: Unsupported metadata type for '{key}': {type(value)}")

        return features

    @staticmethod
    def _numpy_feature(array: np.ndarray) -> tf.train.Feature:
        """Convert numpy array to bytes feature."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[array.tobytes()])
        )

    @staticmethod
    def _bytes_feature(value: bytes) -> tf.train.Feature:
        """Convert bytes to feature."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value])
        )

    @staticmethod
    def _float_feature(value: float) -> tf.train.Feature:
        """Convert float to feature."""
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )

    @staticmethod
    def _int64_feature(value: int) -> tf.train.Feature:
        """Convert int to feature."""
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value])
        )

    @staticmethod
    def _int64_list_feature(values: Union[List[int], Tuple[int, ...]]) -> tf.train.Feature:
        """Convert list of ints to feature."""
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values))
        )

    def _save_dataset_info(self, n_samples: int, files: List[str]) -> None:
        """Save dataset information."""
        info = {
            'total_samples': n_samples,
            'num_files': len(files),
            'samples_per_file': self.config.samples_per_file,
            'files': [os.path.basename(f) for f in files],
            'config': self.config.to_dict(),
        }

        info_path = os.path.join(self.config.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)


class TFRecordReader:
    """
    Read training data from TFRecord files.

    Supports:
    - Automatic data type parsing
    - Multi-GPU data sharding
    - Efficient prefetching and batching
    """

    def __init__(self, tfrecord_dir: str, config: Optional[TFRecordConfig] = None):
        """
        Initialize TFRecord reader.

        Args:
            tfrecord_dir: Directory containing TFRecord files
            config: TFRecord configuration (will load from dir if None)
        """
        self.tfrecord_dir = tfrecord_dir

        # Load configuration
        if config is None:
            config_path = os.path.join(tfrecord_dir, 'config.json')
            if os.path.exists(config_path):
                self.config = TFRecordConfig.load(config_path)
            else:
                print("Warning: No config.json found, using defaults")
                self.config = TFRecordConfig()
        else:
            self.config = config

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()

        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    def _find_tfrecord_files(self) -> List[str]:
        """Find all TFRecord files in directory."""
        pattern = os.path.join(self.tfrecord_dir, f"{self.config.file_prefix}_*.tfrecord")
        files = tf.io.gfile.glob(pattern)
        return sorted(files)

    def create_dataset(
        self,
        batch_size: int,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        num_parallel_reads: int = tf.data.AUTOTUNE,
        prefetch_size: int = tf.data.AUTOTUNE,
        repeat: bool = True,
        drop_remainder: bool = True,
        deterministic: bool = False,
        shard_for_multi_gpu: bool = False,
        **kwargs
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from TFRecord files.

        Args:
            batch_size: Batch size
            shuffle: Shuffle data
            shuffle_buffer_size: Size of shuffle buffer
            num_parallel_reads: Number of files to read in parallel
            prefetch_size: Prefetch buffer size
            repeat: Repeat dataset indefinitely
            drop_remainder: Drop last incomplete batch
            deterministic: Deterministic ordering (slower)
            shard_for_multi_gpu: Auto-shard for distributed training
            **kwargs: Additional metadata keys to parse

        Returns:
            tf.data.Dataset yielding (features_dict, labels) tuples
        """
        # Create dataset from files
        dataset = tf.data.Dataset.from_tensor_slices(self.tfrecord_files)

        # Shard for multi-GPU if requested
        if shard_for_multi_gpu:
            # This is critical for multi-GPU training!
            # Each replica gets a different shard of the data
            dataset = dataset.shard(
                num_shards=tf.distribute.get_strategy().num_replicas_in_sync,
                index=tf.distribute.get_replica_context().replica_id_in_sync_group
                if tf.distribute.get_replica_context() else 0
            )

        # Shuffle files
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.tfrecord_files))

        # Interleave reading from multiple files
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(
                filename,
                compression_type=self.config.compression_type
            ),
            cycle_length=num_parallel_reads,
            num_parallel_calls=num_parallel_reads,
            deterministic=deterministic
        )

        # Shuffle records
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                reshuffle_each_iteration=True
            )

        # Parse examples
        parse_fn = self._create_parse_function(**kwargs)
        dataset = dataset.map(
            parse_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch
        dataset = dataset.batch(
            batch_size,
            drop_remainder=drop_remainder
        )

        # Repeat
        if repeat:
            dataset = dataset.repeat()

        # Prefetch
        dataset = dataset.prefetch(prefetch_size)

        return dataset

    def _create_parse_function(self, **metadata_keys):
        """
        Create parsing function for TFRecord examples.

        Args:
            **metadata_keys: Expected metadata keys and their types
                             e.g., wavelength='float', pattern_id='int'

        Returns:
            Parse function
        """
        def parse_example(serialized_example):
            """Parse single TFRecord example."""
            # Define feature description
            feature_description = {
                # Required: mask and aerial
                'mask': tf.io.FixedLenFeature([], tf.string),
                'mask_shape': tf.io.FixedLenFeature([3], tf.int64),
                'aerial': tf.io.FixedLenFeature([], tf.string),
                'aerial_shape': tf.io.FixedLenFeature([3], tf.int64),
            }

            # Add optional metadata features
            for key, dtype in metadata_keys.items():
                if dtype == 'float':
                    feature_description[f'meta_{key}'] = tf.io.FixedLenFeature([], tf.float32)
                elif dtype == 'int':
                    feature_description[f'meta_{key}'] = tf.io.FixedLenFeature([], tf.int64)
                elif dtype == 'string':
                    feature_description[f'meta_{key}'] = tf.io.FixedLenFeature([], tf.string)
                elif dtype == 'numpy':
                    feature_description[f'meta_{key}'] = tf.io.FixedLenFeature([], tf.string)
                    feature_description[f'meta_{key}_shape'] = tf.io.VarLenFeature(tf.int64)
                    feature_description[f'meta_{key}_dtype'] = tf.io.FixedLenFeature([], tf.string)

            # Parse
            parsed = tf.io.parse_single_example(serialized_example, feature_description)

            # Decode mask
            mask = tf.io.decode_raw(parsed['mask'], tf.float32)
            mask_shape = tf.cast(parsed['mask_shape'], tf.int32)
            mask = tf.reshape(mask, mask_shape)

            # Decode aerial
            aerial = tf.io.decode_raw(parsed['aerial'], tf.float32)
            aerial_shape = tf.cast(parsed['aerial_shape'], tf.int32)
            aerial = tf.reshape(aerial, aerial_shape)

            # Prepare output
            features = {
                'mask': mask,
                'aerial': aerial,
            }

            # Add metadata
            for key, dtype in metadata_keys.items():
                meta_key = f'meta_{key}'
                if dtype == 'numpy':
                    # Decode numpy array
                    dtype_str = parsed[f'{meta_key}_dtype']
                    # For simplicity, assume float32
                    arr = tf.io.decode_raw(parsed[meta_key], tf.float32)
                    shape = tf.sparse.to_dense(parsed[f'{meta_key}_shape'])
                    arr = tf.reshape(arr, tf.cast(shape, tf.int32))
                    features[key] = arr
                else:
                    features[key] = parsed[meta_key]

            # Return (input, label) format for training
            # For U-Net: input=mask, label=aerial
            return features['mask'], features['aerial']

        return parse_example

    def get_sample(self, index: int = 0) -> Dict[str, np.ndarray]:
        """
        Get a single sample for inspection.

        Args:
            index: Sample index

        Returns:
            Dictionary with parsed data
        """
        dataset = self.create_dataset(
            batch_size=1,
            shuffle=False,
            repeat=False,
            drop_remainder=False
        )

        # Skip to index
        dataset = dataset.skip(index)

        # Get first sample
        for mask, aerial in dataset.take(1):
            return {
                'mask': mask[0].numpy(),
                'aerial': aerial[0].numpy(),
            }

        raise IndexError(f"Sample {index} not found")

    def info(self) -> Dict[str, Any]:
        """Get dataset information."""
        info_path = os.path.join(self.tfrecord_dir, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'num_files': len(self.tfrecord_files),
                'files': [os.path.basename(f) for f in self.tfrecord_files],
            }


def convert_numpy_to_tfrecord(
    masks_path: str,
    aerials_path: str,
    output_dir: str,
    samples_per_file: int = 1000,
    compression: str = 'GZIP',
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> List[str]:
    """
    Convert numpy arrays to TFRecord format.

    Args:
        masks_path: Path to masks numpy file (.npy or .npz)
        aerials_path: Path to aerials numpy file (.npy or .npz)
        output_dir: Output directory for TFRecord files
        samples_per_file: Samples per TFRecord file
        compression: Compression type ('GZIP', 'ZLIB', or '')
        metadata: Optional metadata dictionary
        verbose: Print progress

    Returns:
        List of created TFRecord file paths
    """
    # Load numpy data
    if verbose:
        print(f"Loading data from:")
        print(f"  Masks: {masks_path}")
        print(f"  Aerials: {aerials_path}")

    # Load masks
    if masks_path.endswith('.npz'):
        masks_data = np.load(masks_path)
        masks = masks_data['arr_0'] if 'arr_0' in masks_data else masks_data['masks']
    else:
        masks = np.load(masks_path)

    # Load aerials
    if aerials_path.endswith('.npz'):
        aerials_data = np.load(aerials_path)
        aerials = aerials_data['arr_0'] if 'arr_0' in aerials_data else aerials_data['aerials']
    else:
        aerials = np.load(aerials_path)

    if verbose:
        print(f"  Masks shape: {masks.shape}")
        print(f"  Aerials shape: {aerials.shape}")

    # Ensure 3D or 4D
    if masks.ndim == 3:
        masks = masks[..., np.newaxis]
    if aerials.ndim == 3:
        aerials = aerials[..., np.newaxis]

    # Create config
    config = TFRecordConfig(
        output_dir=output_dir,
        samples_per_file=samples_per_file,
        image_height=masks.shape[1],
        image_width=masks.shape[2],
        mask_channels=masks.shape[3],
        aerial_channels=aerials.shape[3],
        compression_type=compression,
    )

    # Create writer
    writer = TFRecordWriter(config)

    # Write dataset
    created_files = writer.write_dataset(
        masks=masks,
        aerials=aerials,
        metadata=metadata,
        verbose=verbose
    )

    return created_files


# Example usage and testing
if __name__ == '__main__':
    print("="*70)
    print("TFRecord Data Manager - Example Usage")
    print("="*70)

    # Generate synthetic data for testing
    print("\n1. Generating synthetic test data...")
    n_samples = 150
    h, w = 256, 256

    masks = np.random.rand(n_samples, h, w).astype(np.float32)
    aerials = np.random.rand(n_samples, h, w).astype(np.float32)

    # Add some metadata
    metadata = {
        'wavelength': np.random.uniform(190, 195, n_samples).astype(np.float32),
        'na': np.random.uniform(1.3, 1.4, n_samples).astype(np.float32),
        'sample_id': list(range(n_samples)),
        'pattern_type': [f'pattern_{i%3}' for i in range(n_samples)],
    }

    print(f"  Created {n_samples} samples of size {h}×{w}")

    # Write to TFRecord
    print("\n2. Writing to TFRecord...")
    config = TFRecordConfig(
        output_dir='./test_tfrecords',
        samples_per_file=50,  # 3 files
        compression_type='GZIP'
    )

    writer = TFRecordWriter(config)
    files = writer.write_dataset(masks, aerials, metadata, verbose=True)

    # Read back
    print("\n3. Reading from TFRecord...")
    reader = TFRecordReader('./test_tfrecords')

    print(f"  Found {len(reader.tfrecord_files)} TFRecord files")
    print(f"  Dataset info: {reader.info()}")

    # Create dataset
    print("\n4. Creating tf.data.Dataset...")
    dataset = reader.create_dataset(
        batch_size=16,
        shuffle=True,
        repeat=False,
        wavelength='float',
        na='float',
        sample_id='int'
    )

    # Test iteration
    print("\n5. Testing iteration...")
    for i, (mask_batch, aerial_batch) in enumerate(dataset.take(3)):
        print(f"  Batch {i+1}: mask {mask_batch.shape}, aerial {aerial_batch.shape}")

    print("\n✓ TFRecord manager test complete!")
